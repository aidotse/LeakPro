"""Background audit worker — called from FastAPI via ThreadPoolExecutor."""

from __future__ import annotations

import asyncio
import logging
import queue
import sys
import threading
import uuid
import yaml
from pathlib import Path
from typing import Any

logger = logging.getLogger("leakpro.webapp.worker")


# ---------------------------------------------------------------------------
# Log capture
# ---------------------------------------------------------------------------

class _QueueHandler(logging.Handler):
    def __init__(self, q: queue.Queue) -> None:
        super().__init__()
        self._q = q

    def emit(self, record: logging.LogRecord) -> None:
        self._q.put(self.format(record))


# ---------------------------------------------------------------------------
# Default handler — used when no custom handler.py is uploaded
# ---------------------------------------------------------------------------

_LEAKPRO_ROOT = Path(__file__).parents[3]  # /home/.../LeakPro

# Deferred import so worker can be imported without torch being on the path
def _build_default_handler():
    import torch
    from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
    from leakpro.schemas import EvalOutput, TrainingOutput
    from tqdm import tqdm

    class _DefaultWebappHandler(AbstractInputHandler):
        class UserDataset(AbstractInputHandler.UserDataset):
            def __init__(self, data, targets, **kwargs):
                self.data = data if isinstance(data, torch.Tensor) else torch.tensor(data).float()
                self.targets = targets if isinstance(targets, torch.Tensor) else torch.tensor(targets).long()
            def __len__(self): return len(self.targets)
            def __getitem__(self, idx): return self.data[idx], self.targets[idx]

        def train(self, dataloader, model, criterion, optimizer, epochs=None, **kwargs):
            if epochs is None:
                raise ValueError("epochs must be provided")
            # Binary jobs carry BCEWithLogitsLoss in their metadata (single-logit head)
            binary = isinstance(criterion, torch.nn.BCEWithLogitsLoss)
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(dev)
            accuracy_history, loss_history = [], []
            for epoch in range(epochs):
                model.train()
                train_loss, train_acc, total = 0.0, 0.0, 0
                for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                    labels = labels.float().view(-1) if binary else labels.long().view(-1)
                    inputs, labels = inputs.to(dev), labels.to(dev)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    if binary:
                        outputs = outputs.squeeze(1)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    preds = (outputs.sigmoid() > 0.5).float() if binary else outputs.argmax(1).float()
                    train_acc += preds.eq(labels).sum().item()
                    train_loss += loss.item() * labels.size(0)
                    total += labels.size(0)
                accuracy_history.append(train_acc / total)
                loss_history.append(train_loss / total)
            model.to("cpu")
            metrics = EvalOutput(
                accuracy=accuracy_history[-1], loss=loss_history[-1],
                extra={"accuracy_history": accuracy_history, "loss_history": loss_history},
            )
            return TrainingOutput(model=model, metrics=metrics)

        def eval(self, loader, model, criterion, device=None):
            binary = isinstance(criterion, torch.nn.BCEWithLogitsLoss)
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(dev)
            model.eval()
            loss, acc, total = 0.0, 0.0, 0
            with torch.no_grad():
                for data, target in loader:
                    target = (target.float() if binary else target.long()).view(-1).to(dev)
                    data = data.to(dev)
                    output = model(data)
                    if binary:
                        output = output.squeeze(1)
                    loss += criterion(output, target).item() * target.size(0)
                    preds = (output.sigmoid() > 0.5).float() if binary else output.argmax(1).float()
                    acc += preds.eq(target).sum().item()
                    total += target.size(0)
            model.to("cpu")
            return EvalOutput(
                accuracy=acc / total if total else 0.0,
                loss=loss / total if total else 0.0,
            )

    return _DefaultWebappHandler


def _get_handler_cls(job_dir: Path, preset: str | None):
    """Return the handler class to pass to LeakPro."""
    import importlib.util
    import inspect

    DefaultHandler = _build_default_handler()

    handler_py = job_dir / "handler.py"
    if not handler_py.exists():
        return DefaultHandler

    # Custom handler.py has train/eval but no UserDataset — graft UserDataset from default
    spec = importlib.util.spec_from_file_location("_custom_handler", handler_py)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(handler_py.parent))
    spec.loader.exec_module(mod)
    custom_cls = next(
        (obj for name in dir(mod)
         if isinstance(obj := getattr(mod, name), type)
         and hasattr(obj, "train") and not inspect.isabstract(obj) and name != "object"),
        None,
    )
    if custom_cls is None:
        return DefaultHandler

    return type("_CombinedHandler", (DefaultHandler,), {
        "train": custom_cls.train,
        "eval":  custom_cls.eval,
    })


# ---------------------------------------------------------------------------
# Core audit function (runs in background thread)
# ---------------------------------------------------------------------------

def run_audit_job(
    job_id: str,
    job_dir: Path,
    job_state: dict[str, Any],
    log_q: queue.Queue,
    save_job: Any = None,
) -> None:
    """Runs inside a ThreadPoolExecutor thread. Updates job_state in-place."""
    leakpro_logger = logging.getLogger("leakpro")
    handler = _QueueHandler(log_q)
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    leakpro_logger.addHandler(handler)

    try:
        job_state["status"] = "running"
        log_q.put(f"[worker] Starting audit for job {job_id}")

        models: list[dict] = job_state.get("models", [])
        handler_config: dict = job_state.get("handler_config", {})
        arch_config: dict = job_state.get("arch_config", {})
        preset = arch_config.get("preset")

        all_results = []

        for model_spec in models:
            model_name = model_spec["name"]
            target_folder = str(job_dir / "models" / model_name)
            # attack_list must be list[dict] — strip UI-only keys
            attack_list = [
                {"attack": a["attack"], **{k: v for k, v in a.get("params", {}).items() if k != "instance_label"}}
                for a in model_spec.get("attacks", [])
            ]

            log_q.put(f"[worker] Auditing model: {model_name}")

            # Guard: if weights were saved as .pt (old naming), rename to .pkl
            target_folder_path = job_dir / "models" / model_name
            pt_path  = target_folder_path / "target_model.pt"
            pkl_path = target_folder_path / "target_model.pkl"
            if pt_path.exists() and not pkl_path.exists():
                import shutil as _shutil
                _shutil.copy2(pt_path, pkl_path)
                log_q.put(f"[worker] Renamed target_model.pt → target_model.pkl for {model_name}")

            # Guard: skip if model_metadata.pkl is missing
            if not (target_folder_path / "model_metadata.pkl").exists():
                log_q.put(f"[worker] WARNING: skipping {model_name} — model_metadata.pkl not found")
                all_results.append({
                    "model_name": model_name,
                    "source": model_spec.get("source", "?"),
                    "dpsgd": False,
                    "attacks": [],
                    "error": "model_metadata.pkl missing — re-train or upload metadata to include this model",
                })
                continue

            # Build audit YAML inline — no dependency on examples/mia/cifar/
            arch_py = job_dir / "arch.py"
            config = {
                "audit": {
                    "random_seed": 1236,
                    "attack_list": [],
                    "output_dir": str(job_dir / "leakpro_output" / model_name),
                    "attack_type": "mia",
                    "data_modality": {"tabular": "tabular", "time_series": "timeseries"}.get(
                        handler_config.get("data_type"), "image"),
                },
                "target": {
                    "module_path": str(arch_py),
                    "model_class": None,
                    "target_folder": target_folder,
                    "data_path": job_state.get("data_path") or "",
                },
                "shadow_model": None,
                "distillation_model": None,
            }

            # Detect model class from arch.py using init_params matching
            if arch_py.exists():
                import importlib.util as _ilu
                import torch.nn as _nn
                import inspect as _inspect
                import joblib as _joblib
                _spec = _ilu.spec_from_file_location("_arch_probe", arch_py)
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                _candidates = [
                    v for v in vars(_mod).values()
                    if isinstance(v, type) and issubclass(v, _nn.Module) and v is not _nn.Module
                ]
                if _candidates:
                    chosen = _candidates[0]
                    try:
                        _meta = _joblib.load(target_folder_path / "model_metadata.pkl")
                        _init_keys = set(_meta.init_params.keys())
                        for _cls in _candidates:
                            _sig_params = set(_inspect.signature(_cls.__init__).parameters) - {"self"}
                            if _init_keys.issubset(_sig_params):
                                chosen = _cls
                                break
                    except Exception:
                        pass
                    config["target"]["model_class"] = chosen.__name__

            if attack_list:
                config["audit"]["attack_list"] = attack_list

            temp_yaml = job_dir / f"_audit_{uuid.uuid4().hex}.yaml"
            try:
                with open(temp_yaml, "w") as f:
                    yaml.dump(config, f)

                handler_cls = _get_handler_cls(job_dir, preset)

                # If user uploaded dataset_handler.py, use its UserDataset instead of the
                # preset handler's built-in one (which may not match the user's data format).
                dataset_handler_py = job_dir / "dataset_handler.py"
                if dataset_handler_py.exists():
                    import importlib.util as _ilu
                    _dspec = _ilu.spec_from_file_location("_dataset_handler", dataset_handler_py)
                    _dmod = _ilu.module_from_spec(_dspec)
                    _dspec.loader.exec_module(_dmod)
                    import inspect as _inspect_ds
                    def _is_concrete_user_dataset(cls: type) -> bool:
                        return (
                            isinstance(cls, type)
                            and not _inspect_ds.isabstract(cls)
                            and cls.__name__ == "UserDataset"
                        )
                    _user_dataset_cls = getattr(_dmod, "UserDataset", None)
                    if _user_dataset_cls is None or not _is_concrete_user_dataset(_user_dataset_cls):
                        _user_dataset_cls = None
                        # UserDataset may be a nested class inside a handler class
                        for _attr_name in dir(_dmod):
                            _obj = getattr(_dmod, _attr_name, None)
                            if isinstance(_obj, type):
                                _nested = getattr(_obj, "UserDataset", None)
                                if _nested is not None and _is_concrete_user_dataset(_nested):
                                    _user_dataset_cls = _nested
                                    break
                    if _user_dataset_cls is not None:
                        handler_cls.UserDataset = _user_dataset_cls
                        log_q.put("[worker] Using custom UserDataset from dataset_handler.py")

                # Add leakpro root to path if needed
                if str(_LEAKPRO_ROOT) not in sys.path:
                    sys.path.insert(0, str(_LEAKPRO_ROOT))

                # Register dataset_handler.py under its original module name so joblib can
                # deserialise population pickles that reference the original class path.
                # Original name is inferred from the handler class name:
                #   CelebADataHandler → celebA_data_handler
                #   CifarDataHandler  → cifar_data_handler
                _dh_path = job_dir / "dataset_handler.py"
                if _dh_path.exists():
                    import importlib.util as _ilu_dh
                    _dh_spec = _ilu_dh.spec_from_file_location("dataset_handler", _dh_path)
                    _dh_mod = _ilu_dh.module_from_spec(_dh_spec)
                    _dh_spec.loader.exec_module(_dh_mod)
                    sys.modules["dataset_handler"] = _dh_mod
                    for _attr in dir(_dh_mod):
                        _obj = getattr(_dh_mod, _attr, None)
                        if (isinstance(_obj, type)
                                and hasattr(_obj, "UserDataset")
                                and _attr not in ("AbstractInputHandler", "object")):
                            _orig = _attr.replace("DataHandler", "_data_handler") \
                                        .replace("InputHandler", "_input_handler")
                            _orig = _orig[0].lower() + _orig[1:]
                            if _orig not in sys.modules:
                                sys.modules[_orig] = _dh_mod
                                log_q.put(f"[worker] Registered dataset_handler.py as '{_orig}' for pickle deserialization")

                # Stub out leakpro.dataset so joblib can deserialise old GeneralDataset pickles.
                # MIAHandler._load_population accesses .data and .targets; the old class stored
                # them as .x and .y — the stub aliases them transparently.
                import types as _types
                import numpy as _np

                if "leakpro.dataset" not in sys.modules:
                    _ds_stub = _types.ModuleType("leakpro.dataset")

                    class _GeneralDataset:
                        def __getattr__(self, name: str):
                            import torch as _torch
                            d = object.__getattribute__(self, "__dict__")
                            if name == "data":
                                x = d.get("x", d.get("X", _np.array([])))
                                if isinstance(x, _np.ndarray):
                                    x = x.astype(_np.float32) / 255.0 if x.dtype == _np.uint8 else x.astype(_np.float32)
                                    if x.ndim == 4 and x.shape[-1] in (1, 3, 4) and x.shape[1] > 4:
                                        x = x.transpose(0, 3, 1, 2)
                                    x = _torch.from_numpy(x)
                                return x
                            if name == "targets":
                                y = d.get("y", d.get("Y", _np.array([])))
                                if isinstance(y, _np.ndarray):
                                    import torch as _t
                                    y = _t.from_numpy(y).long()
                                return y
                            raise AttributeError(name)

                        def __len__(self) -> int:
                            d = object.__getattribute__(self, "__dict__")
                            for k in ("x", "X", "data"):
                                if k in d:
                                    return len(d[k])
                            return 0

                        def __getitem__(self, idx):
                            return self.data[idx], self.targets[idx]

                    _ds_stub.GeneralDataset = _GeneralDataset
                    sys.modules["leakpro.dataset"] = _ds_stub

                from leakpro import LeakPro  # noqa: PLC0415

                lp = LeakPro(handler_cls, str(temp_yaml))
                results = lp.run_audit(create_pdf=False)

                # Detect model class name for display in results
                _preset_class_map = {"cifar_image": "ResNet18", "cifar_wrn": "ResNet18"}
                _model_class: str | None = _preset_class_map.get(preset)  # type: ignore[arg-type]
                if _model_class is None and (job_dir / "arch.py").exists():
                    try:
                        import importlib.util as _ilu3
                        import torch.nn as _nn3
                        _s3 = _ilu3.spec_from_file_location("_mc_probe", job_dir / "arch.py")
                        _m3 = _ilu3.module_from_spec(_s3)
                        _s3.loader.exec_module(_m3)
                        _mc3 = next((v for v in vars(_m3).values()
                                     if isinstance(v, type) and issubclass(v, _nn3.Module)
                                     and v is not _nn3.Module), None)
                        _model_class = _mc3.__name__ if _mc3 else None
                    except Exception:
                        pass
                if model_spec.get("dpsgd") and _model_class == "ResNet18":
                    _model_class = "ResNet18 (DP-SGD)"

                _tp = model_spec.get("train_params") or {}
                _hc = job_state.get("handler_config") or {}
                _dm = job_state.get("data_meta") or {}
                _train_meta: dict | None = {
                    "epochs":             _tp.get("epochs"),
                    "learning_rate":      _tp.get("learning_rate"),
                    "batch_size":         _tp.get("batch_size"),
                    "optimizer":          _tp.get("optimizer"),
                    "f_train":            _tp.get("f_train"),
                    "f_test":             _tp.get("f_test"),
                    "target_delta":       _tp.get("target_delta"),
                    "max_grad_norm":      _tp.get("max_grad_norm"),
                    "virtual_batch_size": _tp.get("virtual_batch_size"),
                    "data_type":          _hc.get("data_type"),
                    "data_shape":         _hc.get("shape"),
                    "n_classes":          _hc.get("n_classes"),
                    "n_samples":          _dm.get("n_samples"),
                } if (_tp or _hc or _dm) else None

                model_results = {
                    "model_name": model_name,
                    "source": model_spec.get("source", "trained"),
                    "dpsgd": model_spec.get("dpsgd", False),
                    "target_epsilon": model_spec.get("target_epsilon"),
                    "train_accuracy": model_spec.get("train_accuracy"),
                    "test_accuracy": model_spec.get("test_accuracy"),
                    "model_class": _model_class,
                    "train_meta": _train_meta,
                    "attacks": [_serialise_result(r) for r in results],
                }
                all_results.append(model_results)
                log_q.put(f"[worker] Done: {model_name} — {len(results)} attack result(s)")

            finally:
                temp_yaml.unlink(missing_ok=True)

        job_state["results"] = all_results
        job_state["status"] = "done"
        if save_job:
            save_job(job_id)
        log_q.put("[worker] All audits complete.")

    except Exception as e:  # noqa: BLE001
        import traceback
        job_state["status"] = "failed"
        job_state["error"] = str(e)
        if save_job:
            save_job(job_id)
        log_q.put(f"[worker] FAILED: {e}\n{traceback.format_exc()}")
    finally:
        leakpro_logger.removeHandler(handler)


def _serialise_result(r: Any) -> dict:
    return {
        "attack_name": r.result_name,
        "roc_auc": float(r.roc_auc) if r.roc_auc is not None else None,
        "tpr_at_fpr": {k: float(v) for k, v in (r.fixed_fpr_table or {}).items()},
        "fpr": r.fpr.tolist() if r.fpr is not None else None,
        "tpr": r.tpr.tolist() if r.tpr is not None else None,
        "signal_values": r.signal_values.tolist() if r.signal_values is not None else None,
        "true_labels": r.true.tolist() if r.true is not None else None,
    }
