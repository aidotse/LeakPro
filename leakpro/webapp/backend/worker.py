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
# Preset handler classes
# ---------------------------------------------------------------------------

_PRESET_HANDLERS: dict[str, str] = {
    "cifar_image": "examples/mia/cifar/cifar_handler.py",
    "cifar_image_dpsgd": "examples/mia/cifar/cifar_handler_dpsgd.py",
}

_LEAKPRO_ROOT = Path(__file__).parents[3]  # /home/.../LeakPro


def _get_handler_cls(job_dir: Path, preset: str | None):
    """Return the handler class to pass to LeakPro."""
    import importlib.util

    if preset and preset in _PRESET_HANDLERS:
        handler_path = _LEAKPRO_ROOT / _PRESET_HANDLERS[preset]
    else:
        handler_path = job_dir / "handler.py"

    spec = importlib.util.spec_from_file_location("user_handler", handler_path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(handler_path.parent))
    spec.loader.exec_module(mod)

    import inspect
    # Return first concrete class that has a 'train' method
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and hasattr(obj, "train") and not inspect.isabstract(obj) and name != "object":
            return obj
    raise ValueError(f"No handler class found in {handler_path}")


# ---------------------------------------------------------------------------
# Core audit function (runs in background thread)
# ---------------------------------------------------------------------------

def run_audit_job(
    job_id: str,
    job_dir: Path,
    job_state: dict[str, Any],
    log_q: queue.Queue,
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

            # Build audit YAML
            cifar_dir = _LEAKPRO_ROOT / "examples/mia/cifar"
            base_yaml = cifar_dir / "audit.yaml"
            with open(base_yaml) as f:
                config = yaml.safe_load(f)

            # Replace all relative paths with absolute paths
            config["target"]["target_folder"] = target_folder
            config["target"]["data_path"] = job_state.get("data_path") or str(cifar_dir / "data/cifar10.pkl")
            config["audit"]["output_dir"] = str(job_dir / "leakpro_output" / model_name)

            # module_path: use the job's arch.py (written for presets or uploaded)
            arch_py = job_dir / "arch.py"
            if arch_py.exists():
                config["target"]["module_path"] = str(arch_py)
                # Detect the model class name from arch.py
                import importlib.util as _ilu
                _spec = _ilu.spec_from_file_location("_arch_probe", arch_py)
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                import torch.nn as _nn
                import inspect as _inspect
                import joblib as _joblib
                _candidates = [
                    v for v in vars(_mod).values()
                    if isinstance(v, type) and issubclass(v, _nn.Module) and v is not _nn.Module
                ]
                if _candidates:
                    # Try to match the class whose __init__ accepts the init_params from metadata
                    chosen = _candidates[-1]
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
            else:
                # Fall back to the example file (absolute path)
                config["target"]["module_path"] = str(cifar_dir / "target_model_class.py")

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

                model_results = {
                    "model_name": model_name,
                    "source": model_spec.get("source", "trained"),
                    "dpsgd": model_spec.get("dpsgd", False),
                    "target_epsilon": model_spec.get("target_epsilon"),
                    "test_accuracy": model_spec.get("test_accuracy"),
                    "attacks": [_serialise_result(r) for r in results],
                }
                all_results.append(model_results)
                log_q.put(f"[worker] Done: {model_name} — {len(results)} attack result(s)")

            finally:
                temp_yaml.unlink(missing_ok=True)

        job_state["results"] = all_results
        job_state["status"] = "done"
        log_q.put("[worker] All audits complete.")

    except Exception as e:  # noqa: BLE001
        import traceback
        job_state["status"] = "failed"
        job_state["error"] = str(e)
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
