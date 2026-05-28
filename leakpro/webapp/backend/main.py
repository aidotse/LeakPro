"""LeakPro webapp — FastAPI backend."""

from __future__ import annotations

import asyncio
import json
import queue
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from leakpro.schemas import EvalOutput, TrainingOutput

# ---------------------------------------------------------------------------
# Default train / eval — used when no custom handler.py is uploaded
# ---------------------------------------------------------------------------

def _default_train(loader, model, criterion, optimizer, epochs,
                   dpsgd_metadata_path=None, virtual_batch_size=16):
    """Built-in training loop. Pass dpsgd_metadata_path to activate DP-SGD."""
    import os, pickle
    import torch
    from tqdm import tqdm

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dpsgd_metadata_path:
        from opacus import PrivacyEngine
        from opacus.accountants.utils import get_noise_multiplier
        from opacus.utils.batch_memory_manager import BatchMemoryManager
        from opacus.validators import ModuleValidator

        errors = ModuleValidator.validate(model, strict=False)
        if errors:
            model = ModuleValidator.fix(model)
            opt_cls = optimizer.__class__
            opt_cfg = {k: v for group in optimizer.param_groups
                       for k, v in group.items() if k != "params"}
            optimizer = opt_cls(model.parameters(), **opt_cfg)

        for module in model.modules():
            if hasattr(module, "inplace") and isinstance(module.inplace, bool):
                module.inplace = False

        # ModuleValidator.fix() replaces BatchNorm but leaves ResNet's skip-connection
        # `out += identity` in-place, which Opacus's backward hooks forbid.
        # Patch every BasicBlock/Bottleneck to use `out = out + identity` instead.
        try:
            import types as _types
            from torchvision.models.resnet import BasicBlock as _BB, Bottleneck as _BN
            def _bb_fwd(self, x):
                identity = x
                out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
                out = self.conv2(out); out = self.bn2(out)
                if self.downsample is not None: identity = self.downsample(x)
                return self.relu(out + identity)
            def _bn_fwd(self, x):
                identity = x
                out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
                out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
                out = self.conv3(out); out = self.bn3(out)
                if self.downsample is not None: identity = self.downsample(x)
                return self.relu(out + identity)
            for _m in model.modules():
                if isinstance(_m, _BB):
                    _m.forward = _types.MethodType(_bb_fwd, _m)
                elif isinstance(_m, _BN):
                    _m.forward = _types.MethodType(_bn_fwd, _m)
        except ImportError:
            pass

        if not os.path.exists(dpsgd_metadata_path):
            raise FileNotFoundError(f"DP-SGD config not found: {dpsgd_metadata_path}")
        with open(dpsgd_metadata_path, "rb") as f:
            cfg = pickle.load(f)

        sample_rate = 1 / len(loader)
        noise_multiplier = get_noise_multiplier(
            target_epsilon=cfg["target_epsilon"],
            target_delta=cfg["target_delta"],
            sample_rate=sample_rate,
            epochs=cfg["epochs"],
            epsilon_tolerance=cfg["epsilon_tolerance"],
            accountant="prv",
            eps_error=cfg["eps_error"],
        )
        privacy_engine = PrivacyEngine(accountant="prv")
        model, optimizer, loader = privacy_engine.make_private(
            module=model, optimizer=optimizer, data_loader=loader,
            noise_multiplier=noise_multiplier, max_grad_norm=cfg["max_grad_norm"],
        )

        model.to(dev)
        accuracy_history, loss_history = [], []
        with BatchMemoryManager(data_loader=loader,
                                max_physical_batch_size=virtual_batch_size,
                                optimizer=optimizer) as mem_loader:
            for epoch in range(epochs):
                model.train()
                train_loss, train_acc = 0.0, 0.0
                for inputs, labels in tqdm(mem_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                    labels = labels.long().view(-1)
                    inputs, labels = inputs.to(dev), labels.to(dev)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_acc += outputs.argmax(1).eq(labels).sum().item()
                    train_loss += loss.item() * labels.size(0)
                n = len(mem_loader.dataset)
                accuracy_history.append(train_acc / n)
                loss_history.append(train_loss / n)

        model.to("cpu")
        if hasattr(model, "_module"):
            model = model._module

    else:
        model.to(dev)
        accuracy_history, loss_history = [], []
        for epoch in range(epochs):
            model.train()
            train_loss, train_acc, total = 0.0, 0.0, 0
            for inputs, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
                labels = labels.long().view(-1)
                inputs, labels = inputs.to(dev), labels.to(dev)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_acc += outputs.argmax(1).eq(labels).sum().item()
                train_loss += loss.item() * labels.size(0)
                total += labels.size(0)
            accuracy_history.append(train_acc / total)
            loss_history.append(train_loss / total)
        model.to("cpu")

    metrics = EvalOutput(
        accuracy=accuracy_history[-1] if accuracy_history else 0.0,
        loss=loss_history[-1] if loss_history else 0.0,
        extra={"accuracy_history": accuracy_history, "loss_history": loss_history},
    )
    return TrainingOutput(model=model, metrics=metrics)


def _default_eval(loader, model, criterion):
    """Built-in eval loop."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    model.eval()
    loss, acc, total = 0.0, 0.0, 0
    with torch.no_grad():
        for data, target in loader:
            target = target.long().view(-1).to(dev)
            data = data.to(dev)
            output = model(data)
            loss += criterion(output, target).item() * target.size(0)
            acc += output.argmax(1).eq(target).sum().item()
            total += target.size(0)
    model.to("cpu")
    return EvalOutput(accuracy=acc / total if total else 0.0, loss=loss / total if total else 0.0)


# ---------------------------------------------------------------------------
# Preset architecture files written to disk when a preset is selected
# ---------------------------------------------------------------------------

_PRESET_ARCH_IMAGE = '''\
"""Preset architecture — ResNet-18 trained from scratch for any image dataset."""
import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
'''

_PRESET_ARCHS: dict[str, str] = {
    "cifar_wrn":   _PRESET_ARCH_IMAGE,
    "cifar_image": _PRESET_ARCH_IMAGE,
}

from .checker import run_check
from .inspector import inspect
from .models import (
    ArchConfig,
    AttackParams,
    CompatResult,
    DataMeta,
    HandlerConfig,
    JobStatus,
    JobSummary,
    ModelAttackConfig,
    ModelInfo,
    TrainParams,
)
from .worker import run_audit_job

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

JOBS_ROOT = Path(__file__).parents[3] / "webapp_jobs"
JOBS_ROOT.mkdir(parents=True, exist_ok=True)

# In-memory job registry  { job_id: { status, created_at, models, ... } }
_jobs: dict[str, dict[str, Any]] = {}
# Per-job log queues
_log_queues: dict[str, queue.Queue] = {}
_executor = ThreadPoolExecutor(max_workers=4)


def _job_dir(job_id: str) -> Path:
    return JOBS_ROOT / job_id


def _get_job(job_id: str) -> dict:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _jobs[job_id]


# ---------------------------------------------------------------------------
# Job state persistence — survives backend restarts
# ---------------------------------------------------------------------------

def _save_job(job_id: str) -> None:
    """Write job state to state.json so it survives backend restarts."""
    job = _jobs.get(job_id)
    if job is None:
        return
    state_path = _job_dir(job_id) / "state.json"
    serializable: dict[str, Any] = {}
    for k, v in job.items():
        if isinstance(v, JobStatus):
            serializable[k] = v.value
        elif isinstance(v, (str, int, float, bool, list, dict, type(None))):
            serializable[k] = v
    try:
        with open(state_path, "w") as f:
            json.dump(serializable, f, indent=2)
    except Exception:
        pass  # Never crash a request just because we couldn't save


def _load_jobs() -> None:
    """On startup, restore all persisted jobs from state.json files."""
    for job_dir in JOBS_ROOT.iterdir():
        if not job_dir.is_dir():
            continue
        state_path = job_dir / "state.json"
        if not state_path.exists():
            continue
        try:
            with open(state_path) as f:
                state = json.load(f)
            job_id = job_dir.name
            if "status" in state:
                try:
                    state["status"] = JobStatus(state["status"])
                except ValueError:
                    state["status"] = JobStatus.pending
            # Jobs that were running when the backend crashed are now failed
            if state.get("status") == JobStatus.running:
                state["status"] = JobStatus.failed
                state["error"] = "Backend restarted during job"
            _jobs[job_id] = state
            _log_queues[job_id] = queue.Queue()
        except Exception:
            pass  # Skip corrupted state files


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_jobs()
    yield


app = FastAPI(title="LeakPro Webapp API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------

@app.get("/jobs")
async def list_jobs() -> list[dict]:
    """Return all known jobs — used for cross-session result comparison."""
    out = []
    for jid, job in _jobs.items():
        models = job.get("models", [])
        attacks_per_model = {
            m["name"]: [a.get("attack", "") for a in m.get("attacks", [])]
            for m in models
        }
        out.append({
            "job_id": jid,
            "status": job.get("status", "unknown"),
            "created_at": job.get("created_at", ""),
            "model_names": [m["name"] for m in models],
            "attacks_per_model": attacks_per_model,
        })
    out.sort(key=lambda x: x["created_at"], reverse=True)
    return out


@app.post("/jobs", response_model=JobSummary)
async def create_job() -> JobSummary:
    job_id = uuid.uuid4().hex
    _job_dir(job_id).mkdir(parents=True)
    (_job_dir(job_id) / "models").mkdir()
    _jobs[job_id] = {
        "status": JobStatus.pending,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "models": [],
    }
    _log_queues[job_id] = queue.Queue()
    _save_job(job_id)
    return JobSummary(job_id=job_id, status=JobStatus.pending,
                      created_at=_jobs[job_id]["created_at"])


@app.get("/jobs/{job_id}/status")
async def get_status(job_id: str) -> dict:
    job = _get_job(job_id)
    return {"job_id": job_id, "status": job["status"], "error": job.get("error")}


# ---------------------------------------------------------------------------
# Step 1 — Upload dataset
# ---------------------------------------------------------------------------

@app.post("/jobs/{job_id}/upload/data", response_model=DataMeta)
async def upload_data(job_id: str, file: UploadFile) -> DataMeta:
    job = _get_job(job_id)
    dest = _job_dir(job_id) / f"data{Path(file.filename).suffix}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    meta = inspect(dest)
    job["data_path"] = str(dest)
    job["data_meta"] = meta.model_dump()
    _save_job(job_id)
    return meta


@app.post("/jobs/{job_id}/data-path", response_model=DataMeta)
async def set_data_path(job_id: str, body: dict) -> DataMeta:
    """Use a dataset already on the server by absolute path."""
    job = _get_job(job_id)
    path = Path(body.get("path", ""))
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {path}")
    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"Path is not a file: {path}")
    try:
        meta = inspect(path)
    except Exception as e:
        import traceback
        raise HTTPException(status_code=400, detail=f"Failed to inspect file: {e}\n{traceback.format_exc()}") from e
    job["data_path"] = str(path)
    job["data_meta"] = meta.model_dump()
    _save_job(job_id)
    return meta


@app.post("/jobs/{job_id}/upload/dataset-handler")
async def upload_dataset_handler(job_id: str, file: UploadFile) -> dict:
    """Upload a custom dataset_handler.py defining UserDataset."""
    _get_job(job_id)
    dest = _job_dir(job_id) / "dataset_handler.py"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"ok": True, "filename": file.filename}


@app.post("/jobs/{job_id}/dataset-handler-path")
async def set_dataset_handler_path(job_id: str, body: dict) -> dict:
    """Use a dataset_handler.py already on the server by absolute path."""
    _get_job(job_id)
    path = Path(body.get("path", ""))
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {path}")
    if not path.suffix == ".py":
        raise HTTPException(status_code=400, detail="Path must point to a .py file")
    dest = _job_dir(job_id) / "dataset_handler.py"
    shutil.copy2(path, dest)
    return {"ok": True, "filename": path.name}


# ---------------------------------------------------------------------------
# Step 2 — Confirm format
# ---------------------------------------------------------------------------

@app.post("/jobs/{job_id}/handler-config")
async def set_handler_config(job_id: str, config: HandlerConfig) -> dict:
    job = _get_job(job_id)
    job["handler_config"] = config.model_dump()
    _save_job(job_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Step 3 — Architecture + training loop
# ---------------------------------------------------------------------------

@app.post("/jobs/{job_id}/upload/arch")
async def upload_arch(job_id: str, file: UploadFile) -> dict:
    _get_job(job_id)
    dest = _job_dir(job_id) / "arch.py"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"ok": True, "filename": file.filename}


@app.post("/jobs/{job_id}/arch-path")
async def set_arch_path(job_id: str, body: dict) -> dict:
    _get_job(job_id)
    path = Path(body.get("path", ""))
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {path}")
    # Symlink or copy into job dir
    dest = _job_dir(job_id) / "arch.py"
    shutil.copy2(path, dest)
    return {"ok": True, "filename": path.name}


@app.post("/jobs/{job_id}/upload/handler")
async def upload_handler(job_id: str, file: UploadFile) -> dict:
    _get_job(job_id)
    dest = _job_dir(job_id) / "handler.py"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"ok": True, "filename": file.filename}


@app.post("/jobs/{job_id}/handler-path")
async def set_handler_path(job_id: str, body: dict) -> dict:
    _get_job(job_id)
    path = Path(body.get("path", ""))
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {path}")
    dest = _job_dir(job_id) / "handler.py"
    shutil.copy2(path, dest)
    return {"ok": True, "filename": path.name}


@app.post("/jobs/{job_id}/arch-config")
async def set_arch_config(job_id: str, config: ArchConfig) -> dict:
    job = _get_job(job_id)
    job["arch_config"] = config.model_dump()
    # For presets write a built-in arch.py so /check can find it
    if config.preset and config.preset in _PRESET_ARCHS:
        arch_dest = _job_dir(job_id) / "arch.py"
        arch_dest.write_text(_PRESET_ARCHS[config.preset])
    _save_job(job_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Step 4A — Upload existing model weights + compatibility check
# ---------------------------------------------------------------------------

@app.post("/jobs/{job_id}/upload/weights")
async def upload_weights(job_id: str, model_name: str, file: UploadFile) -> dict:
    _get_job(job_id)
    model_dir = _job_dir(job_id) / "models" / model_name
    model_dir.mkdir(exist_ok=True)
    dest = model_dir / "target_model.pkl"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"ok": True, "path": str(dest)}


@app.post("/jobs/{job_id}/upload/model-metadata")
async def upload_model_metadata(job_id: str, model_name: str, file: UploadFile) -> dict:
    _get_job(job_id)
    model_dir = _job_dir(job_id) / "models" / model_name
    model_dir.mkdir(exist_ok=True)
    dest = model_dir / "model_metadata.pkl"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"ok": True, "path": str(dest)}


@app.post("/jobs/{job_id}/model-metadata-path")
async def set_model_metadata_path(job_id: str, body: dict) -> dict:
    """Copy a metadata file already on the server into the job directory."""
    _get_job(job_id)
    path = Path(body.get("path", ""))
    model_name = body.get("model_name", "uploaded_model")
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {path}")
    model_dir = _job_dir(job_id) / "models" / model_name
    model_dir.mkdir(exist_ok=True)
    dest = model_dir / "model_metadata.pkl"
    shutil.copy2(path, dest)
    return {"ok": True, "path": str(dest)}


@app.post("/jobs/{job_id}/validate/model-metadata")
async def validate_model_metadata(job_id: str, model_name: str) -> dict:
    """Check that an uploaded model_metadata.pkl has all required MIA fields."""
    _get_job(job_id)
    meta_path = _job_dir(job_id) / "models" / model_name / "model_metadata.pkl"
    if not meta_path.exists():
        raise HTTPException(status_code=400, detail="No metadata file uploaded yet")

    required = ["train_indices", "test_indices", "optimizer", "criterion",
                "data_loader", "epochs", "train_result", "test_result", "dataset"]
    import pickle

    class _SafeUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except (ImportError, AttributeError):
                return type(name, (), {})

    try:
        with open(meta_path, "rb") as f:
            raw = _SafeUnpickler(f).load()
        attrs: dict = raw.__dict__ if hasattr(raw, "__dict__") else (raw if isinstance(raw, dict) else {})
        present = [k for k in required if k in attrs]
        missing = [k for k in required if k not in attrs]
        return {"ok": not missing, "present_fields": present, "missing_fields": missing}
    except Exception as e:
        import traceback
        return {"ok": False, "present_fields": [], "missing_fields": required,
                "error": f"{e}\n{traceback.format_exc()}"}


@app.post("/jobs/{job_id}/weights-path")
async def set_weights_path(job_id: str, body: dict) -> dict:
    _get_job(job_id)
    path = Path(body.get("path", ""))
    model_name = body.get("model_name", "uploaded_model")
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {path}")
    model_dir = _job_dir(job_id) / "models" / model_name
    model_dir.mkdir(exist_ok=True)
    dest = model_dir / "target_model.pkl"
    shutil.copy2(path, dest)
    return {"ok": True, "path": str(dest)}


@app.post("/jobs/{job_id}/check", response_model=CompatResult)
async def check_compat(job_id: str, model_name: str) -> CompatResult:
    job = _get_job(job_id)
    job_dir = _job_dir(job_id)
    arch_path = job_dir / "arch.py"
    weights_path = job_dir / "models" / model_name / "target_model.pkl"
    data_meta = job.get("data_meta", {})
    data_shape = data_meta.get("shape", [3, 32, 32])
    # Convert channel-last (H,W,C) → channel-first (C,H,W) for the dummy forward pass
    if len(data_shape) == 3 and data_shape[-1] in (1, 3, 4) and data_shape[0] > 4:
        data_shape = [data_shape[2], data_shape[0], data_shape[1]]

    if not arch_path.exists() or arch_path.stat().st_size == 0:
        raise HTTPException(status_code=400, detail="No architecture file found. Please upload arch.py or provide a server path before validating.")

    data_path_str = job.get("data_path")
    result = run_check(
        arch_path=arch_path,
        weights_path=weights_path if weights_path.exists() else None,
        data_shape=data_shape,
        data_path=Path(data_path_str) if data_path_str else None,
    )
    # Register model in job state so set_attack_config and the worker can find it
    job_models = job.setdefault("models", [])
    existing = next((m for m in job_models if m["name"] == model_name), None)
    if existing is None:
        _train_acc, _test_acc = None, None
        try:
            import joblib as _jl
            _meta = _jl.load(job_dir / "models" / model_name / "model_metadata.pkl")
            _attrs = _meta.__dict__ if hasattr(_meta, "__dict__") else (_meta if isinstance(_meta, dict) else {})
            _tr = _attrs.get("train_result")
            _te = _attrs.get("test_result")
            if _tr is not None:
                _v = getattr(_tr, "accuracy", None) or (_tr.get("accuracy") if isinstance(_tr, dict) else None)
                _train_acc = float(_v) if _v is not None else None
            if _te is not None:
                _v = getattr(_te, "accuracy", None) or (_te.get("accuracy") if isinstance(_te, dict) else None)
                _test_acc = float(_v) if _v is not None else None
        except Exception:
            pass
        job_models.append({
            "name": model_name,
            "source": "uploaded",
            "target_folder": str(job_dir / "models" / model_name),
            "status": "ready" if result.ok else "error",
            "dpsgd": False,
            "train_accuracy": _train_acc,
            "test_accuracy": _test_acc,
        })
    elif result.ok:
        existing["status"] = "ready"
    _save_job(job_id)
    return result


# ---------------------------------------------------------------------------
# Step 4A-delete — Remove a model from job state
# ---------------------------------------------------------------------------

@app.delete("/jobs/{job_id}/models/{model_name}")
async def remove_model(job_id: str, model_name: str) -> dict:
    """Remove a model entry from the job state."""
    job = _get_job(job_id)
    job["models"] = [m for m in job.get("models", []) if m["name"] != model_name]
    _save_job(job_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Step 4B — Train a model
# ---------------------------------------------------------------------------

@app.post("/jobs/{job_id}/train")
async def train_model(job_id: str, params: TrainParams) -> dict:
    """Enqueue a training job. Progress streams via WS /jobs/{id}/logs."""
    job = _get_job(job_id)
    log_q = _log_queues[job_id]

    def _train() -> None:
        import io
        import re
        import sys as _sys

        # Capture all print() / tqdm output into the log queue
        class _StdoutCapture(io.TextIOBase):
            def write(self, s: str) -> int:  # noqa: D102
                line = s.strip()
                if line:
                    log_q.put(line)
                return len(s)
            def flush(self) -> None:  # noqa: D102
                pass

        old_stdout = _sys.stdout
        _sys.stdout = _StdoutCapture()

        try:
            log_q.put(f"[train] Starting: {params.name}  ({params.epochs} epochs, lr={params.learning_rate})")
            leakpro_root = Path(__file__).parents[3]
            if str(leakpro_root) not in _sys.path:
                _sys.path.insert(0, str(leakpro_root))

            job_dir = _job_dir(job_id)
            target_folder = job_dir / "models" / params.name
            target_folder.mkdir(parents=True, exist_ok=True)

            # Save training log to file
            _train_log_path = target_folder / "train.log"
            _train_log_file = open(_train_log_path, "w")
            _orig_log_q_put = log_q.put
            def _log_and_save(msg):
                _orig_log_q_put(msg)
                if not msg.startswith("__"):
                    _train_log_file.write(msg + "\n")
                    _train_log_file.flush()
            log_q.put = _log_and_save

            model_entry = {
                "name": params.name,
                "source": "trained",
                "target_folder": str(target_folder),
                "dpsgd": params.dpsgd,
                "target_epsilon": params.target_epsilon,
                "train_params": params.model_dump(),
                "status": "training",
            }
            job["models"] = [m for m in job.get("models", []) if m["name"] != params.name]
            job["models"].append(model_entry)

            # ── Load CIFAR dataset & model, run training ────────────────────
            cifar_dir = leakpro_root / "examples" / "mia" / "cifar"
            if str(cifar_dir) not in _sys.path:
                _sys.path.insert(0, str(cifar_dir))

            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader
            import importlib.util

            import inspect

            # Load dataset, apply f_train fraction
            data_path = Path(job.get("data_path", ""))
            if data_path.exists():
                import pickle

                def _safe_load_dataset(path: Path):
                    """Load a dataset pickle, falling back to SafeUnpickler for missing modules."""
                    try:
                        with open(path, "rb") as _f:
                            return pickle.load(_f)
                    except Exception:
                        pass
                    # Fall back: stub out missing classes, then reconstruct as TensorDataset
                    class _SafeUnpickler(pickle.Unpickler):
                        def find_class(self, module, name):
                            try:
                                return super().find_class(module, name)
                            except (ImportError, AttributeError):
                                return type(name, (), {})
                    with open(path, "rb") as _f:
                        raw = _SafeUnpickler(_f).load()
                    import numpy as np
                    # Use __dict__ directly — bypasses any descriptor protocol on the stub class
                    attrs = getattr(raw, "__dict__", {})
                    data_arr = targets_arr = None
                    for data_attr in ("data", "x", "X", "features", "samples"):
                        val = attrs.get(data_attr)
                        if val is not None:
                            data_arr = val if isinstance(val, np.ndarray) else (val.numpy() if hasattr(val, "numpy") else None)
                            if data_arr is not None:
                                break
                    for tgt_attr in ("targets", "y", "Y", "labels"):
                        val = attrs.get(tgt_attr)
                        if val is not None:
                            targets_arr = val if isinstance(val, (list, np.ndarray)) else (val.tolist() if hasattr(val, "tolist") else None)
                            if targets_arr is not None:
                                break
                    if data_arr is None:
                        raise ValueError(
                            f"Cannot extract data array from {type(raw).__name__}. "
                            f"Available keys: {list(attrs.keys())}"
                        )
                    _raw_np = np.array(data_arr)
                    x = torch.from_numpy(_raw_np.astype("float32"))
                    # Only normalize if data is raw uint8 pixels — already-normalized floats must not be touched
                    if _raw_np.dtype == np.uint8 or (_raw_np.dtype != np.float32 and _raw_np.dtype != np.float64 and x.max() > 1.0):
                        x = x / 255.0
                    # Channel-last (N,H,W,C) → channel-first (N,C,H,W)
                    if x.ndim == 4 and x.shape[-1] in (1, 3, 4) and x.shape[1] > 4:
                        x = x.permute(0, 3, 1, 2).contiguous()
                    y = torch.tensor(np.array(targets_arr), dtype=torch.long) if targets_arr is not None else torch.zeros(len(x), dtype=torch.long)
                    return torch.utils.data.TensorDataset(x, y)

                dataset = _safe_load_dataset(data_path)
                if not isinstance(dataset, torch.utils.data.Dataset):
                    raise ValueError(f"Loaded object is not a Dataset: {type(dataset)}")
            else:
                # Fallback: download CIFAR-10
                import torchvision
                import torchvision.transforms as T
                transform = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                dataset = torchvision.datasets.CIFAR10(root=str(cifar_dir / "data"), train=True, download=True, transform=transform)

            # Split dataset into non-overlapping train and test subsets
            import random as _random
            n_total = len(dataset)
            all_idx = list(range(n_total))
            _random.shuffle(all_idx)
            train_size = max(1, int(n_total * params.f_train))
            test_size  = max(1, int(n_total * params.f_test))
            if train_size + test_size > n_total:
                test_size = max(1, n_total - train_size)
            train_indices_list = all_idx[:train_size]
            test_indices_list  = all_idx[train_size:train_size + test_size]

            train_subset = torch.utils.data.Subset(dataset, train_indices_list)
            test_subset  = torch.utils.data.Subset(dataset, test_indices_list)

            log_q.put(f"[train] Dataset: {train_size} train / {test_size} test  (f_train={params.f_train}, f_test={params.f_test})")
            loader      = DataLoader(train_subset, batch_size=params.batch_size, shuffle=True,  num_workers=0)
            test_loader = DataLoader(test_subset,  batch_size=params.batch_size, shuffle=False, num_workers=0)

            # Load model architecture from arch.py (preset writes it there; custom uploads it)
            if True:
                import importlib.util as _ilu2
                import inspect as _inspect2
                import torch.nn as _nn2

                # Register job_dir on sys.path so pickle can reimport the module by name
                if str(job_dir) not in _sys.path:
                    _sys.path.insert(0, str(job_dir))

                # Load under a stable importable name and register in sys.modules
                _arch_mod_name = "target_model_class"
                if _arch_mod_name in _sys.modules:
                    del _sys.modules[_arch_mod_name]  # avoid stale cache between models
                _arch_spec = _ilu2.spec_from_file_location(_arch_mod_name, job_dir / "arch.py")
                _arch_mod = _ilu2.module_from_spec(_arch_spec)
                _sys.modules[_arch_mod_name] = _arch_mod  # register before exec so relative imports work
                _arch_spec.loader.exec_module(_arch_mod)

                try:
                    _arch_cls = next(
                        v for v in vars(_arch_mod).values()
                        if isinstance(v, type) and issubclass(v, _nn2.Module) and v is not _nn2.Module
                    )
                except StopIteration:
                    raise ValueError(
                        "No nn.Module subclass found in arch.py. "
                        "Make sure your arch.py defines a class that inherits from torch.nn.Module."
                    )

                # Detect num_classes from the FULL population so labels seen only
                # in the test/audit split don't produce out-of-bounds model outputs.
                _all_labels = []
                for _, _lbl in DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0):
                    _all_labels.append(_lbl.view(-1))
                _num_classes_data = int(torch.cat(_all_labels).max().item()) + 1
                log_q.put(f"[train] Detected {_num_classes_data} classes from dataset labels")

                # Instantiate model, passing num_classes if the constructor accepts it
                def _make_model(cls, num_classes, dpsgd=False):
                    sig = _inspect2.signature(cls.__init__)
                    kwargs = {}
                    if "num_classes" in sig.parameters:
                        kwargs["num_classes"] = num_classes
                    if dpsgd and "dpsgd" in sig.parameters:
                        kwargs["dpsgd"] = True
                    return cls(**kwargs)

                if params.dpsgd:
                    model = _make_model(_arch_cls, _num_classes_data, dpsgd=True)
                    if not hasattr(model, "dpsgd") or not model.dpsgd:
                        from opacus.validators import ModuleValidator as _MV  # noqa: PLC0415
                        model = _MV.fix(model)
                        log_q.put(f"[train] Applied ModuleValidator.fix() to {_arch_cls.__name__} for DP-SGD")
                    else:
                        log_q.put(f"[train] Instantiated {_arch_cls.__name__}(dpsgd=True, num_classes={_num_classes_data})")
                else:
                    model = _make_model(_arch_cls, _num_classes_data)
                # Check if arch uses pretrained weights
                _arch_src = (job_dir / "arch.py").read_text()
                _pretrained = (
                    ("ResNet18_Weights" in _arch_src and "weights=None" not in _arch_src) or
                    ("pretrained=True" in _arch_src)
                )
                log_q.put(f"[train] Model: {_arch_cls.__name__}(num_classes={_num_classes_data}, pretrained={'yes' if _pretrained else 'no'})")
                if params.dpsgd:
                    log_q.put(f"[train] DP-SGD: epsilon={params.target_epsilon}, delta={params.target_delta}, max_grad_norm={params.max_grad_norm}, virtual_batch_size={params.virtual_batch_size or 16}")

            criterion = nn.CrossEntropyLoss()
            if params.optimizer == "sgd":
                optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)
            else:
                optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

            # Build dpsgd_dic.pkl if training with DP-SGD
            _dpsgd_path = None
            _vbs = params.virtual_batch_size or 16
            if params.dpsgd:
                import pickle as _pkl
                _dpsgd_meta = {
                    "target_epsilon":    params.target_epsilon or 10.0,
                    "target_delta":      params.target_delta if params.target_delta is not None else 1e-5,
                    "sample_rate":       1.0 / len(loader),
                    "epochs":            params.epochs,
                    "epsilon_tolerance": 0.01,
                    "accountant":        "prv",
                    "eps_error":         0.01,
                    "max_grad_norm":     params.max_grad_norm or 1.0,
                }
                _dpsgd_path = target_folder / "dpsgd_dic.pkl"
                with open(_dpsgd_path, "wb") as _f:
                    _pkl.dump(_dpsgd_meta, _f)
                log_q.put(f"[train] DP-SGD config saved (ε={_dpsgd_meta['target_epsilon']}, δ={_dpsgd_meta['target_delta']}, max_grad_norm={_dpsgd_meta['max_grad_norm']})")

            # Monkey-patch tqdm to emit __PROGRESS__ markers
            _orig_tqdm = None
            try:
                import tqdm as _tqdm_mod
                _OrigTqdm = _tqdm_mod.tqdm

                class _ProgressTqdm(_OrigTqdm):
                    def __init__(self, *a, **kw):
                        super().__init__(*a, **kw)
                        # Parse "Epoch X/Y" from desc
                        desc = kw.get("desc", "")
                        m = re.search(r"Epoch\s+(\d+)/(\d+)", desc or "")
                        if m:
                            log_q.put(f"__PROGRESS__{params.name}|{m.group(1)}|{m.group(2)}")

                _tqdm_mod.tqdm = _ProgressTqdm
                _orig_tqdm = (_tqdm_mod, _OrigTqdm)
            except ImportError:
                pass

            _custom_handler_path = job_dir / "handler.py"
            if _custom_handler_path.exists():
                _hspec = importlib.util.spec_from_file_location("_custom_handler", _custom_handler_path)
                _hmod = importlib.util.module_from_spec(_hspec)
                _hspec.loader.exec_module(_hmod)
                _HandlerCls = next(
                    obj for name in dir(_hmod)
                    if isinstance(obj := getattr(_hmod, name), type)
                    and hasattr(obj, "train") and not inspect.isabstract(obj)
                    and name != "object"
                )
                _handler = _HandlerCls()
                _dpsgd_kwargs = ({"dpsgd_metadata_path": str(_dpsgd_path), "virtual_batch_size": _vbs}
                                 if _dpsgd_path else {})
                result = _handler.train(loader, model, criterion, optimizer,
                                        epochs=params.epochs, **_dpsgd_kwargs)
                test_eval = _handler.eval(test_loader, result.model, criterion)
            else:
                result = _default_train(
                    loader, model, criterion, optimizer, epochs=params.epochs,
                    dpsgd_metadata_path=str(_dpsgd_path) if _dpsgd_path else None,
                    virtual_batch_size=_vbs,
                )
                test_eval = _default_eval(test_loader, result.model, criterion)

            # Restore tqdm
            if _orig_tqdm:
                _orig_tqdm[0].tqdm = _orig_tqdm[1]

            # Save trained model
            import json as _json
            torch.save(result.model.state_dict(), target_folder / "target_model.pkl")

            # Evaluate on test set and save MIA metadata
            try:
                from leakpro import LeakPro  # noqa: PLC0415
                import pickle as _pickle
                metadata = LeakPro.make_mia_metadata(
                    train_result=result,
                    optimizer=optimizer,
                    loss_fn=criterion,
                    dataloader=loader,
                    test_result=test_eval,
                    epochs=params.epochs,
                    train_indices=train_indices_list,
                    test_indices=test_indices_list,
                    dataset_name="cifar10",
                )
                with open(target_folder / "model_metadata.pkl", "wb") as _f:
                    _pickle.dump(metadata, _f)
                log_q.put(f"[train] model_metadata.pkl saved  (train={len(train_indices_list)}, test={len(test_indices_list)})")
            except Exception as _meta_err:  # noqa: BLE001
                import traceback as _tb
                log_q.put(f"[train] WARNING: could not save model_metadata.pkl: {_meta_err}\n{_tb.format_exc()}")
            train_acc = None
            test_acc = None
            loss_history: list = []
            acc_history: list = []
            val_loss_history: list = []
            val_acc_history: list = []
            if result.metrics:
                train_acc = float(result.metrics.accuracy)
                extra = result.metrics.extra or {}
                loss_history = [float(x) for x in extra.get("loss_history", [])]
                acc_history = [float(x) for x in extra.get("accuracy_history", [])]
                val_loss_history = [float(x) for x in extra.get("val_loss_history", [])]
                val_acc_history = [float(x) for x in extra.get("val_acc_history", [])]
            try:
                test_acc = float(test_eval.accuracy)
                test_loss = float(test_eval.loss) if test_eval.loss is not None else None
            except Exception:
                test_acc = None
                test_loss = None
            model_entry["status"] = "ready"
            model_entry["train_accuracy"] = train_acc
            model_entry["test_accuracy"] = test_acc
            _save_job(job_id)
            log_q.put(f"[train] Done: {params.name}  accuracy={test_acc:.4f}" if test_acc else f"[train] Done: {params.name}")
            # Emit metrics for frontend chart
            log_q.put("__METRICS__" + _json.dumps({
                "model": params.name,
                "loss_history": loss_history,
                "accuracy_history": acc_history,
                "val_loss_history": val_loss_history,
                "val_acc_history": val_acc_history,
                "test_acc_final": test_acc,
                "test_loss_final": test_loss,
            }))

        except Exception as e:  # noqa: BLE001
            import traceback
            model_entry = next((m for m in job.get("models", []) if m["name"] == params.name), {})
            model_entry["status"] = "error"
            _save_job(job_id)
            log_q.put(f"[train] FAILED {params.name}: {e}\n{traceback.format_exc()}")
        finally:
            _sys.stdout = old_stdout
            log_q.put = _orig_log_q_put
            _train_log_file.close()
            log_q.put("__TRAIN_DONE__")

    _executor.submit(_train)
    return {"ok": True, "model_name": params.name}


# ---------------------------------------------------------------------------
# Step 5 — Attack config per model
# ---------------------------------------------------------------------------

@app.post("/jobs/{job_id}/attack-config")
async def set_attack_config(job_id: str, configs: list[ModelAttackConfig]) -> dict:
    job = _get_job(job_id)
    cfg_map = {c.model_name: c.attacks for c in configs}
    existing_names = {m["name"] for m in job.get("models", [])}

    # Upsert any model the frontend knows about that isn't in the backend state yet
    for model_name in cfg_map:
        if model_name not in existing_names:
            job.setdefault("models", []).append({
                "name": model_name,
                "source": "uploaded",
                "target_folder": str(_job_dir(job_id) / "models" / model_name),
                "status": "ready",
                "dpsgd": False,
            })
            existing_names.add(model_name)

    for model in job.get("models", []):
        if model["name"] in cfg_map:
            model["attacks"] = [a.model_dump() for a in cfg_map[model["name"]]]
    _save_job(job_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Step 6 — Run audit
# ---------------------------------------------------------------------------

@app.post("/jobs/{job_id}/start")
async def start_audit(job_id: str) -> dict:
    job = _get_job(job_id)
    if job["status"] == JobStatus.running:
        raise HTTPException(status_code=409, detail="Audit already running")
    log_q = _log_queues[job_id]
    _executor.submit(run_audit_job, job_id, _job_dir(job_id), job, log_q, _save_job)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Step 7 — Results
# ---------------------------------------------------------------------------

@app.get("/jobs/{job_id}/results")
async def get_results(job_id: str) -> dict:
    job = _get_job(job_id)
    if job["status"] != JobStatus.done:
        raise HTTPException(status_code=425, detail=f"Job status: {job['status']}")
    results = job.get("results", [])
    # Tag each result with its originating job_id so the frontend can fetch
    # sample images from the correct dataset even in cross-session comparisons.
    for r in results:
        r["job_id"] = job_id
    return {"job_id": job_id, "results": results}


@app.get("/jobs/{job_id}/sample_image/{index}")
async def get_sample_image(job_id: str, index: int):
    """Return a single sample image from the job's dataset as PNG."""
    import io
    import joblib
    import numpy as np
    from PIL import Image
    from fastapi.responses import Response

    job = _get_job(job_id)
    data_path = job.get("data_path")
    if not data_path or not Path(data_path).exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = joblib.load(data_path)

    try:
        img = dataset.data[index]
        import torch
        if isinstance(img, torch.Tensor):
            img = img.numpy()
    except Exception:
        try:
            img = dataset.x[index]
        except Exception:
            raise HTTPException(status_code=404, detail=f"Cannot read sample at index {index}")

    if img.dtype != np.uint8:
        img = (np.array(img, dtype=np.float32) * 255).clip(0, 255).astype(np.uint8)
    # C×H×W → H×W×C
    if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[1] > 4:
        img = img.transpose(1, 2, 0)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# ---------------------------------------------------------------------------
# WebSocket — live log stream
# ---------------------------------------------------------------------------

@app.websocket("/jobs/{job_id}/logs")
async def log_stream(websocket: WebSocket, job_id: str) -> None:
    if job_id not in _jobs:
        await websocket.close(code=4004)
        return
    await websocket.accept()
    log_q = _log_queues[job_id]
    loop = asyncio.get_event_loop()
    try:
        while True:
            try:
                msg = await loop.run_in_executor(None, lambda: log_q.get(timeout=0.2))
                await websocket.send_text(msg)
            except queue.Empty:
                status = _jobs[job_id]["status"]
                if status in (JobStatus.done, JobStatus.failed):
                    await websocket.send_text(f"__STATUS__{status}")
                    break
                await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# Serve React SPA (in production; dev uses Vite dev server)
# ---------------------------------------------------------------------------

_FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="static")
