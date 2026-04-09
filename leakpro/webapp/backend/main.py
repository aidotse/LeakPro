"""LeakPro webapp — FastAPI backend."""

from __future__ import annotations

import asyncio
import queue
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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

app = FastAPI(title="LeakPro Webapp API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
# Job lifecycle
# ---------------------------------------------------------------------------

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
    return meta


# ---------------------------------------------------------------------------
# Step 2 — Confirm format
# ---------------------------------------------------------------------------

@app.post("/jobs/{job_id}/handler-config")
async def set_handler_config(job_id: str, config: HandlerConfig) -> dict:
    job = _get_job(job_id)
    job["handler_config"] = config.model_dump()
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
    return {"ok": True}


# ---------------------------------------------------------------------------
# Step 4A — Upload existing model weights + compatibility check
# ---------------------------------------------------------------------------

@app.post("/jobs/{job_id}/upload/weights")
async def upload_weights(job_id: str, model_name: str, file: UploadFile) -> dict:
    _get_job(job_id)
    model_dir = _job_dir(job_id) / "models" / model_name
    model_dir.mkdir(exist_ok=True)
    dest = model_dir / "target_model.pt"
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


@app.post("/jobs/{job_id}/weights-path")
async def set_weights_path(job_id: str, body: dict) -> dict:
    _get_job(job_id)
    path = Path(body.get("path", ""))
    model_name = body.get("model_name", "uploaded_model")
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {path}")
    model_dir = _job_dir(job_id) / "models" / model_name
    model_dir.mkdir(exist_ok=True)
    dest = model_dir / "target_model.pt"
    shutil.copy2(path, dest)
    return {"ok": True, "path": str(dest)}


@app.post("/jobs/{job_id}/check", response_model=CompatResult)
async def check_compat(job_id: str, model_name: str) -> CompatResult:
    job = _get_job(job_id)
    job_dir = _job_dir(job_id)
    arch_path = job_dir / "arch.py"
    weights_path = job_dir / "models" / model_name / "target_model.pt"
    data_meta = job.get("data_meta", {})
    data_shape = data_meta.get("shape", [3, 32, 32])

    if not arch_path.exists():
        raise HTTPException(status_code=400, detail="No architecture file uploaded yet")

    result = run_check(
        arch_path=arch_path,
        weights_path=weights_path if weights_path.exists() else None,
        data_shape=data_shape,
    )
    return result


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

            target_folder = _job_dir(job_id) / "models" / params.name
            target_folder.mkdir(parents=True, exist_ok=True)

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

            # Load handler class (skip abstract base classes)
            import inspect
            handler_path = cifar_dir / "cifar_handler.py"
            spec = importlib.util.spec_from_file_location("cifar_handler", handler_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            HandlerCls = next(
                obj for name in dir(mod)
                if isinstance(obj := getattr(mod, name), type)
                and hasattr(obj, "train")
                and not inspect.isabstract(obj)
                and name != "object"
            )

            # Load dataset, apply f_train fraction
            data_path = Path(job.get("data_path", ""))
            if data_path.exists():
                import pickle
                with open(data_path, "rb") as f:
                    dataset = pickle.load(f)
                if not isinstance(dataset, torch.utils.data.Dataset):
                    raise ValueError(f"Loaded object is not a Dataset: {type(dataset)}")
            else:
                # Fallback: download CIFAR-10
                import torchvision
                import torchvision.transforms as T
                transform = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                dataset = torchvision.datasets.CIFAR10(root=str(cifar_dir / "data"), train=True, download=True, transform=transform)

            # Apply f_train fraction
            train_size = max(1, int(len(dataset) * params.f_train))
            if train_size < len(dataset):
                import torch as _torch
                indices = _torch.randperm(len(dataset))[:train_size].tolist()
                dataset = torch.utils.data.Subset(dataset, indices)
            log_q.put(f"[train] Dataset: {len(dataset)} samples  (f_train={params.f_train})")
            loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=2)

            # Load model architecture
            from target_model_class import ResNet18  # noqa: PLC0415
            model = ResNet18()
            criterion = nn.CrossEntropyLoss()
            if params.optimizer == "sgd":
                optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)
            else:
                optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

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

            handler = HandlerCls()
            result = handler.train(loader, model, criterion, optimizer, epochs=params.epochs)

            # Restore tqdm
            if _orig_tqdm:
                _orig_tqdm[0].tqdm = _orig_tqdm[1]

            # Save trained model
            import json as _json
            torch.save(result.model.state_dict(), target_folder / "target_model.pt")
            test_acc = None
            loss_history: list = []
            acc_history: list = []
            if result.metrics:
                test_acc = float(result.metrics.accuracy)
                extra = result.metrics.extra or {}
                loss_history = [float(x) for x in extra.get("loss_history", [])]
                acc_history = [float(x) for x in extra.get("accuracy_history", [])]
            model_entry["status"] = "ready"
            model_entry["test_accuracy"] = test_acc
            log_q.put(f"[train] Done: {params.name}  accuracy={test_acc:.4f}" if test_acc else f"[train] Done: {params.name}")
            # Emit metrics for frontend chart
            log_q.put("__METRICS__" + _json.dumps({
                "model": params.name,
                "loss_history": loss_history,
                "accuracy_history": acc_history,
            }))

        except Exception as e:  # noqa: BLE001
            import traceback
            model_entry = next((m for m in job.get("models", []) if m["name"] == params.name), {})
            model_entry["status"] = "error"
            log_q.put(f"[train] FAILED {params.name}: {e}\n{traceback.format_exc()}")
        finally:
            _sys.stdout = old_stdout
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
    for model in job.get("models", []):
        if model["name"] in cfg_map:
            model["attacks"] = [a.model_dump() for a in cfg_map[model["name"]]]
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
    _executor.submit(run_audit_job, job_id, _job_dir(job_id), job, log_q)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Step 7 — Results
# ---------------------------------------------------------------------------

@app.get("/jobs/{job_id}/results")
async def get_results(job_id: str) -> dict:
    job = _get_job(job_id)
    if job["status"] != JobStatus.done:
        raise HTTPException(status_code=425, detail=f"Job status: {job['status']}")
    return {"job_id": job_id, "results": job.get("results", [])}


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
