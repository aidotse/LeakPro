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

    # Return first class that has a 'train' method
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and hasattr(obj, "train") and name != "object":
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
            attack_list = [a["attack"] for a in model_spec.get("attacks", [])]

            log_q.put(f"[worker] Auditing model: {model_name}")

            # Build audit YAML
            base_yaml = _LEAKPRO_ROOT / "examples/mia/cifar/audit.yaml"
            with open(base_yaml) as f:
                config = yaml.safe_load(f)

            config["target"]["target_folder"] = target_folder
            if attack_list:
                config["audit"]["attack_list"] = attack_list

            temp_yaml = job_dir / f"_audit_{uuid.uuid4().hex}.yaml"
            try:
                with open(temp_yaml, "w") as f:
                    yaml.dump(config, f)

                handler_cls = _get_handler_cls(job_dir, preset)

                # Add leakpro root to path if needed
                if str(_LEAKPRO_ROOT) not in sys.path:
                    sys.path.insert(0, str(_LEAKPRO_ROOT))

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
