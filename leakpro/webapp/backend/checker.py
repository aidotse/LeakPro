"""Model compatibility check — runs user-uploaded arch.py in a subprocess."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import textwrap
from pathlib import Path

from .models import CompatResult

logger = logging.getLogger("leakpro.webapp.checker")


_RUNNER_SCRIPT = textwrap.dedent("""
import sys, json, importlib.util, traceback, pathlib
import torch
import torch.nn.functional as F

arch_path    = sys.argv[1]
weights_path = sys.argv[2]
shape_json   = sys.argv[3]
data_path    = sys.argv[4] if len(sys.argv) > 4 else "none"

data_shape = json.loads(shape_json)

try:
    spec = importlib.util.spec_from_file_location("user_arch", arch_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    import torch.nn as nn
    import inspect
    candidates = [
        v for v in vars(mod).values()
        if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module
    ]
    if not candidates:
        raise ValueError("No nn.Module subclass found in arch file")

    # --- Priority 1: init_params from model_metadata.pkl ---
    init_params = {}
    if weights_path != "none":
        import joblib
        meta_path = pathlib.Path(weights_path).parent / "model_metadata.pkl"
        if meta_path.exists():
            try:
                meta = joblib.load(str(meta_path))
                init_params = getattr(meta, "init_params", {}) or {}
            except Exception:
                pass

    # --- Priority 2: auto-detect from state dict (no metadata yet) ---
    _cached_state = None
    if not init_params and weights_path != "none":
        try:
            _cached_state = torch.load(weights_path, map_location="cpu", weights_only=True)
        except Exception:
            try:
                _cached_state = torch.load(weights_path, map_location="cpu", weights_only=False)
            except Exception:
                pass
        if isinstance(_cached_state, dict) and "state_dict" in _cached_state:
            _cached_state = _cached_state["state_dict"]
        if isinstance(_cached_state, dict):
            _sd = _cached_state
            _auto = {}
            for _key in ("fc.weight", "classifier.weight", "head.weight"):
                if _key in _sd:
                    _auto["num_classes"] = int(_sd[_key].shape[0])
                    break
            # WideResNet: fc.weight shape = [num_classes, 64 * widen_factor]
            if "block1.layer.0.conv1.weight" in _sd and "fc.weight" in _sd:
                _last_ch = int(_sd["fc.weight"].shape[1])
                _auto["widen_factor"] = max(1, _last_ch // 64)
                _n = len(set(k.split(".")[2] for k in _sd if k.startswith("block1.layer.")))
                _auto["depth"] = _n * 6 + 4
                _auto.setdefault("dropRate", 0.0)
            if _auto:
                init_params = _auto

    # --- Pick the class that accepts all detected init_params keys ---
    cls = candidates[-1]
    if init_params:
        for _cls in candidates:
            _sig = set(inspect.signature(_cls.__init__).parameters) - {"self"}
            if set(init_params.keys()).issubset(_sig):
                cls = _cls
                break

    # --- Instantiate: try init_params, fill missing required args with common defaults ---
    def _try_init(c, params):
        try:
            return c(**params)
        except TypeError:
            return None

    model = _try_init(cls, init_params)
    if model is None:
        _COMMON = {"num_classes": 10, "depth": 28, "widen_factor": 2, "dropRate": 0.0, "in_channels": 3}
        kwargs = dict(init_params)
        for _name, _param in inspect.signature(cls.__init__).parameters.items():
            if _name == "self":
                continue
            if _param.default is inspect.Parameter.empty and _name not in kwargs:
                kwargs[_name] = _COMMON.get(_name, 10)
        model = cls(**kwargs)
    model.eval()

    if weights_path != "none":
        # Reuse cached state dict if already loaded during auto-detection
        if _cached_state is not None:
            state = _cached_state
        else:
            try:
                state = torch.load(weights_path, map_location="cpu", weights_only=True)
            except Exception:
                state = torch.load(weights_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
        if isinstance(state, dict):
            model.load_state_dict(state, strict=False)

    dummy = torch.randn(1, *data_shape)
    with torch.no_grad():
        out = model(dummy)

    param_count = sum(p.numel() for p in model.parameters())
    result = {
        "ok": True,
        "input_shape": list(data_shape),
        "output_shape": list(out.shape[1:]),
        "param_count": param_count,
    }

    # Optional: run 2 real data samples through the model. The dataset is loaded
    # and normalised by the same code the training path uses (dataio.load_dataset),
    # so this check sees exactly the tensors the model will be trained on.
    if data_path != "none":
        try:
            from leakpro.webapp.backend.dataio import load_dataset
            dataset = load_dataset(pathlib.Path(data_path))
            sample_outputs = []
            for i in range(min(2, len(dataset))):
                x, y = dataset[i]
                x = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits = model(x)
                probs = F.softmax(logits, dim=1)
                top1 = int(probs.argmax(dim=1).item())
                conf = round(float(probs.max(dim=1).values.item()), 4)
                true_label = int(y) if y is not None else None
                sample_outputs.append({
                    "sample": i,
                    "top1_class": top1,
                    "confidence": conf,
                    "true_label": true_label,
                })
            result["sample_outputs"] = sample_outputs
        except Exception as _se:
            # The main check passed; report why the sample pass did not, never hide it.
            result["sample_outputs"] = None
            result["sample_error"] = f"{type(_se).__name__}: {_se}"
    print(json.dumps(result))
except Exception:
    print(json.dumps({"ok": False, "error": traceback.format_exc()}))
""")


def run_check(
    arch_path: Path,
    weights_path: Path | None,
    data_shape: list[int],
    data_path: Path | None = None,
) -> CompatResult:
    """Run compat check in an isolated subprocess. Returns CompatResult."""
    runner = Path(arch_path).parent / "_leakpro_check_runner.py"
    runner.write_text(_RUNNER_SCRIPT)
    cmd = [
        sys.executable,
        str(runner),
        str(arch_path),
        str(weights_path) if weights_path else "none",
        json.dumps(data_shape),
    ]
    if data_path is not None:
        cmd.append(str(data_path))
    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        raw = result.stdout.strip()
        if not raw:
            return CompatResult(ok=False, error=result.stderr or "No output from check subprocess")
        data = json.loads(raw)
        if data.get("sample_error"):
            logger.warning("Compatibility check: model loaded but the sample pass failed: %s",
                           data["sample_error"])
        return CompatResult(**data)
    except subprocess.TimeoutExpired:
        return CompatResult(ok=False, error="Compatibility check timed out (>60s)")
    except Exception as e:
        return CompatResult(ok=False, error=str(e))
    finally:
        runner.unlink(missing_ok=True)
