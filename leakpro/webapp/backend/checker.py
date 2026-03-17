"""Model compatibility check — runs user-uploaded arch.py in a subprocess."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

from .models import CompatResult


_RUNNER_SCRIPT = textwrap.dedent("""
import sys, json, importlib.util, traceback
import torch

arch_path, weights_path, shape_json = sys.argv[1], sys.argv[2], sys.argv[3]
data_shape = json.loads(shape_json)

try:
    spec = importlib.util.spec_from_file_location("user_arch", arch_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Find the first nn.Module subclass defined in the file
    import torch.nn as nn
    cls = next(
        v for v in vars(mod).values()
        if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module
    )
    model = cls()
    model.eval()

    if weights_path != "none":
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict):
            model.load_state_dict(state, strict=False)

    dummy = torch.randn(1, *data_shape)
    with torch.no_grad():
        out = model(dummy)

    param_count = sum(p.numel() for p in model.parameters())
    print(json.dumps({
        "ok": True,
        "input_shape": list(data_shape),
        "output_shape": list(out.shape[1:]),
        "param_count": param_count,
    }))
except Exception:
    print(json.dumps({"ok": False, "error": traceback.format_exc()}))
""")


def run_check(arch_path: Path, weights_path: Path | None, data_shape: list[int]) -> CompatResult:
    """Run compat check in an isolated subprocess. Returns CompatResult."""
    runner = Path(arch_path).parent / "_leakpro_check_runner.py"
    runner.write_text(_RUNNER_SCRIPT)
    try:
        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                str(runner),
                str(arch_path),
                str(weights_path) if weights_path else "none",
                json.dumps(data_shape),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        raw = result.stdout.strip()
        if not raw:
            return CompatResult(ok=False, error=result.stderr or "No output from check subprocess")
        data = json.loads(raw)
        return CompatResult(**data)
    except subprocess.TimeoutExpired:
        return CompatResult(ok=False, error="Compatibility check timed out (>30s)")
    except Exception as e:
        return CompatResult(ok=False, error=str(e))
    finally:
        runner.unlink(missing_ok=True)
