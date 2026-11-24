#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Optional dependencies are imported only when the feature that needs them runs."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

OPTIONAL = ["diffusers", "cma", "lpips", "transformers", "safetensors", "torchvision"]
REPO_ROOT = Path(__file__).resolve().parents[4]


def test_importing_the_package_and_every_preset_loads_no_optional_dependency():
    code = (
        "import sys\n"
        "import leakpro.attacks.gia_attacks.modular as m\n"
        "import leakpro.attacks.gia_attacks.modular.presets as p\n"
        "import leakpro.attacks.gia_attacks.modular.components.drivers, leakpro.attacks.gia_attacks.modular.components.evaluators\n"
        f"print([n for n in {OPTIONAL!r} if n in sys.modules])\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT)
    assert out.stdout.strip().splitlines()[-1] == "[]"


def test_missing_lpips_raises_an_actionable_import_error(monkeypatch):
    import torch

    from leakpro.attacks.gia_attacks.modular.components.utils.image_metrics import lpips_per_image

    monkeypatch.setitem(sys.modules, "lpips", None)   # makes `import lpips` raise ImportError
    with pytest.raises(ImportError, match="pip install lpips"):
        lpips_per_image(torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 8, 8))
