#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""The legacy-API examples under examples/gia/legacy still import."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gia_slow

ROOT = Path(__file__).resolve().parents[4]
EXAMPLES = ROOT / "examples/gia"


def test_legacy_examples_import() -> None:
    out = subprocess.run([sys.executable, str(EXAMPLES / "legacy/check_imports.py")],
                         capture_output=True, text=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, check=False)
    assert out.returncode == 0, out.stdout
