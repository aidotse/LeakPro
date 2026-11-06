#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Every example runs end to end with --quick (offline, CPU, synthetic data) and reports a RESULT line."""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gia_slow

ROOT = Path(__file__).resolve().parents[4]
EXAMPLES = ROOT / "examples/gia"
# Every example section declares its runs in its _runs.py (see examples/gia/common/run_all.py).
SECTIONS = {f.parent.name: importlib.import_module(f"examples.gia.{f.parent.name}._runs")
            for f in sorted(EXAMPLES.glob("*/_runs.py"))}
# Optional dependencies a case needs (the case is skipped when one is missing), and scripts that print no metrics.
REQUIRES = {name: deps for runs in SECTIONS.values() for name, deps in getattr(runs, "REQUIRES", {}).items()}
NO_RESULT = {name for runs in SECTIONS.values() for name in getattr(runs, "NO_RESULT", ())}


def _modules() -> dict[str, tuple[str, ...]]:
    """Each section's scripts (subpackages included) with their defaults, then the sections' extra variants."""
    found, extra = {}, {}
    for section, runs in SECTIONS.items():
        for f in sorted((EXAMPLES / section).rglob("*.py")):
            if not f.name.startswith("_"):
                rel = f.relative_to(EXAMPLES).with_suffix("")
                found[rel.as_posix()] = (".".join(("examples", "gia", *rel.parts)),)
        extra.update(getattr(runs, "QUICK_VARIANTS", {}))
    return {**found, **extra}


CASES = _modules()


def test_every_section_and_script_is_covered() -> None:
    """A new example package without a _runs.py, or a section without a discovered script, fails."""
    packages = {p.parent.name for p in EXAMPLES.glob("*/__init__.py")} - {"common"}
    assert packages <= set(SECTIONS), f"example packages without a _runs.py: {sorted(packages - set(SECTIONS))}"
    for section in SECTIONS:
        assert any(name.startswith(f"{section}/") for name in CASES), f"no example script found in {section}"


@pytest.mark.parametrize("name", CASES)
def test_example_runs_quick(name: str, tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    for dep in REQUIRES.get(name, ()):
        if importlib.util.find_spec(dep) is None:
            pytest.skip(f"optional dependency {dep} not installed")
    module, *flags = CASES[name]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": ""}
    cmd = [sys.executable, "-m", module, *flags, "--quick", "--device", "cpu", "--out", str(tmp_path / "out")]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path, env=env, timeout=600, check=False)
    assert out.returncode == 0, out.stderr[-2000:]
    if name in NO_RESULT:
        return
    line = next(ln for ln in out.stdout.splitlines() if ln.startswith("RESULT "))
    assert json.loads(line[len("RESULT "):])
    assert (tmp_path / "out" / "results.json").exists()


def _quick_rows(module: str, flags: list[str], tmp_path: Path) -> dict:
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": ""}
    cmd = [sys.executable, "-m", module, *flags, "--quick", "--device", "cpu", "--out", str(tmp_path / "out")]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path, env=env, timeout=600, check=False)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads((tmp_path / "out" / "results.json").read_text())["rows"]


def test_batch_sweep_wins_over_num_images(tmp_path: Path) -> None:
    rows = _quick_rows("examples.gia.image_classification.scaling", ["--sweep", "batch", "--num-images", "2"], tmp_path)
    assert {k: len(v["psnr"]["values"]) for k, v in rows.items()} == {"batch=1": 1, "batch=2": 2, "batch=3": 3}


def test_num_images_sets_the_number_of_scored_images(tmp_path: Path) -> None:
    rows = _quick_rows("examples.gia.image_classification.compare_attacks", ["--num-images", "3", "--attacks", "DLG"],
                       tmp_path)
    assert rows and all(len(row["psnr"]["values"]) == 3 for row in rows.values())
