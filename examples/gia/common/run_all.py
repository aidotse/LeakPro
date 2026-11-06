#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Run every GIA example at full size, one output directory each, resumable.

    python -m examples.gia.common.run_all                  # everything not yet done at this commit
    python -m examples.gia.common.run_all --only fedavg    # selected examples
    python -m examples.gia.common.run_all --any-commit     # resume: accept results from any commit
    python -m examples.gia.common.run_all --dry-run        # print the commands only

Each example section declares its runs in its own ``_runs.py``; this script runs those of every section present.
An example is done when ``outputs/NAME/results.json`` exists, was written at the current commit
(or at any commit with ``--any-commit``) and the latest attempt did not fail (``run.status``). GPU examples wait until enough GPU memory is free; nothing
is ever killed on the GPU. A failing example is logged and the run continues.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[3]
OUTPUTS = REPO / "examples/gia/outputs"
SECTIONS: list[ModuleType] = [importlib.import_module(f"examples.gia.{f.parent.name}._runs")
                              for f in sorted((REPO / "examples/gia").glob("*/_runs.py"))]

# name (= output directory) -> (module, flags of the full run). Flags are the "full run" setting of each example.
FULL: dict[str, tuple[str, list[str]]] = {name: run for s in SECTIONS for name, run in s.FULL.items()}
# Needs a manual dataset download: only run when it is present.
OPTIONAL: dict[str, tuple[str, list[str], Path]] = {
    name: run for s in SECTIONS for name, run in getattr(s, "NEEDS_DATA", {}).items()}


def registry() -> dict[str, tuple[str, list[str]]]:
    """The full-run registry plus the optional entries whose data is present."""
    return {**FULL, **{n: (m, f) for n, (m, f, needs) in OPTIONAL.items() if needs.exists()}}


def needs_gpu(flags: list[str]) -> bool:
    """CPU-only entries pass ``--device cpu`` and skip the GPU wait."""
    return not any(a == "--device" and flags[i + 1] == "cpu" for i, a in enumerate(flags[:-1]))


def deterministic_flag(module: str) -> list[str]:
    """Full runs are deterministic (the standalone YOLO demo has its own parser and CPU-only seeding)."""
    return [] if module.endswith("yolo_attack") else ["--deterministic"]


def command(name: str, out_root: Path = OUTPUTS) -> list[str]:
    """The command line that runs one registry entry."""
    module, flags = registry()[name]
    return ["nice", "-n", "10", sys.executable, "-m", module, "--out", str(out_root / name), *deterministic_flag(module), *flags]


def head_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def is_done(out_dir: Path, head: str | None, any_commit: bool = False) -> bool:
    """``results.json`` exists, was written at ``head`` (any commit with ``any_commit``) and the last attempt succeeded."""
    f = out_dir / "results.json"
    if not f.exists():
        return False
    status = out_dir / "run.status"
    if status.exists() and status.read_text().strip() != "ok":
        return False  # the results are left over from an earlier attempt; the latest one failed
    if any_commit:
        return True
    try:
        return json.loads(f.read_text()).get("env", {}).get("commit") == head
    except (OSError, ValueError):
        return False


def free_gpu_gb() -> float | None:
    """Free memory of GPU 0 in GB, or None without nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout.splitlines()[0]
        used, total = (float(x) for x in out.split(","))
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None
    return (total - used) / 1024


def wait_for_gpu(min_free_gb: float, poll_s: float = 60.0, max_wait_s: float = 6 * 3600.0) -> bool:
    """Block until ``min_free_gb`` is free (True), or ``max_wait_s`` has passed (False). No nvidia-smi: True."""
    waited = 0.0
    while True:
        free = free_gpu_gb()
        if free is None or free >= min_free_gb:
            return True
        if waited >= max_wait_s:
            return False
        print(f"  GPU busy ({free:.1f} GB free < {min_free_gb} GB); waiting", flush=True)  # noqa: T201
        time.sleep(poll_s)
        waited += poll_s


def run_one(name: str, out_root: Path, timeout_s: float) -> tuple[str, float]:
    """Run one entry, log to ``run.log``; returns (status, seconds). Never raises."""
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    start = time.time()
    with open(out_dir / "run.log", "w") as log:
        try:
            rc = subprocess.run(command(name, out_root), cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=timeout_s, check=False).returncode
            status = "ok" if rc == 0 else f"failed (exit {rc})"
        except subprocess.TimeoutExpired:
            status = "timeout"
        except OSError as e:
            status = f"error ({e})"
    (out_dir / "run.status").write_text(status + "\n")
    return status, time.time() - start


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", nargs="+", default=None, help="Run only these entries.")
    p.add_argument("--min-free-gpu-gb", type=float, default=10.0)
    p.add_argument("--timeout-min", type=float, default=90.0, help="Per-example timeout (deterministic kernels run about 3x slower).")
    p.add_argument("--any-commit", action="store_true", help="Treat results from any commit as done.")
    p.add_argument("--out-root", type=Path, default=OUTPUTS)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    reg = registry()
    names = a.only or list(reg)
    if unknown := [n for n in names if n not in reg]:
        p.error(f"unknown entries {unknown}; available: {list(reg)}")
    head, summary = head_commit(), []
    for name in names:
        if a.dry_run:
            print(shlex.join(command(name, a.out_root)))  # noqa: T201
            continue
        if is_done(a.out_root / name, head, a.any_commit):
            summary.append((name, "skipped (done)", 0.0))
            continue
        print(f"[{name}] starting", flush=True)  # noqa: T201
        if needs_gpu(reg[name][1]) and not wait_for_gpu(a.min_free_gpu_gb):
            summary.append((name, "skipped (GPU busy)", 0.0))
            continue
        status, secs = run_one(name, a.out_root, a.timeout_min * 60)
        print(f"[{name}] {status} in {secs / 60:.1f} min", flush=True)  # noqa: T201
        summary.append((name, status, secs))
    if summary:
        print("\n" + "\n".join(f"{n:28s} {s:22s} {t / 60:6.1f} min" for n, s, t in summary))  # noqa: T201
    return 1 if any(s.startswith(("failed", "timeout", "error")) for _, s, _ in summary) else 0


if __name__ == "__main__":
    sys.exit(main())
