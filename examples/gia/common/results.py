#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""The results file every example writes and the one-line summary it prints."""
from __future__ import annotations

import datetime
import json
import platform
import statistics
import subprocess
from pathlib import Path

import torch

from examples.gia.common.cli import REPO

Rows = dict[str, dict[str, list[float]]]


def environment() -> dict:
    """Date, commit, torch version, device, python and determinism of this run (``commit`` is None outside git)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                                check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "date": datetime.date.today().isoformat(),
        "commit": commit,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "python": platform.python_version(),
        "deterministic": torch.are_deterministic_algorithms_enabled() and torch.backends.cudnn.deterministic,
    }


def _summary(values: list[float]) -> dict:
    return {"mean": statistics.fmean(values), "std": statistics.pstdev(values), "values": list(values)}


def mean(cell: list[float] | dict) -> float:
    """Mean of a metric, given its raw values (before saving) or its saved summary (after ``load``)."""
    return cell["mean"] if isinstance(cell, dict) else statistics.fmean(cell)


def save(out: Path, rows: Rows, config: dict) -> Path:
    """Write ``out/results.json``: ``{example, config, env, rows: {name: {metric: {mean, std, values}}}}``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "example": out.name,
        "config": config,
        "env": environment(),
        "rows": {name: {metric: _summary(v) for metric, v in metrics.items()} for name, metrics in rows.items()},
    }
    path = out / "results.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def load(out: Path) -> dict:
    """Read the ``results.json`` written by ``save``."""
    return json.loads((Path(out) / "results.json").read_text())


def emit(rows: Rows) -> None:
    """Print one ``RESULT {json}`` line mapping ``<name>.<metric>`` to the mean."""
    flat = {f"{name}.{metric}": statistics.fmean(v) for name, metrics in rows.items() for metric, v in metrics.items()}
    print("RESULT " + json.dumps(flat))


def per_image(evaluation) -> dict[str, list[float]]:  # noqa: ANN001
    """Per-image ``psnr``, ``ssim`` (and ``lpips`` when present) of an ``EvalResult``."""
    return {m: list(evaluation.extras[f"{m}_per_image"]) for m in ("psnr", "ssim", "lpips")
            if f"{m}_per_image" in evaluation.extras}
