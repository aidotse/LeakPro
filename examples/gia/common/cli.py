#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""The command line every GIA example shares."""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]


def _module_name() -> str:
    main = sys.modules.get("__main__")
    spec = getattr(main, "__spec__", None)
    return spec.name if spec is not None else "example"


def parse(
    description: str,
    argv: list[str] | None = None,
    *,
    extra: Callable[[argparse.ArgumentParser], None] | None = None,
    plots: bool = True,
) -> argparse.Namespace:
    """Parse the shared flags plus the example's own (added by ``extra``); ``command`` records how it was invoked.

    ``plots=False`` is for scripts that draw no figures: they do not offer ``--replot``.
    """
    module = _module_name()
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--quick", action="store_true", help="Offline CPU smoke run on synthetic data (at most 2 minutes).")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", type=Path, default=REPO / "examples/gia/outputs" / module.rsplit(".", 1)[-1],
                   help="Directory for results.json and plots.")
    p.add_argument("--num-images", type=int, default=None)
    p.add_argument("--data-dir", type=Path, default=REPO / "data")
    p.add_argument("--iter-scale", type=float, default=1.0,
                   help="Scale every full-run optimisation budget (e.g. 0.25 for a shorter run on a slow GPU); ignored by --quick.")
    p.add_argument("--deterministic", action="store_true",
                   help="Bit-reproducible run: deterministic CUDA kernels, cudnn without autotuning (slower; run_all always sets it).")
    if plots:
        p.add_argument("--replot", action="store_true", help="Redraw plots from saved results; no attack is run.")
    if extra is not None:
        extra(p)
    args = p.parse_args(argv)
    if args.deterministic:
        set_deterministic(args.seed)
    args.command = "python -m " + module + "".join(f" {x}" for x in (argv if argv is not None else sys.argv[1:]))
    return args


def set_deterministic(seed: int | None = None) -> None:
    """Make the process reproducible: deterministic kernels (warn, not fail, on ops without one) and seeded RNGs.

    ``CUBLAS_WORKSPACE_CONFIG`` is read when cuBLAS is first used, so this must run before any CUDA work.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if seed is not None:
        torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    """``auto`` is cuda when available, else cpu."""
    return torch.device("cuda" if name == "cuda" or (name == "auto" and torch.cuda.is_available()) else "cpu")


def device_for(args: argparse.Namespace) -> torch.device:
    """The device to run on: ``--quick`` stays on the CPU unless ``--device cuda`` is explicit."""
    return resolve_device("cpu" if args.quick and args.device == "auto" else args.device)
