#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Loading the recorded v1 reference fixtures."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "v1"


@lru_cache(maxsize=1)
def manifest() -> dict[str, Any]:
    """The recording manifest: provenance plus one entry per case."""
    return json.loads((FIXTURE_DIR / "manifest.json").read_text())


def cases(kind: str) -> list[str]:
    """Names of all recorded cases of one kind (signal, labels, loss_points, trajectory)."""
    return sorted(c for c, meta in manifest()["cases"].items() if meta["kind"] == kind)


def load(case: str) -> dict[str, Any]:
    """Load one case. weights_only=False is required for the pickled nn.Module; the files are repo-owned."""
    return torch.load(FIXTURE_DIR / manifest()["cases"][case]["file"], weights_only=False)
