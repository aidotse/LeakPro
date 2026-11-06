#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""What a driver tells callbacks after each step."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StepEvent:
    """One completed driver step, as seen by callbacks.

    ``artifacts`` holds the representation's artifacts after the step. The tensors are detached but
    not cloned, so a callback that keeps them across steps must clone them itself.
    """

    stage_idx: int
    step: int                       # 0-based optimizer step / generation / reverse-diffusion step
    loss: float                     # objective reported for this step (NaN when the driver has none)
    components: Mapping[str, float]  # weighted per-term values (empty when not available)
    artifacts: Mapping[str, Any]
