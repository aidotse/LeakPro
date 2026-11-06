#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Inputs handed to loss components and the scope each loss applies to."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from torch import Tensor

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, Context, TargetInfo


class LossScope(Enum):
    """Controls which pass a LossTerm participates in inside the driver's closure.

    PER_SLICE (default):
        Evaluated once per seed slice [N,C,H,W] inside the per-seed loop.
        Has access to a real dummy_obs (model forward included).
        Backward is called per slice, freeing the activation graph immediately → VRAM bounded.
        All losses that depend on gradient matching or BN hooks must be PER_SLICE.

    GROUP:
        Evaluated once on the FULL live reconstruction [*axes,N,C,H,W] after the per-slice loop.
        dummy_obs is None (no model forward — data-space only).
        A single backward call on the full tensor allows gradients to flow across seeds.
        Use for cross-seed regularizers (e.g. GroupConsistencyLoss).
    """

    PER_SLICE = "per_slice"
    GROUP = "group"


@dataclass(frozen=True)
class LossInputs:
    """Immutable input bundle passed to every LossTerm.compute() call.

    PER_SLICE terms receive:
      - recon:      seed-sliced tensor [N,C,H,W] (or [E,N,C,H,W] when an epoch axis is also present)
      - dummy_obs:  ClientObservation from simulate_dummy_signal on recon_s
      - seed_index: which seed slice this is (0-based)
      - full_recon: the full live [*axes,N,C,H,W] tensor (NOT detached, for read-only reference)

    GROUP terms receive:
      - recon:      the full live [*axes,N,C,H,W] tensor (same object as full_recon)
      - dummy_obs:  None
      - seed_index: None
      - full_recon: same as recon
    """

    recon: Tensor
    target: "TargetInfo"
    target_obs: "ClientObservation"
    ctx: "Context"
    dummy_obs: "ClientObservation | None" = None
    seed_index: int | None = None
    full_recon: "Tensor | None" = None
    # Which axis `seed_index` indexes.  The driver's parallel axis is
    # configurable (GradientDriver.parallel_axis), so a term that cares about
    # the axis identity reads it here instead of assuming "seed".
    slice_axis: str | None = None
    # The model-input tensor matching ``recon`` (sliced like it), for terms evaluated in model-input
    # space. None where the caller has no model-input view of the reconstruction.
    model_input: "Tensor | None" = None
