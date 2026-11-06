#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Target source that supplies a fixed, observed label set."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from torch import Tensor

from leakpro.attacks.gia_attacks.modular.core.interfaces import TargetSource

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.state import Context, State, TargetInfo


class FixedTarget(TargetSource):
    """Fixed supervision signal — reads hard labels from state.target at bind time."""

    def __init__(self) -> None:
        self._value: TargetInfo | None = None

    def bind(self, state: State, _ctx: Context) -> FixedTarget:
        """Return a copy holding the labels from state.target."""
        bound = copy.copy(self)
        if state.target is None:
            raise ValueError("FixedTarget requires state.target to be set.")
        bound._value = state.target
        return bound

    def current(self) -> TargetInfo:
        """Return the stored target."""
        return self._value

    def slice(self, target: TargetInfo, dim: int, indices: int | Tensor) -> TargetInfo:
        """Select indices along an axis of a prefixed target; unprefixed targets pass through."""
        # Fixed targets are shared across the axis prefix, so a seed/epoch slice is
        # a no-op.  A prefixed target here means one was handed over from a prior
        # JointTarget stage — slicing it generically is correct, and silently
        # returning it unchanged would leave a shape mismatch for the loss.
        if target is not None and target.axis_ndim > 0:
            return target.index_axis(dim, indices)
        return target

    def parameters(self) -> list[dict]:
        """Return no optimizable parameters."""
        return []
