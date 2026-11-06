#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Target source for tasks without labels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from leakpro.attacks.gia_attacks.modular.core.interfaces import TargetSource
from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo

if TYPE_CHECKING:
    from torch import Tensor

    from leakpro.attacks.gia_attacks.modular.core.state import Context, State


class NullTarget(TargetSource):
    """No-op target for tasks that do not require a classification label.

    Returns an empty TargetInfo every step and exposes no optimizable parameters.
    Use this with objectives whose loss needs no supervision signal.
    """

    def bind(self, state: State, ctx: Context) -> NullTarget:  # noqa: ARG002 -- interface signature
        """Return this target unchanged."""
        return self

    def current(self) -> TargetInfo:
        """Return an empty TargetInfo."""
        return TargetInfo()

    def slice(self, target: TargetInfo, dim: int, indices: int | Tensor) -> TargetInfo:  # noqa: ARG002 -- interface signature
        """Return the target unchanged."""
        return target

    def reduce(self, target: TargetInfo, dim: int, method: str) -> TargetInfo:  # noqa: ARG002 -- interface signature
        """Return the target unchanged."""
        return target

    def parameters(self) -> list[dict]:
        """Return no optimizable parameters."""
        return []
