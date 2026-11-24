#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Target source that jointly optimizes soft labels."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from leakpro.attacks.gia_attacks.modular.core.interfaces import TargetSource
from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.state import Context, State


class JointTarget(TargetSource):
    """Jointly-optimized soft labels — the optimizer tweaks these alongside the image."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self._soft_label: Tensor | None = None

    def bind(self, state: State, ctx: Context) -> JointTarget:
        """Return a copy with soft labels resumed from the state or initialized from a standard normal."""
        bound = copy.copy(self)
        batch_size = ctx.input_shape[0]
        prefix = ctx.axes.shape
        bound._axis_ndim = len(prefix)

        # If soft targets were optimized in a previous stage (or loaded from state), resume them
        if state.target is not None and state.target.soft is not None:
            bound._soft_label = state.target.soft.detach().clone().to(ctx.device).requires_grad_(True)
            return bound

        # Otherwise, initialize fresh soft labels from N(0, 1)
        bound._soft_label = torch.randn(
            (*prefix, batch_size, bound.num_classes), device=ctx.device, requires_grad=True
        )
        return bound

    def current(self) -> TargetInfo:
        """Return the soft labels and their argmax as the hard value."""
        return TargetInfo(
            value=self._soft_label.argmax(dim=-1),
            soft=self._soft_label,
            axis_ndim=self._axis_ndim,
        )

    def slice(self, target: TargetInfo, dim: int, indices: int | Tensor) -> TargetInfo:
        """Select indices along dim from the jointly-optimized soft label tensors.

        int   — scalar axis index; the selected dimension is removed (.select).
        Tensor — 1D index tensor; the dimension is preserved (torch.index_select).
        """
        return target.index_axis(dim, indices)

    def reduce(self, target: TargetInfo, dim: int, method: str) -> TargetInfo:
        """Collapse jointly-optimized soft labels along a parallel axis.

        Supported methods:
          "mean"           — average soft labels across dim, recompute argmax.
          "match_epoch_0"  — select the dim=0 slice (epoch 0 is the reference ordering
                             after Hungarian matching in EpochMatchingAggregator).
        """
        if method == "mean":
            if target.soft is None:
                raise ValueError("JointTarget.reduce requires target.soft to be set.")
            # Average every continuous field across the axis, then re-derive the hard
            # value from the averaged distribution (a mean of argmaxes is meaningless).
            # Integer fields cannot be averaged — they fall back to the first slice and
            # are expected to be re-derived from a continuous field, as `value` is below.
            reduced = target.map(
                lambda t: t.mean(dim=dim) if t.is_floating_point() else t.select(dim, 0),
                axis_ndim=max(target.axis_ndim - 1, 0),
            )
            reduced.value = reduced.soft.argmax(dim=-1)
            return reduced

        if method == "match_epoch_0":
            if target.soft is None:
                raise ValueError("JointTarget.reduce(match_epoch_0) requires target.soft.")
            reduced = target.index_axis(dim, 0)
            reduced.value = reduced.soft.argmax(dim=-1)
            return reduced

        raise ValueError(f"JointTarget.reduce: unsupported method '{method}'.")

    def parameters(self) -> list[dict]:
        """Return the soft labels as the single optimizable parameter group."""
        return [{"name": "soft_labels", "params": [self._soft_label], "lr_scale": 1.0}]
