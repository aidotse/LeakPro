#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Aggregators that collapse parallel axes (seeds, epochs) of a driver result into one reconstruction."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from leakpro.attacks.gia_attacks.modular.components.utils.matching import hungarian_align
from leakpro.attacks.gia_attacks.modular.components.utils.registration import align_translation
from leakpro.attacks.gia_attacks.modular.core.axes import AxisManager
from leakpro.attacks.gia_attacks.modular.core.interfaces import Aggregator
from leakpro.attacks.gia_attacks.modular.core.result import DriverResult, select_artifacts

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
    from leakpro.attacks.gia_attacks.modular.core.state import Context, TargetInfo


class MeanAggregator(Aggregator):
    """Collapses a parallel axis by averaging, optionally after aligning every slice to a reference slice.

    recon is reduced via .mean(dim=axis_dim).  The target is reduced by delegating
    to stage.target_source.reduce(target, dim=axis_dim, method="mean"), which lets
    each TargetSource define its own averaging semantics (e.g. JointTarget averages
    its soft-label distribution; FixedTarget returns the target unchanged).

    With ``max_shift > 0``, each slice is first shifted (integer, circular, at most ``max_shift``
    pixels per direction) to best match the reference, the slice with the lowest ``slice_losses``
    (slice 0 when the driver reports none). Independent restarts recover the same image up to a
    small translation, and averaging them unaligned blurs it.

    If the named axis is not registered both recon and target are returned as-is.

    Args:
        axis: Name of the axis to collapse.  Defaults to ``"seed"`` to match
            ``GradientDriver.parallel_axis``; set both together when driving a
            different parallel dimension.
        max_shift: Largest translation, in pixels, tried per direction; 0 averages without aligning.

    """

    def __init__(self, axis: str = "seed", max_shift: int = 0) -> None:
        if max_shift < 0:
            raise ValueError(f"max_shift must be non-negative, got {max_shift}.")
        self.axis = axis
        self.max_shift = max_shift

    def collapsed_axes(self) -> tuple[str, ...]:
        """Axis collapsed by this aggregator."""
        return (self.axis,)

    def aggregate(self, result: DriverResult, ctx: Context, stage: OptimStage) -> DriverResult:
        """Averages the reconstruction over the axis, optionally after aligning each slice to the best-loss slice by shift."""
        seed_dim = ctx.axes.get_dim(self.axis)
        if seed_dim is None:
            return result
        reduced_target = stage.target_source.reduce(result.target, dim=seed_dim, method="mean")
        recon = result.input
        if self.max_shift > 0:
            slices = recon.movedim(seed_dim, 0)
            ref_idx = 0 if result.slice_losses is None else int(result.slice_losses.argmin().item())
            reference = slices[ref_idx]
            recon = torch.stack([align_translation(reference, s, self.max_shift) for s in slices]).movedim(0, seed_dim)
        return _collapsed(result, recon.mean(dim=seed_dim), reduced_target)


class BestLossAggregator(Aggregator):
    """Collapses a parallel axis by selecting the slice with the lowest PER_SLICE loss.

    The losses are the driver's ``slice_losses``, measured at the iterate each slice returns, so
    no slice is re-scored here. Input and artifacts are sliced to the winning index, and the target
    through stage.target_source.slice() — fixed targets are returned unchanged, jointly-optimized
    targets are sliced to the winner.

    If the named axis is not registered the result is returned as-is.

    Args:
        axis: Name of the axis to collapse; see ``MeanAggregator``.

    """

    def __init__(self, axis: str = "seed") -> None:
        self.axis = axis

    def collapsed_axes(self) -> tuple[str, ...]:
        """Axis collapsed by this aggregator."""
        return (self.axis,)

    def aggregate(self, result: DriverResult, ctx: Context, stage: OptimStage) -> DriverResult:
        """Keeps the slice with the lowest per-slice loss along the axis."""
        seed_dim = ctx.axes.get_dim(self.axis)
        if seed_dim is None:
            return result
        if result.slice_losses is None:
            raise ValueError(
                f"BestLossAggregator needs the driver's per-slice losses along axis {self.axis!r}, "
                "but the DriverResult has slice_losses=None."
            )
        best = int(result.slice_losses.argmin().item())
        return DriverResult(
            input=result.input.select(seed_dim, best),
            target=stage.target_source.slice(result.target, seed_dim, best),
            artifacts=select_artifacts(result.artifacts, seed_dim, best, prefix=tuple(ctx.axes.shape)),
            loss=float(result.slice_losses[best].item()),
            iterations=result.iterations,
            slice_losses=None,
        )


class ChainAggregator(Aggregator):
    """Applies a sequence of aggregators in order.

    The (recon, target) output of each aggregator is fed as input to the next,
    so axes are collapsed one at a time.  Each aggregator sees a context whose axes are the ones
    left after the earlier aggregators collapsed theirs, so it addresses the right tensor
    dimensions whatever order the axes were declared in.  Useful for combining, e.g.,
    BestLossAggregator (collapses seed) followed by EpochMatchingAggregator
    (collapses epoch).
    """

    def __init__(self, aggregators: list[Aggregator]) -> None:
        self.aggregators = aggregators
        # Raises for a member that does not declare its axes, before any run depends on it.
        self._collapsed = tuple(name for agg in aggregators for name in agg.collapsed_axes())

    def collapsed_axes(self) -> tuple[str, ...]:
        """Axes collapsed by the wrapped aggregators."""
        return self._collapsed

    def aggregate(self, result: DriverResult, ctx: Context, stage: OptimStage) -> DriverResult:
        """Applies the wrapped aggregators in order, dropping each collapsed axis from the context."""
        for agg in self.aggregators:
            result = agg.aggregate(result, ctx, stage)
            collapsed = set(agg.collapsed_axes())
            remaining = AxisManager(*(a for a in ctx.axes.axes if a.name not in collapsed))
            ctx = replace(ctx, axes=remaining)
        return result


class EpochMatchingAggregator(Aggregator):
    """Collapses the epoch axis via Hungarian-matched averaging.

    In FedAvg attacks each epoch may process the same images in a different order.
    This aggregator aligns every epoch's reconstructions back to epoch 0's ordering
    using the Hungarian algorithm (scipy.optimize.linear_sum_assignment), then
    averages the matched images across all epochs.

    The target is taken directly from epoch 0, which is the natural reference after
    the reordering step.

    If no epoch axis is registered both recon and target are returned unchanged.
    """

    def collapsed_axes(self) -> tuple[str, ...]:
        """Axis collapsed by this aggregator."""
        return ("epoch",)

    def aggregate(self, result: DriverResult, ctx: Context, stage: OptimStage) -> DriverResult:
        """Aligns every epoch's ordering to epoch 0 and averages the aligned reconstructions."""
        epoch_dim = ctx.axes.get_dim("epoch")
        if epoch_dim is None:
            return result
        recon = result.input

        epoch_size = ctx.axes.get_size("epoch")

        # Epoch 0 is the reference ordering.
        ref = recon.select(epoch_dim, 0)           # [..., N, C, H, W] with epoch dim removed
        matched_sum = ref.detach().clone().float()  # accumulate in float for numerical stability

        for e in range(1, epoch_size):
            epoch_e = recon.select(epoch_dim, e)   # same shape as ref
            aligned_e, _ = hungarian_align(ref.detach(), epoch_e.detach())
            matched_sum = matched_sum + aligned_e.float()

        averaged = (matched_sum / epoch_size).to(recon.dtype)

        # Delegate target reduction to the TargetSource so each source
        # defines its own epoch-0 selection semantics (e.g. JointTarget
        # selects its soft tensor; FixedTarget returns the target unchanged).
        reduced_target = stage.target_source.reduce(result.target, dim=epoch_dim, method="match_epoch_0")
        return _collapsed(result, averaged, reduced_target)


def _collapsed(result: DriverResult, recon: Tensor, target: TargetInfo | None) -> DriverResult:
    """``result`` with its axis collapsed into ``recon``/``target``.

    Tensor artifacts carry the collapsed axis and no longer match the reconstruction, so they are
    dropped; a later stage that needs them fails at ``bind`` instead of resuming from a mismatch.
    The loss stays the driver's (mean over slices).
    """
    artifacts = {k: v for k, v in result.artifacts.items() if not isinstance(v, Tensor)}
    return DriverResult(input=recon, target=target, artifacts=artifacts, loss=result.loss,
                        iterations=result.iterations, slice_losses=None)
