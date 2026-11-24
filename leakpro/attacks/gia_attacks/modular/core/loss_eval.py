#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""The single evaluator of a stage's weighted loss terms.

Drivers, aggregators, diffusion strategies and tests all score a reconstruction through
``evaluate_stage_losses``, so every path builds the same ``LossInputs`` (including the model-input
view) and the dummy client signal is simulated once per evaluation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import Tensor

from leakpro.attacks.gia_attacks.modular.core.loss_inputs import LossInputs, LossScope

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.interfaces import LossTerm
    from leakpro.attacks.gia_attacks.modular.core.state import Context, TargetInfo


@dataclass(frozen=True)
class LossBreakdown:
    """The weighted stage objective at one point, and its value per term.

    Attributes:
        total: Weighted sum of the evaluated terms (scalar), accumulated in the stage's term order.
        terms: Weighted value per term name.

    """

    total: Tensor
    terms: dict[str, Tensor]


def evaluate_stage_losses(
    losses: Mapping[str, tuple[LossTerm, float]],
    ctx: Context,
    recon: Tensor,
    target: TargetInfo,
    *,
    model_input: Tensor | None = None,
    scope: LossScope = LossScope.PER_SLICE,
    create_graph: bool = True,
    seed_index: int | None = None,
    full_recon: Tensor | None = None,
    slice_axis: str | None = None,
) -> LossBreakdown:
    """Evaluate the stage terms of ``scope`` at ``recon``.

    Args:
        losses: The stage's ``{name: (term, weight)}`` mapping; terms of other scopes are skipped.
        ctx: Run context (task, modality, observation).
        recon: Reconstruction in data space (a slice for PER_SLICE, the full iterate for GROUP).
        target: Supervision matching ``recon``.
        model_input: Model-input view of ``recon``; defaults to ``ctx.modality.to_model_input(recon)``.
        scope: Which terms to evaluate. PER_SLICE simulates the dummy client signal once; GROUP
            passes no dummy signal.
        create_graph: Whether the dummy signal keeps a graph for differentiating through it. A
            gradient-free caller passes False so no second-order graph is built.
        seed_index: Index of ``recon`` along ``slice_axis`` (PER_SLICE).
        full_recon: The full live iterate, for terms that read across slices.
        slice_axis: Name of the axis ``seed_index`` indexes.

    Returns:
        The weighted total and the weighted value per evaluated term. An empty term set gives a
        zero total.

    """
    selected = {n: (t, w) for n, (t, w) in losses.items() if getattr(t, "scope", LossScope.PER_SLICE) is scope}
    total = recon.new_zeros(())
    if not selected:
        return LossBreakdown(total=total, terms={})
    if model_input is None:
        model_input = ctx.modality.to_model_input(recon)
    dummy_obs = None
    if scope is LossScope.PER_SLICE:
        dummy_obs = ctx.task.simulate_dummy_signal(model_input, target, ctx, create_graph=create_graph)
    inputs = LossInputs(
        recon=recon,
        target=target,
        target_obs=ctx.observation,
        ctx=ctx,
        dummy_obs=dummy_obs,
        seed_index=seed_index,
        full_recon=full_recon,
        slice_axis=slice_axis,
        model_input=model_input,
    )
    terms: dict[str, Tensor] = {}
    for name, (term, weight) in selected.items():
        value = weight * term.compute(inputs)
        terms[name] = value
        total = total + value
    return LossBreakdown(total=total, terms=terms)
