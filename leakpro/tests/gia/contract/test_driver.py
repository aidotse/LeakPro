#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Driver contract: a DriverResult of the right shape, loss and budget; the stage configuration is untouched."""
from __future__ import annotations

import math
from dataclasses import replace

import pytest

import torch

from leakpro.attacks.gia_attacks.modular.components.drivers.gradient import GradientDriver
from leakpro.attacks.gia_attacks.modular.components.initializers import Uniform
from leakpro.attacks.gia_attacks.modular.components.representations.pixel import PixelRepresentation
from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
from leakpro.attacks.gia_attacks.modular.core.axes import AxisManager
from leakpro.attacks.gia_attacks.modular.core.interfaces import Driver, LossTerm
from leakpro.attacks.gia_attacks.modular.core.loss_inputs import LossInputs
from leakpro.attacks.gia_attacks.modular.core.result import DriverResult
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.state import State
from leakpro.tests.gia import helpers
from leakpro.tests.gia.contract.conftest import assert_unchanged, client_target, make_ctx, snapshot
from leakpro.tests.gia.contract.registry import build, cases

pytestmark = pytest.mark.parametrize(("cls", "factory"), cases(Driver))


def test_run_returns_a_complete_driver_result(cls, factory) -> None:  # noqa: ANN001
    case = build(cls, factory)
    ctx = make_ctx(axes=case.axes)
    stage = OptimStage(losses=case.losses, rep=case.rep, driver=case.component, target_source=FixedTarget())
    configured = {"driver": case.component, "rep": stage.rep, "target_source": stage.target_source,
                  **{f"loss {n}": t for n, (t, _) in stage.losses.items()}}
    before = {name: snapshot(obj) for name, obj in configured.items()}

    # Bound as OptimStage.run binds it.
    state = State(target=client_target())
    ctx.session.begin_stage(ctx)
    losses = {n: (t.bind(ctx), w) for n, (t, w) in stage.losses.items()}
    bound = replace(stage, rep=stage.rep.bind(state, ctx), target_source=stage.target_source.bind(state, ctx),
                    losses=losses)
    try:
        result = case.component.run(bound, ctx)
    finally:
        for term, _ in losses.values():
            term.release()

    assert isinstance(result, DriverResult)
    assert tuple(result.input.shape) == (*ctx.axes.shape, *ctx.input_shape)
    assert not result.input.requires_grad
    assert result.iterations == case.budget
    if case.nan_loss:
        assert math.isnan(result.loss)
    else:
        assert math.isfinite(result.loss)
    for name, obj in configured.items():
        assert_unchanged(obj, before[name])


def _run_bound(case, driver, rep, axes=None, callbacks=()) -> DriverResult:  # noqa: ANN001
    ctx = make_ctx(axes=axes if axes is not None else case.axes)
    stage = OptimStage(losses=case.losses, rep=rep, driver=driver, target_source=FixedTarget())
    state = State(target=client_target())
    ctx.session.begin_stage(ctx)
    losses = {n: (t.bind(ctx), w) for n, (t, w) in stage.losses.items()}
    bound = replace(stage, rep=rep.bind(state, ctx), target_source=stage.target_source.bind(state, ctx), losses=losses)
    try:
        return driver.run(bound, ctx, callbacks=callbacks)
    finally:
        for term, _ in losses.values():
            term.release()


class _Quadratic(LossTerm):
    """``mean((x - 0.5)^2)``: with a large Adam step the iterate overshoots 0.5 and the loss oscillates."""

    def compute(self, inputs: LossInputs) -> torch.Tensor:
        return ((inputs.recon - 0.5) ** 2).mean()


def test_gradient_driver_returns_the_best_evaluated_iterate(cls, factory) -> None:  # noqa: ANN001
    """With return_best the loss is the minimum over the iterates the steps started from (not the last one);
    without it, it is the loss of the iterate after the final step."""
    if cls is not GradientDriver:
        pytest.skip("only a best-tracking gradient driver returns the best evaluated iterate")
    case = build(cls, factory)
    rep = PixelRepresentation(init=Uniform(seed=7))
    no_axis = AxisManager()
    # A large step makes the loss oscillate, so the best candidate is neither the first nor the last.
    fixed = {"lr": 0.2, "schedule": None}
    case.losses = {"quad": (_Quadratic(), 1.0)}

    def run(max_iter: int, return_best: bool) -> tuple[float, list[float]]:
        recorder = helpers.LossRecorder()
        driver = replace(case.component, max_iter=max_iter, return_best=return_best, **fixed)
        return _run_bound(case, driver, rep, axes=no_axis, callbacks=[recorder]).loss, recorder.losses

    best, candidates = run(6, True)
    assert candidates.index(min(candidates)) not in (0, len(candidates) - 1), candidates
    assert best == pytest.approx(min(candidates), rel=1e-5)
    assert best < candidates[-1] and best < candidates[0]

    final, shorter = run(5, False)
    longer = run(6, False)[1]
    assert shorter == pytest.approx(longer[:5], rel=1e-5)
    # The returned loss is the one of the iterate the next step would have started from.
    assert final == pytest.approx(longer[5], rel=1e-5)
