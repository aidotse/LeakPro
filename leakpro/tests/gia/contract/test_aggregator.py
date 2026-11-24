#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Aggregator contract: collapses its axis into a DriverResult and never modifies its input."""
from __future__ import annotations

import pytest
import torch

from leakpro.attacks.gia_attacks.modular.components.aggregators import BestLossAggregator, MeanAggregator
from leakpro.attacks.gia_attacks.modular.components.drivers.gradient import GradientDriver
from leakpro.attacks.gia_attacks.modular.components.representations.pixel import PixelRepresentation
from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
from leakpro.attacks.gia_attacks.modular.core.interfaces import Aggregator
from leakpro.attacks.gia_attacks.modular.core.result import DriverResult
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation
from leakpro.tests.gia.contract.conftest import client_target, make_ctx
from leakpro.tests.gia.contract.registry import build, cases

pytestmark = pytest.mark.parametrize(("cls", "factory"), cases(Aggregator))


def test_aggregate_collapses_the_axis_without_modifying_its_input(cls, factory) -> None:  # noqa: ANN001
    case = build(cls, factory)
    ctx = make_ctx(axes=case.axes, observation=ClientObservation())
    gen = torch.Generator().manual_seed(6)
    size = case.axes.shape[0]
    recon = torch.rand(*ctx.axes.shape, *ctx.input_shape, generator=gen)
    result = DriverResult(input=recon, target=client_target(), artifacts={"pixels": recon.clone(), "start_layer": 1},
                          loss=0.5, iterations=3, slice_losses=torch.linspace(1.0, 0.0, size))
    stage = OptimStage(losses={}, rep=PixelRepresentation(), driver=GradientDriver(), target_source=FixedTarget())
    before = (recon.clone(), result.artifacts["pixels"].clone(), result.slice_losses.clone())

    out = case.component.aggregate(result, ctx, stage)

    assert isinstance(out, DriverResult)
    assert tuple(out.input.shape) == tuple(ctx.input_shape)
    assert out.slice_losses is None
    assert out.iterations == result.iterations
    for name, value in out.artifacts.items():
        if isinstance(value, torch.Tensor):
            assert tuple(value.shape) == tuple(ctx.input_shape), name
    torch.testing.assert_close(result.input, before[0], rtol=0, atol=0)
    torch.testing.assert_close(result.artifacts["pixels"], before[1], rtol=0, atol=0)
    torch.testing.assert_close(result.slice_losses, before[2], rtol=0, atol=0)


def test_collapsed_axes_names_axes_of_the_case_and_the_result_loses_them(cls, factory) -> None:  # noqa: ANN001
    case = build(cls, factory)
    names = tuple(a.name for a in case.axes.axes)
    collapsed = case.component.collapsed_axes()
    assert isinstance(collapsed, tuple)
    assert collapsed
    assert set(collapsed) <= set(names)
    # What the aggregator declares is what it removes: the result carries no axis prefix of this case.
    ctx = make_ctx(axes=case.axes, observation=ClientObservation())
    recon = torch.rand(*ctx.axes.shape, *ctx.input_shape)
    size = case.axes.shape[0]
    result = DriverResult(input=recon, target=client_target(), artifacts={"pixels": recon.clone()},
                          loss=0.5, iterations=1, slice_losses=torch.linspace(1.0, 0.0, size))
    stage = OptimStage(losses={}, rep=PixelRepresentation(), driver=GradientDriver(), target_source=FixedTarget())
    out = case.component.aggregate(result, ctx, stage)
    assert out.input.dim() == recon.dim() - len(collapsed)


def test_best_loss_returns_the_argmin_slice_and_mean_returns_the_axis_mean(cls, factory) -> None:  # noqa: ANN001
    if cls not in (MeanAggregator, BestLossAggregator):
        pytest.skip("slice selection / averaging is specific to Mean and BestLoss")
    case = build(cls, factory)
    ctx = make_ctx(axes=case.axes, observation=ClientObservation())
    recon = torch.rand(*ctx.axes.shape, *ctx.input_shape, generator=torch.Generator().manual_seed(8))
    losses = torch.tensor([0.9, 0.1])
    result = DriverResult(input=recon, target=client_target(), artifacts={"pixels": recon.clone()},
                          loss=0.5, iterations=1, slice_losses=losses)
    stage = OptimStage(losses={}, rep=PixelRepresentation(), driver=GradientDriver(), target_source=FixedTarget())
    out = case.component.aggregate(result, ctx, stage)
    expected = recon[1] if cls is BestLossAggregator else recon.mean(dim=0)
    torch.testing.assert_close(out.input, expected, rtol=1e-6, atol=1e-6)
