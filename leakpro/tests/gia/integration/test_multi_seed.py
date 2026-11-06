#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Parity test: multi-seed OptimStage collapses to canonical [N, C, H, W].

An OptimStage with:
  - GradientDriver (gradient-based optimization)
  - BestLossAggregator (seed selector)
  - AxisManager(Axis("seed", 4)) (4 parallel seeds)

must complete without crashing and produce state.input with shape == INPUT_SHAPE,
i.e. the seed dimension must be fully collapsed by the aggregator.
"""
from __future__ import annotations


import torch
import torch.nn as nn


from leakpro.attacks.gia_attacks.modular.core.axes import Axis, AxisManager
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, Context, State, TargetInfo
from leakpro.attacks.gia_attacks.modular.components.aggregators import BestLossAggregator
from leakpro.attacks.gia_attacks.modular.components.drivers import GradientDriver
from leakpro.attacks.gia_attacks.modular.components.losses import CosineGradientLoss, TVLoss
from leakpro.attacks.gia_attacks.modular.components.modalities.image import ImageNormalizationModality
from leakpro.attacks.gia_attacks.modular.components.representations.pixel import PixelRepresentation
from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.core.protocol import FedSGD
from leakpro.attacks.gia_attacks.modular.core.task import Task


INPUT_SHAPE = (1, 3, 8, 8)
N_SEEDS     = 4
DEVICE      = torch.device("cpu")
MEAN        = [0.5, 0.5, 0.5]
STD         = [0.5, 0.5, 0.5]


def _make_model() -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 10))


def _make_ctx(axes: AxisManager) -> Context:
    model = _make_model()
    task = Task(CrossEntropy(), FedSGD())
    imgs = torch.rand(INPUT_SHAPE)
    obs_raw = task.victim_signal(model, imgs, TargetInfo(value=torch.tensor([0])))
    obs = ClientObservation(gradients=[g.detach().clone() for g in obs_raw.gradients])
    return Context(
        model=model,
        observation=obs,
        modality=ImageNormalizationModality(mean=MEAN, std=STD, device=DEVICE),
        task=task,
        input_shape=INPUT_SHAPE,
        device=DEVICE,
        axes=axes,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_multi_seed_stage_completes_without_crash():
    """GradientDriver + BestLossAggregator with 4 seeds must not raise."""
    axes = AxisManager(Axis("seed", N_SEEDS))
    ctx = _make_ctx(axes)
    state = State(target=TargetInfo(value=torch.tensor([0])))

    stage = OptimStage(
        losses={
            "grad_match": (CosineGradientLoss(), 1.0),
            "tv": (TVLoss(), 1e-4),
        },
        rep=PixelRepresentation(min_val=0.0, max_val=1.0),
        driver=GradientDriver(lr=1e-2, max_iter=3, return_best=True, schedule=None),
        target_source=FixedTarget(),
        aggregator=BestLossAggregator(),
    )
    out = stage.run(state, ctx)
    assert out.input is not None
    print("[OK] multi-seed stage completed without crash")


def test_multi_seed_output_shape_is_input_shape():
    """After aggregation, state.input must be exactly INPUT_SHAPE (seed dim collapsed)."""
    axes = AxisManager(Axis("seed", N_SEEDS))
    ctx = _make_ctx(axes)
    state = State(target=TargetInfo(value=torch.tensor([0])))

    stage = OptimStage(
        losses={
            "grad_match": (CosineGradientLoss(), 1.0),
            "tv": (TVLoss(), 1e-4),
        },
        rep=PixelRepresentation(min_val=0.0, max_val=1.0),
        driver=GradientDriver(lr=1e-2, max_iter=3, return_best=True, schedule=None),
        target_source=FixedTarget(),
        aggregator=BestLossAggregator(),
    )
    out = stage.run(state, ctx)

    assert out.input.shape == INPUT_SHAPE, (
        f"Expected {INPUT_SHAPE} after seed aggregation, got {tuple(out.input.shape)}"
    )
    print(f"[OK] multi-seed output shape: {tuple(out.input.shape)}")


def test_multi_seed_output_is_in_valid_range():
    """PixelRepresentation clamps output to [0, 1]; aggregated result must respect this."""
    axes = AxisManager(Axis("seed", N_SEEDS))
    ctx = _make_ctx(axes)
    state = State(target=TargetInfo(value=torch.tensor([0])))

    stage = OptimStage(
        losses={"grad_match": (CosineGradientLoss(), 1.0)},
        rep=PixelRepresentation(min_val=0.0, max_val=1.0),
        driver=GradientDriver(lr=1e-2, max_iter=3, return_best=True, schedule=None),
        target_source=FixedTarget(),
        aggregator=BestLossAggregator(),
    )
    out = stage.run(state, ctx)

    assert out.input.min().item() >= 0.0 - 1e-5
    assert out.input.max().item() <= 1.0 + 1e-5
    print("[OK] multi-seed output within [0, 1]")


def test_no_seed_axis_shape_unchanged():
    """Without a seed axis, BestLossAggregator is a no-op and shape is INPUT_SHAPE."""
    axes = AxisManager()  # no axes
    ctx = _make_ctx(axes)
    state = State(target=TargetInfo(value=torch.tensor([0])))

    stage = OptimStage(
        losses={"grad_match": (CosineGradientLoss(), 1.0)},
        rep=PixelRepresentation(min_val=0.0, max_val=1.0),
        driver=GradientDriver(lr=1e-2, max_iter=3, return_best=True, schedule=None),
        target_source=FixedTarget(),
        aggregator=BestLossAggregator(),
    )
    out = stage.run(state, ctx)

    assert out.input.shape == INPUT_SHAPE
    print("[OK] no seed axis: shape unchanged")
