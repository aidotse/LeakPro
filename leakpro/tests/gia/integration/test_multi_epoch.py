#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Integration test: multi-epoch stateful task engine.

Proves that:
  1. Task(CrossEntropy(), FedAvg()) can simulate FedAvg parameter
     updates on a 5-D [E, N, C, H, W] input without crashing.
  2. An OptimStage with AxisManager(Axis("epoch", 3)) and EpochMatchingAggregator
     completes optimization and collapses the epoch dimension via Hungarian matching,
     producing a final reconstruction with shape == (N, C, H, W).
"""
from __future__ import annotations


import torch
import torch.nn as nn


from leakpro.attacks.gia_attacks.modular.core.axes import Axis, AxisManager
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, Context, State, TargetInfo
from leakpro.attacks.gia_attacks.modular.components.aggregators import EpochMatchingAggregator
from leakpro.attacks.gia_attacks.modular.components.drivers import GradientDriver
from leakpro.attacks.gia_attacks.modular.components.losses import CosineGradientLoss
from leakpro.attacks.gia_attacks.modular.components.modalities.image import ImageNormalizationModality
from leakpro.attacks.gia_attacks.modular.components.representations.pixel import PixelRepresentation
from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.core.protocol import FedAvg
from leakpro.attacks.gia_attacks.modular.core.task import Task


N_EPOCHS   = 3
BATCH      = 1
INPUT_SHAPE = (BATCH, 3, 8, 8)
DEVICE     = torch.device("cpu")
MEAN       = [0.5, 0.5, 0.5]
STD        = [0.5, 0.5, 0.5]


def _make_model() -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 10))


def test_multi_epoch_pipeline_completes_and_collapses():
    """End-to-end multi-epoch test.

    Steps:
      1. Victim simulates FedAvg updates across N_EPOCHS epochs (5-D input).
      2. Attacker reconstructs under an epoch-prefixed axis.
      3. EpochMatchingAggregator collapses [E, N, C, H, W] → [N, C, H, W].
    """
    torch.manual_seed(42)
    model = _make_model().to(DEVICE)
    task = Task(CrossEntropy(), FedAvg())
    target_info = TargetInfo(value=torch.tensor([2]))

    # ── Step 1: victim generates ΔW payload from 3 epochs of 5-D input ──────
    target_images = torch.rand(N_EPOCHS, *INPUT_SHAPE)  # [E, N, C, H, W]
    # input_shape declares the canonical per-epoch shape, so the leading
    # dimension of target_images is understood as the epoch prefix.
    obs_raw = task.victim_signal(
        model, target_images, target_info, input_shape=INPUT_SHAPE
    )
    # Detach so the attacker starts from a clean observation.
    obs = ClientObservation(gradients=[g.detach().clone() for g in obs_raw.gradients])

    # ── Step 2: attacker context with epoch axis ──────────────────────────────
    ctx = Context(
        model=model,
        observation=obs,
        modality=ImageNormalizationModality(mean=MEAN, std=STD, device=DEVICE),
        task=task,
        input_shape=INPUT_SHAPE,
        device=DEVICE,
        axes=AxisManager(Axis("epoch", N_EPOCHS)),
    )

    # ── Step 3: run stage with EpochMatchingAggregator ────────────────────────
    stage = OptimStage(
        losses={"grad_match": (CosineGradientLoss(), 1.0)},
        rep=PixelRepresentation(min_val=0.0, max_val=1.0),
        driver=GradientDriver(lr=0.01, max_iter=3, return_best=True, schedule=None),
        target_source=FixedTarget(),
        aggregator=EpochMatchingAggregator(),
    )
    out = stage.run(State(target=target_info), ctx)

    # ── Assertions ────────────────────────────────────────────────────────────
    assert out.input is not None, "state.input must not be None after the stage"
    assert out.input.shape == INPUT_SHAPE, (
        f"EpochMatchingAggregator must collapse epoch dim: "
        f"expected {INPUT_SHAPE}, got {tuple(out.input.shape)}"
    )
    print(f"[OK] multi-epoch pipeline: output shape {tuple(out.input.shape)}")
