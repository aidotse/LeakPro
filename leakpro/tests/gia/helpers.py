#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Tiny, download-free building blocks shared by the GIA tests."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from leakpro.attacks.gia_attacks.modular.components.modalities.image import ImageNormalizationModality
from leakpro.attacks.gia_attacks.modular.core.attack import Attack
from leakpro.attacks.gia_attacks.modular.core.events import StepEvent
from leakpro.attacks.gia_attacks.modular.core.interfaces import Callback
from leakpro.attacks.gia_attacks.modular.core.loss_eval import evaluate_stage_losses
from leakpro.attacks.gia_attacks.modular.core.simulate import simulate_client
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, Context, TargetInfo

NUM_CLASSES = 10
IMAGE_SHAPE = (3, 8, 8)
# Non-uniform per-channel statistics on purpose: they make normalisation bugs visible.
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2470, 0.2435, 0.2616]
DEVICE = torch.device("cpu")


def tiny_bn_cnn(seed: int = 0) -> nn.Sequential:
    """Conv + BN + linear head: the smallest model exercising BN statistics and iDLG label leakage."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(2),
        nn.Flatten(),
        nn.Linear(8 * 2 * 2, NUM_CLASSES),
    ).to(DEVICE)


def client_batch(batch_size: int = 2, seed: int = 1) -> tuple[Tensor, Tensor]:
    """Data-space images in [0,1] and distinct labels (distinct so iDLG is well posed)."""
    gen = torch.Generator().manual_seed(seed)
    images = torch.rand(batch_size, *IMAGE_SHAPE, generator=gen)
    labels = torch.randperm(NUM_CLASSES, generator=gen)[:batch_size]
    return images.to(DEVICE), labels.to(DEVICE)


def image_modality() -> ImageNormalizationModality:
    """The CIFAR-normalising image modality used throughout the suite."""
    return ImageNormalizationModality(mean=CIFAR_MEAN, std=CIFAR_STD, device=DEVICE)


def observe(attack: Attack, model: nn.Module, images: Tensor, labels: Tensor) -> ClientObservation:
    """What the server intercepts when the victim trains on (images, labels); the caller's model is untouched."""
    return simulate_client(model, images, TargetInfo(value=labels), task=attack.task, modality=attack.modality)


def stage_terms(stage: OptimStage, ctx: Context, x: Tensor, target: TargetInfo) -> dict[str, Tensor]:
    """Each weighted stage loss term at data-space point ``x``, evaluated as GradientDriver's closure does.

    Single-slice only (no seed axis): enough for the presets the tests evaluate directly.
    """
    ctx.session.begin_stage(ctx)
    losses = {n: (t.bind(ctx), w) for n, (t, w) in stage.losses.items()}
    try:
        return evaluate_stage_losses(losses, ctx, x, target, seed_index=0, full_recon=x).terms
    finally:
        for term, _ in losses.values():
            term.release()


def stage_objective(stage: OptimStage, ctx: Context, x: Tensor, target: TargetInfo) -> Tensor:
    """The weighted stage objective at data-space point ``x`` (the sum of ``stage_terms``)."""
    return sum(stage_terms(stage, ctx, x, target).values())


class BoundRecorder:
    """Stands in for a stage's representation and keeps the bound copy a stage run creates.

    ``bind`` returns a fresh bound copy and leaves the configured object unchanged, so a test
    that inspects the run's parameters after ``stage.run`` reads them from ``recorder.bound``.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.bound: Any = None

    def bind(self, state: Any, ctx: Context) -> Any:
        self.bound = self.inner.bind(state, ctx)
        return self.bound


class LossRecorder(Callback):
    """Collects the per-step total loss reported by the driver."""

    def __init__(self) -> None:
        self.losses: list[float] = []

    def on_step_end(self, event: StepEvent) -> None:
        self.losses.append(float(event.loss))
