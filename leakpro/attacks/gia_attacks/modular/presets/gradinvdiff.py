#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""GradInvDiff (Wang et al., MICCAI 2025): diffusion-guided gradient inversion.

Adaptive mean optimization (AMO) with gradient-aligned noise injection (GANI).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from leakpro.attacks.gia_attacks.modular.components.diffusion.mean_strategies import AdaptiveMeanOptimization
from leakpro.attacks.gia_attacks.modular.components.diffusion.noise_strategies import GradientAlignedNoiseInjection
from leakpro.attacks.gia_attacks.modular.components.diffusion.schedules import LinearDecaySchedule
from leakpro.attacks.gia_attacks.modular.components.drivers.diffusion import DiffusionSamplingDriver
from leakpro.attacks.gia_attacks.modular.components.evaluators import ImageEvaluator
from leakpro.attacks.gia_attacks.modular.components.losses import CosineGradientLoss
from leakpro.attacks.gia_attacks.modular.components.modalities.image import ImageNormalizationModality
from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.components.representations.diffusion import DiffusionRepresentation
from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
from leakpro.attacks.gia_attacks.modular.core.attack import Attack
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.task import Task
from leakpro.attacks.gia_attacks.modular.core.threat_model import ThreatModel

if TYPE_CHECKING:
    from leakpro.fl_utils.diffusion_handler import DDPMSchedulerWrapper


def gradinvdiff(
    unet: nn.Module,
    scheduler: DDPMSchedulerWrapper,
    mean: list[float],
    std: list[float],
    device: torch.device,
    num_classes: int,  # noqa: ARG001 -- uniform preset signature
    *,
    inner_steps: int = 5,
    inner_lr: float = 0.01,
    num_steps: int | None = None,
) -> Attack:
    """Diffusion-guided gradient inversion via adaptive mean optimization (GradInvDiff, Wang et al. 2025).

    Ground-truth labels are supplied by the caller via ``run_attack(...,
    target=TargetInfo(value=labels))`` — this preset performs no label
    inference and does not accept labels directly; ``FixedTarget`` reads
    them from ``state.target`` at bind time.

    Threat model: The attacker sees the client's gradient and knows the model; no other observation field. The diffusion
        model is the attacker's own component.

    References:
        - Wang, Gan, Fang, Zhu and Liu, "GradInvDiff: Stealing Medical Privacy in Federated Learning via
          Diffusion-Based Gradient Inversion", MICCAI 2025, doi:10.1007/978-3-032-05185-1_26.

    Deviations:
        - Gradient matching uses cosine distance; the sampler is DDPM with optional step subsampling.
        - Inner-loop size and learning rate are the preset's defaults; labels are supplied by the caller (``target=``).

    Args:
        unet: Frozen denoising UNet, same as ``ggdm()``.
        scheduler: The paired ``DDPMSchedulerWrapper``, same as ``ggdm()``.
        mean: Per-channel dataset normalization mean.
        std: Per-channel dataset normalization std.
        device: Device to run the attack on.
        num_classes: Number of classifier output classes (unused — kept for
            preset-signature consistency with the GAN-based presets).
        inner_steps: Inner Adam loop size for ``AdaptiveMeanOptimization``'s mu* search.
        inner_lr: Inner Adam loop learning rate for ``AdaptiveMeanOptimization``'s mu* search
            (the paper's value by default). The matching loss is evaluated at the data-space
            point ``x = (x_t + 1) / 2``, so its gradient with respect to ``x_t`` is half the
            data-space gradient.
        num_steps: Subsample the reverse-diffusion loop to this many steps.

    """
    return Attack(
        threat_model=ThreatModel(),
        modality=ImageNormalizationModality(mean=mean, std=std, device=device),
        evaluator=ImageEvaluator(),
        task=Task(CrossEntropy()),
        stages=[
            OptimStage(
                losses={"cosine": (CosineGradientLoss(), 1.0)},
                rep=DiffusionRepresentation(),
                driver=DiffusionSamplingDriver(
                    unet=unet, scheduler=scheduler,
                    mean_strategy=AdaptiveMeanOptimization(
                        schedule=LinearDecaySchedule(), inner_steps=inner_steps, inner_lr=inner_lr,
                    ),
                    noise_strategy=GradientAlignedNoiseInjection(),
                    sampler="ddpm", num_steps=num_steps,
                ),
                target_source=FixedTarget(),
                aggregator=None,
            )
        ],
        target_inference=None,
    )
