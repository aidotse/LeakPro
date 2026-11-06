#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""GGDM (Gu et al., WWW 2024): gradient inversion by guiding a pretrained DDPM's reverse process."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from leakpro.attacks.gia_attacks.modular.components.diffusion.mean_strategies import SimilarityGuidance
from leakpro.attacks.gia_attacks.modular.components.diffusion.noise_strategies import StandardNoiseInjection
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


def ggdm(
    unet: nn.Module,
    scheduler: DDPMSchedulerWrapper,
    mean: list[float],
    std: list[float],
    device: torch.device,
    num_classes: int,  # noqa: ARG001 -- uniform preset signature
    *,
    gamma: float = 1.0,
    num_steps: int | None = None,
) -> Attack:
    """Diffusion-guided gradient inversion via similarity guidance (GGDM, Gu et al. 2024).

    Ground-truth labels are supplied by the caller via ``run_attack(...,
    target=TargetInfo(value=labels))`` — this preset performs no label
    inference and does not accept labels directly; ``FixedTarget`` reads
    them from ``state.target`` at bind time.

    Threat model: The attacker sees the client's gradient and knows the model; no other observation field. The diffusion
        model is the attacker's own component.

    References:
        - Gu, Zhang, Li, Wei, Li and Huang, "Federated Learning Vulnerabilities: Privacy Attacks with Denoising
          Diffusion Probabilistic Models", Proceedings of the ACM Web Conference (WWW) 2024, pp. 1149-1157,
          doi:10.1145/3589334.3645514.

    Deviations:
        - Gradient matching uses cosine distance; the sampler is DDPM with optional step subsampling.
        - Labels are supplied by the caller (``target=``).

    Args:
        unet: Frozen denoising UNet (e.g. from
            ``leakpro.fl_utils.diffusion_handler.load_pretrained_ddpm``).
        scheduler: The paired ``DDPMSchedulerWrapper``.
        mean: Per-channel dataset normalization mean.
        std: Per-channel dataset normalization std.
        device: Device to run the attack on.
        num_classes: Number of classifier output classes (unused — kept for
            preset-signature consistency with the GAN-based presets).
        gamma: SimilarityGuidance's guidance strength; the paper's 1.0 by default. The
            guidance gradient is taken through the data-space map ``x = (x_t + 1) / 2``, so its
            magnitude with respect to ``x_t`` is half the data-space gradient.
        num_steps: Subsample the reverse-diffusion loop to this many steps
            (``None`` runs every timestep).

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
                    mean_strategy=SimilarityGuidance(gamma=gamma),
                    noise_strategy=StandardNoiseInjection(),
                    sampler="ddpm", num_steps=num_steps,
                ),
                target_source=FixedTarget(),
                aggregator=None,
            )
        ],
        target_inference=None,
    )
