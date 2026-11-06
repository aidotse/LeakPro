#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""GIAS preset.

Jeon et al. 2021: two-stage latent-space + generator fine-tuning gradient
inversion attack for image classifiers.
"""
from __future__ import annotations

import torch
from torch import nn

from leakpro.attacks.gia_attacks.modular.components.aggregators import BestLossAggregator
from leakpro.attacks.gia_attacks.modular.components.drivers import GradientDriver
from leakpro.attacks.gia_attacks.modular.components.evaluators import ImageEvaluator
from leakpro.attacks.gia_attacks.modular.components.losses import CosineGradientLoss
from leakpro.attacks.gia_attacks.modular.components.modalities.image import ImageNormalizationModality
from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.components.representations.gan import (
    GANRepresentation,
    UnfrozenGANRepresentation,
)
from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
from leakpro.attacks.gia_attacks.modular.core.attack import Attack
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.task import Task
from leakpro.attacks.gia_attacks.modular.core.threat_model import ThreatModel


def gias(
    generator: nn.Module,
    latent_dim: int,
    mean: list[float],
    std: list[float],
    device: torch.device,
    num_classes: int,
    *,
    conditional: bool = False,
    stage1_lr: float = 0.1,
    stage1_max_iter: int = 1000,
    stage2_lr: float = 0.01,
    stage2_max_iter: int = 500,
) -> Attack:
    """Two-stage GAN-based gradient inversion (GIAS, Jeon et al. 2021).

    Stage 1 searches the generator's latent space; stage 2 fine-tunes the
    generator's own weights around the latent stage 1 ended on, using
    ``state.artifacts["latent"]`` to hand the latent code from stage 1 to
    stage 2 (the cross-stage handoff mechanism). The handed-over
    latent is the one of stage 1's best iterate: the driver returns the image,
    target and artifacts of the same evaluation.

    Ground-truth labels are supplied by the caller via ``run_attack(...,
    target=TargetInfo(value=labels))`` — this preset performs no label
    inference and does not accept labels directly; ``FixedTarget`` reads
    them from ``state.target`` at bind time.

    Threat model: The attacker sees the client's gradient and knows the model; no other observation field. The generator
        is the attacker's own component.

    References:
        - Jeon et al., Gradient Inversion with Generative Image Prior, NeurIPS 2021,
          arXiv:2110.14962.

    Deviations:
        - Both stages match gradients with cosine distance and no further prior; one latent start, no restarts.
        - Iteration counts and learning rates are the preset's defaults; labels are supplied by the caller (``target=``).

    Args:
        generator: Pre-trained generator (e.g. via
            ``leakpro.fl_utils.gan_handler.load_pretrained_gan``).
        latent_dim: Generator latent dimensionality.
        mean: Per-channel dataset normalization mean of the model input (the modality maps
            the generator's decoded [0, 1] image into it).
        std: Per-channel dataset normalization std (paired with ``mean``).
        device: Device to run the attack on.
        num_classes: Number of classifier output classes.
        conditional: Pass labels through to the generator's forward call.
        stage1_lr: Adam learning rate for the latent-search stage.
        stage1_max_iter: Adam iterations for the latent-search stage.
        stage2_lr: Adam learning rate for the generator fine-tuning stage.
        stage2_max_iter: Adam iterations for the generator fine-tuning stage.

    """
    losses = {"cosine": (CosineGradientLoss(), 1.0)}

    return Attack(
        threat_model=ThreatModel(),
        modality=ImageNormalizationModality(mean=mean, std=std, device=device),
        evaluator=ImageEvaluator(),
        task=Task(CrossEntropy()),
        stages=[
            OptimStage(
                losses=losses,
                rep=GANRepresentation(generator, latent_dim, conditional=conditional, num_classes=num_classes),
                driver=GradientDriver(lr=stage1_lr, max_iter=stage1_max_iter, return_best=True),
                target_source=FixedTarget(),
                aggregator=BestLossAggregator(),
            ),
            OptimStage(
                losses=losses,
                rep=UnfrozenGANRepresentation(generator, latent_dim, conditional=conditional, num_classes=num_classes),
                driver=GradientDriver(lr=stage2_lr, max_iter=stage2_max_iter, return_best=True),
                target_source=FixedTarget(),
                aggregator=BestLossAggregator(),
            ),
        ],
        target_inference=None,
    )
