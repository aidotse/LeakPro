#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""GGL (Li et al. 2022): gradient-free latent search via CMA-ES."""
from __future__ import annotations

import torch
from torch import nn

from leakpro.attacks.gia_attacks.modular.components.drivers.cma_es import CMAESDriver
from leakpro.attacks.gia_attacks.modular.components.evaluators import ImageEvaluator
from leakpro.attacks.gia_attacks.modular.components.losses import CosineGradientLoss
from leakpro.attacks.gia_attacks.modular.components.modalities.image import ImageNormalizationModality
from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.components.representations.gan import GANRepresentation
from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
from leakpro.attacks.gia_attacks.modular.core.attack import Attack
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.task import Task
from leakpro.attacks.gia_attacks.modular.core.threat_model import ThreatModel


def ggl(
    generator: nn.Module,
    latent_dim: int,
    mean: list[float],
    std: list[float],
    device: torch.device,
    num_classes: int,
    *,
    conditional: bool = False,
    population_size: int = 32,
    max_generations: int = 200,
) -> Attack:
    """Gradient-free latent-space gradient inversion via CMA-ES (GGL, Li et al. 2022).

    Threat model: The attacker sees the client's gradient and knows the model; no other observation field. The generator
        is the attacker's own component.

    References:
        - Li et al., Auditing Privacy Defenses in Federated Learning via Generative Gradient Leakage,
          CVPR 2022, arXiv:2203.15696.

    Deviations:
        - Only the CMA-ES latent search with a cosine gradient distance is implemented; the paper's gradient-based
          refinement is not.
        - Population size and generation budget are the preset's defaults; labels are supplied by the caller
          (``target=``).

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
        population_size: CMA-ES population size — see ``CMAESDriver``.
        max_generations: CMA-ES generation budget — see ``CMAESDriver``.

    Notes:
    -----
        Ground-truth labels are supplied by the caller via ``run_attack(...,
        target=TargetInfo(value=labels))`` — this preset performs no label
        inference and does not accept labels directly; ``FixedTarget`` reads
        them from ``state.target`` at bind time.

    """
    return Attack(
        threat_model=ThreatModel(),
        modality=ImageNormalizationModality(mean=mean, std=std, device=device),
        evaluator=ImageEvaluator(),
        task=Task(CrossEntropy()),
        stages=[
            OptimStage(
                losses={"cosine": (CosineGradientLoss(), 1.0)},
                rep=GANRepresentation(generator, latent_dim, conditional=conditional, num_classes=num_classes),
                driver=CMAESDriver(population_size=population_size, max_generations=max_generations),
                target_source=FixedTarget(),
                aggregator=None,
            )
        ],
        target_inference=None,
    )
