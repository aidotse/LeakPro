#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""GIFD (Fang et al. 2023): progressive feature-domain gradient inversion.

Stage 0 searches the generator's latent space (spherical constraint); each
subsequent stage optimizes the intermediate features after one more generator
layer group. ``GIFDIntermediateRepresentation.bind`` seeds it by applying the
previous stage's layer group to that stage's best latent or features (the paper's ``h*``).
"""
from __future__ import annotations

import torch

from leakpro.attacks.gia_attacks.modular.components.aggregators import BestLossAggregator
from leakpro.attacks.gia_attacks.modular.components.drivers import GradientDriver
from leakpro.attacks.gia_attacks.modular.components.evaluators import ImageEvaluator
from leakpro.attacks.gia_attacks.modular.components.losses import CosineGradientLoss
from leakpro.attacks.gia_attacks.modular.components.modalities.image import ImageNormalizationModality
from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.components.representations.gan import GANRepresentation
from leakpro.attacks.gia_attacks.modular.components.representations.gifd import GIFDIntermediateRepresentation
from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
from leakpro.attacks.gia_attacks.modular.core.attack import Attack
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.task import Task
from leakpro.attacks.gia_attacks.modular.core.threat_model import ThreatModel
from leakpro.fl_utils.gan_models import LayeredGANWrapper


def gifd(
    wrapped_generator: LayeredGANWrapper,
    latent_dim: int,
    mean: list[float],
    std: list[float],
    device: torch.device,
    num_classes: int,
    *,
    l1_radii: list[float] | None = None,
    stage_max_iter: int = 500,
    stage_lr: float = 0.01,
) -> Attack:
    """Progressive multi-stage GAN feature-domain inversion (GIFD, Fang et al. 2023).

    Ground-truth labels are supplied by the caller via ``run_attack(...,
    target=TargetInfo(value=labels))`` — this preset performs no label
    inference and does not accept labels directly; ``FixedTarget`` reads
    them from ``state.target`` at bind time.

    Each intermediate stage starts from the previous stage's best iterate pushed through one more
    generator layer group; the stages hand this over themselves, so no callback is needed.

    Threat model: The attacker sees the client's gradient and knows the model; no other observation field. The generator
        is the attacker's own component.

    References:
        - Fang et al., GIFD: A Generative Gradient Inversion Method with Feature Domain Optimization,
          ICCV 2023, arXiv:2308.04699.

    Deviations:
        - Gradient matching is cosine distance only; one latent start, no restarts.
        - The L1-ball radii, iteration count and learning rate are the preset's defaults; labels are supplied by the
          caller (``target=``).

    Args:
        wrapped_generator: A ``LayeredGANWrapper`` around the pre-trained
            generator — its ``num_layer_groups`` determines how many
            intermediate-feature stages this preset builds (stages
            1..num_layer_groups-1; a stage with ``start_layer ==
            num_layer_groups`` would be a no-op, since
            ``LayeredGANWrapper.forward_from`` loops
            ``range(start_layer, num_layer_groups)``, so it is never built).
        latent_dim: Generator latent dimensionality.
        mean: Per-channel dataset normalization mean of the model input (the modality maps
            the generator's decoded [0, 1] image into it).
        std: Per-channel dataset normalization std (paired with ``mean``).
        device: Device to run the attack on.
        num_classes: Number of classifier output classes.
        l1_radii: Per-intermediate-stage L1-ball radius, one entry per
            intermediate stage (``len == wrapped_generator.num_layer_groups - 1``).
            Defaults to a radius that shrinks by half each stage, starting at 1000.0.
        stage_max_iter: Adam iterations, shared by every stage (stage 0 included).
        stage_lr: Adam learning rate, shared by every stage (stage 0 included).

    """
    num_layer_groups = wrapped_generator.num_layer_groups
    num_intermediate_stages = max(num_layer_groups - 1, 0)
    if l1_radii is None:
        l1_radii = [1000.0 / (2 ** i) for i in range(num_intermediate_stages)]
    if len(l1_radii) != num_intermediate_stages:
        raise ValueError(
            f"l1_radii must have one entry per intermediate stage ({num_intermediate_stages}), got {len(l1_radii)}."
        )

    losses = {"cosine": (CosineGradientLoss(), 1.0)}

    stages = [
        OptimStage(
            losses=losses,
            rep=GANRepresentation(wrapped_generator, latent_dim, spherical=True, num_classes=num_classes),
            driver=GradientDriver(lr=stage_lr, max_iter=stage_max_iter, return_best=True),
            target_source=FixedTarget(),
            aggregator=BestLossAggregator(),
        ),
    ]
    for layer_idx in range(num_intermediate_stages):
        stages.append(
            OptimStage(
                losses=losses,
                rep=GIFDIntermediateRepresentation(wrapped_generator, start_layer=layer_idx + 1, l1_radius=l1_radii[layer_idx]),
                driver=GradientDriver(lr=stage_lr, max_iter=stage_max_iter, return_best=True),
                target_source=FixedTarget(),
                aggregator=BestLossAggregator(),
            )
        )

    return Attack(
        threat_model=ThreatModel(),
        modality=ImageNormalizationModality(mean=mean, std=std, device=device),
        evaluator=ImageEvaluator(),
        task=Task(CrossEntropy()),
        stages=stages,
        target_inference=None,
    )
