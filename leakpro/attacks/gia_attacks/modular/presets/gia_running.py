#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""GIA Running preset: BN statistics inferred from the client's pre- and post-training running statistics."""
from __future__ import annotations

import torch

from leakpro.attacks.gia_attacks.modular.components.aggregators import BestLossAggregator
from leakpro.attacks.gia_attacks.modular.components.drivers import GradientDriver
from leakpro.attacks.gia_attacks.modular.components.drivers.grad_transforms import GradTransform
from leakpro.attacks.gia_attacks.modular.components.evaluators import ImageEvaluator
from leakpro.attacks.gia_attacks.modular.components.losses import BNStatsLoss, CosineGradientLoss, TVLoss
from leakpro.attacks.gia_attacks.modular.components.modalities.image import ImageNormalizationModality
from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
from leakpro.attacks.gia_attacks.modular.core.attack import Attack
from leakpro.attacks.gia_attacks.modular.core.interfaces import ModelMode, Schedule, Space
from leakpro.attacks.gia_attacks.modular.core.protocol import FedSGD
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.task import Task
from leakpro.attacks.gia_attacks.modular.core.threat_model import ThreatModel
from leakpro.attacks.gia_attacks.modular.presets._options import (
    STEP_DECAY,
    LabelSource,
    labels_to_inference,
    pixel_representation,
)


def gia_running(
    mean: list[float],
    std: list[float],
    device: torch.device,
    *,
    lr: float = 0.1,
    max_iter: int = 3000,
    tv_weight: float = 0.052,
    bn_weight: float = 1.6e-4,
    bn_momentum: float | None = None,
    labels: LabelSource = "oracle",
    space: Space = "model_input",
    schedule: Schedule | None = STEP_DECAY,
    grad_transforms: tuple[GradTransform, ...] = (),
    patience: int | None = None,
    model_mode: ModelMode = "train",
) -> Attack:
    """Return an Attack config for GIA Running.

    Cosine gradient matching plus total variation and a BN-statistics term, optimised in
    ``space`` (default: the model's normalised input, so the weights keep that meaning). The client
    runs one ``FedSGD`` step in ``model_mode``: one train-mode step updates the BN running statistics exactly
    once, so the inferred strategy can invert that update to the client's batch statistics. Statistics
    accumulated over several FedAvg local steps mix batches seen at different weights and cannot be
    inverted; ``BNStatsLoss(strategy="inferred")`` raises for them.

    Threat model: the attacker sees the BN running statistics before and after local training and the client's batch size.
    With ``labels="oracle"`` the caller passes ``target=`` to ``run_attack``.

    References:
        - BN-statistics regularisation of Huang, Gupta, Song, Li, Arora, Evaluating Gradient Inversion Attacks and
          Defenses in Federated Learning, NeurIPS 2021, arXiv:2112.00059.

    Deviations:
        - The BN term inverts the update of the running statistics (strategy ``inferred``); this variant is LeakPro's and
          is not in the paper.
        - The client runs one ``FedSGD`` step; weights, step counts and the step schedule are the LeakPro reference
          values.
        - Labels are supplied by the caller (``target=``) by default.

    Args:
        mean: Per-channel mean of the image normalisation.
        std: Per-channel standard deviation of the image normalisation.
        device: Device of the reconstruction.
        lr: Adam learning rate.
        max_iter: Optimisation steps.
        tv_weight: Weight of the total-variation prior.
        bn_weight: Weight of the BN-statistics term.
        bn_momentum: EMA momentum used to invert the statistics update; ``None`` reads each layer's own momentum.
        labels: Label source: ``"oracle"`` or ``"idlg"``.
        space: Optimisation space of the pixels.
        schedule: Learning-rate schedule; the default scales the rate by 0.1 at ``n // 2``, ``n // 1.6``, ``n // 1.142``.
        grad_transforms: Transforms applied to the gradient before each optimizer step.
        patience: Steps without improvement before stopping; ``None`` runs to ``max_iter``.
        model_mode: Module mode of the client model.

    """
    return Attack(
        modality=ImageNormalizationModality(mean=mean, std=std, device=device),
        evaluator=ImageEvaluator(),
        task=Task(CrossEntropy(), FedSGD(), model_mode=model_mode),
        threat_model=ThreatModel(has_bn_statistics=True, has_local_hyperparameters=True),
        target_inference=labels_to_inference(labels),
        stages=[
            OptimStage(
                losses={
                    "grad_match": (CosineGradientLoss(), 1.0),
                    "tv": (TVLoss(space=space), tv_weight),
                    "bn_stats": (BNStatsLoss(strategy="inferred", momentum=bn_momentum), bn_weight),
                },
                rep=pixel_representation(space),
                driver=GradientDriver(
                    lr=lr, max_iter=max_iter, return_best=True,
                    schedule=schedule, grad_transforms=grad_transforms, patience=patience,
                ),
                target_source=FixedTarget(),
                aggregator=BestLossAggregator(),
            )
        ],
    )
