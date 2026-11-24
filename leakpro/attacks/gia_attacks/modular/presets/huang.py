#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Huang et al. 2021 preset — BN-statistics-regularized cosine gradient matching."""
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


def huang(
    mean: list[float],
    std: list[float],
    device: torch.device,
    *,
    lr: float = 0.1,
    max_iter: int = 10000,
    tv_weight: float = 0.052,
    bn_weight: float = 1.6e-4,
    labels: LabelSource = "oracle",
    space: Space = "model_input",
    schedule: Schedule | None = STEP_DECAY,
    grad_transforms: tuple[GradTransform, ...] = (),
    patience: int | None = None,
    model_mode: ModelMode = "train",
) -> Attack:
    """Return an Attack config reproducing Huang et al. 2021.

    Extends Geiping 2020 with BN statistics regularization, requiring the threat
    model to grant access to the victim's running BN mean/var.

    Threat model: The attacker sees the client's gradient and BN statistics and knows the model.

    References:
        - Huang, Gupta, Song, Li, Arora, Evaluating Gradient Inversion Attacks and Defenses in Federated Learning, NeurIPS
          2021, arXiv:2112.00059.

    Deviations:
        - Matches the client's running BN statistics (strategy ``running``), not the statistics of the private batch.
        - Runs one restart with cosine matching, TV and the BN term; weights (TV 0.052, BN 1.6e-4), 10000 steps and the
          step schedule are the LeakPro reference values.
        - Labels are supplied by the caller (``target=``) by default; ``labels="idlg"`` infers them.

    """
    return Attack(
        modality=ImageNormalizationModality(mean=mean, std=std, device=device),
        evaluator=ImageEvaluator(),
        task=Task(CrossEntropy(), FedSGD(), model_mode=model_mode),
        threat_model=ThreatModel(has_bn_statistics=True),
        target_inference=labels_to_inference(labels),
        stages=[
            OptimStage(
                losses={
                    "grad_match": (CosineGradientLoss(), 1.0),
                    "tv": (TVLoss(space=space), tv_weight),
                    "bn_stats": (BNStatsLoss(strategy="running"), bn_weight),
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
