#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Yin et al. 2021 preset — multi-seed batch recovery (See Through Gradients)."""
from __future__ import annotations

import torch

from leakpro.attacks.gia_attacks.modular.components.aggregators import MeanAggregator
from leakpro.attacks.gia_attacks.modular.components.drivers import GradientDriver
from leakpro.attacks.gia_attacks.modular.components.drivers.grad_transforms import GaussianNoise, GradTransform
from leakpro.attacks.gia_attacks.modular.components.drivers.schedules import Cosine
from leakpro.attacks.gia_attacks.modular.components.evaluators import ImageEvaluator
from leakpro.attacks.gia_attacks.modular.components.losses import (
    BNStatsLoss,
    GroupConsistencyLoss,
    L2GradientLoss,
    L2ImageNormLoss,
    TVLoss,
)
from leakpro.attacks.gia_attacks.modular.components.modalities.image import ImageNormalizationModality
from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
from leakpro.attacks.gia_attacks.modular.core.attack import Attack
from leakpro.attacks.gia_attacks.modular.core.axes import Axis, AxisManager
from leakpro.attacks.gia_attacks.modular.core.interfaces import ModelMode, Schedule, Space
from leakpro.attacks.gia_attacks.modular.core.protocol import FedSGD
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.task import Task
from leakpro.attacks.gia_attacks.modular.core.threat_model import ThreatModel
from leakpro.attacks.gia_attacks.modular.presets._options import LabelSource, labels_to_inference, pixel_representation

_DEFAULT_SCHEDULE = Cosine()  # frozen and stateless, so one shared instance is safe


def yin(
    mean: list[float],
    std: list[float],
    device: torch.device,
    *,
    num_seeds: int = 4,
    lr: float = 0.1,
    max_iter: int = 8000,
    tv_weight: float = 1e-4,
    bn_weight: float = 1e-2,
    l2_weight: float = 1e-6,
    group_weight: float = 1e-2,
    labels: LabelSource = "oracle",
    space: Space = "model_input",
    schedule: Schedule | None = _DEFAULT_SCHEDULE,
    grad_transforms: tuple[GradTransform, ...] = (),
    patience: int | None = None,
    gradient_noise_std: float = 0.0,
    registration_max_shift: int = 0,
    model_mode: ModelMode = "train",
) -> Attack:
    """Return an Attack config reproducing Yin et al. 2021 (See Through Gradients).

    Runs num_seeds independent random restarts in parallel, matching L2 gradient
    distance and BN statistics, then averages all seeds via MeanAggregator as
    prescribed in the paper.

    Optional regularizers:
      l2_weight   — spatially-normalized L2 image norm (0 disables).
      group_weight — cross-seed group consistency penalty (0 disables).
      gradient_noise_std — Gaussian noise added to the gradients before the other
        ``grad_transforms`` (0 disables).
      registration_max_shift — when positive, seeds are translation-aligned (up to this many
        pixels) before averaging; 0 averages them as they are.

    Threat model: The attacker sees the client's gradient and BN statistics and knows the model.

    References:
        - Yin, Molchanov, Alvarez, Li, Mallya, Hoiem, Jha, Kautz, See through Gradients: Image Batch Recovery via
          GradInversion, CVPR 2021, arXiv:2104.07586.

    Deviations:
        - No RANSAC-Flow registration. Seeds are averaged as they are; ``registration_max_shift > 0`` aligns them by
          integer translation first.
        - No gradient noise by default (``gradient_noise_std`` adds it); the group-consistency weight (1e-2), seed count
          (4) and 8000 steps are the LeakPro reference values.
        - Matches gradients with an L2 distance and the image-norm prior as a per-image sum; labels are supplied by the
          caller (``target=``) by default.

    """
    losses: dict = {
        "l2": (L2GradientLoss(), 1.0),
        "bn_stats": (BNStatsLoss(strategy="running"), bn_weight),
        "tv": (TVLoss(space=space), tv_weight),
    }
    if l2_weight > 0:
        losses["l2_image"] = (L2ImageNormLoss(space=space, reduction="per_image_sum"), l2_weight)
    if group_weight > 0:
        losses["group"] = (GroupConsistencyLoss(space=space), group_weight)

    if gradient_noise_std > 0:
        grad_transforms = (GaussianNoise(gradient_noise_std), *grad_transforms)
    aggregator = MeanAggregator(max_shift=registration_max_shift)

    return Attack(
        modality=ImageNormalizationModality(mean=mean, std=std, device=device),
        evaluator=ImageEvaluator(),
        task=Task(CrossEntropy(), FedSGD(), model_mode=model_mode),
        threat_model=ThreatModel(has_bn_statistics=True),
        target_inference=labels_to_inference(labels),
        axes=AxisManager(Axis("seed", num_seeds)),
        stages=[
            OptimStage(
                losses=losses,
                rep=pixel_representation(space),
                driver=GradientDriver(
                    lr=lr, max_iter=max_iter, return_best=True,
                    schedule=schedule, grad_transforms=grad_transforms, patience=patience,
                ),
                target_source=FixedTarget(),
                aggregator=aggregator,
            )
        ],
    )
