#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Combined multi-epoch + multi-seed GIA preset."""
from __future__ import annotations

import torch

from leakpro.attacks.gia_attacks.modular.components.aggregators import (
    BestLossAggregator,
    ChainAggregator,
    EpochMatchingAggregator,
)
from leakpro.attacks.gia_attacks.modular.components.drivers import GradientDriver
from leakpro.attacks.gia_attacks.modular.components.evaluators import ImageEvaluator
from leakpro.attacks.gia_attacks.modular.components.losses import CosineGradientLoss, TVLoss
from leakpro.attacks.gia_attacks.modular.components.modalities.image import ImageNormalizationModality
from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.components.representations.pixel import PixelRepresentation
from leakpro.attacks.gia_attacks.modular.components.target_inference import IDLGInference
from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
from leakpro.attacks.gia_attacks.modular.core.attack import Attack
from leakpro.attacks.gia_attacks.modular.core.axes import Axis, AxisManager
from leakpro.attacks.gia_attacks.modular.core.interfaces import ModelMode
from leakpro.attacks.gia_attacks.modular.core.protocol import FedAvg
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.task import Task
from leakpro.attacks.gia_attacks.modular.core.threat_model import ThreatModel


def combined(
    mean: list[float],
    std: list[float],
    device: torch.device,
    *,
    num_epochs: int = 3,
    num_seeds: int = 4,
    lr: float = 0.1,
    max_iter: int = 4800,
    tv_weight: float = 1e-4,
    model_mode: ModelMode = "train",
) -> Attack:
    """Return an Attack that combines multi-epoch simulation with multi-seed restarts.

    The reconstruction tensor has shape [epoch, seed, N, C, H, W].  After the
    driver loop, ChainAggregator first selects the best seed (BestLossAggregator),
    then aligns and averages across epochs (EpochMatchingAggregator), producing a
    single [N, C, H, W] output.

    Threat model: The attacker sees the client's weight update and knows the model and the local training configuration.

    References:
        - Yin et al. 2021 (multi-seed restarts, see ``yin``) combined with Dimitrov et al. 2022 (multi-epoch updates, see
          ``dimitrov``).

    Deviations:
        - The combination is LeakPro's: no paper evaluates it. Seeds are reduced by the best loss, then epochs are
          matched.
        - Hyperparameters are LeakPro defaults, not paper values.

    """
    return Attack(
        threat_model=ThreatModel(has_local_hyperparameters=True),
        modality=ImageNormalizationModality(mean=mean, std=std, device=device),
        evaluator=ImageEvaluator(),
        task=Task(CrossEntropy(), FedAvg(local_epochs=num_epochs), model_mode=model_mode),
        target_inference=IDLGInference(),
        axes=AxisManager(Axis("epoch", num_epochs), Axis("seed", num_seeds)),
        stages=[
            OptimStage(
                losses={
                    "grad_match": (CosineGradientLoss(), 1.0),
                    "tv": (TVLoss(), tv_weight),
                },
                rep=PixelRepresentation(min_val=0.0, max_val=1.0),
                driver=GradientDriver(lr=lr, max_iter=max_iter, return_best=True),
                target_source=FixedTarget(),
                aggregator=ChainAggregator([BestLossAggregator(), EpochMatchingAggregator()]),
            )
        ],
    )
