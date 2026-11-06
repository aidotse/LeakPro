#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""DLG (Zhu et al. 2019) preset."""
from __future__ import annotations

import torch

from leakpro.attacks.gia_attacks.modular.components.aggregators import BestLossAggregator
from leakpro.attacks.gia_attacks.modular.components.drivers import GradientDriver
from leakpro.attacks.gia_attacks.modular.components.evaluators import ImageEvaluator
from leakpro.attacks.gia_attacks.modular.components.losses import L2GradientLoss
from leakpro.attacks.gia_attacks.modular.components.modalities.image import ImageNormalizationModality
from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.components.representations.pixel import PixelRepresentation
from leakpro.attacks.gia_attacks.modular.components.targets import JointTarget
from leakpro.attacks.gia_attacks.modular.core.attack import Attack
from leakpro.attacks.gia_attacks.modular.core.interfaces import ModelMode
from leakpro.attacks.gia_attacks.modular.core.protocol import FedSGD
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.task import Task
from leakpro.attacks.gia_attacks.modular.core.threat_model import ThreatModel


def dlg(
    mean: list[float],
    std: list[float],
    device: torch.device,
    num_classes: int,
    *,
    lr: float = 1.0,
    max_iter: int = 300,
    model_mode: ModelMode = "train",
) -> Attack:
    """Return an Attack config reproducing the DLG attack (Zhu et al. 2019).

    Uses L2 gradient matching and jointly optimises labels alongside the image
    via L-BFGS with no TV regularization.

    Threat model: The attacker sees the client's gradient and knows the model; no other observation field.

    References:
        - Zhu, Liu, Han, Deep Leakage from Gradients, NeurIPS 2019.

    Deviations:
        - The image is a clipped [0, 1] tensor (U(0, 1) start) that enters the model after mean/std normalisation, not
          unconstrained Gaussian dummy data.
        - Labels are optimised jointly as soft labels; the L-BFGS settings (lr 1.0, 300 steps) are the
          preset's defaults.

    """
    return Attack(
        threat_model=ThreatModel(),
        modality=ImageNormalizationModality(mean=mean, std=std, device=device),
        evaluator=ImageEvaluator(),
        task=Task(CrossEntropy(), FedSGD(), model_mode=model_mode),
        stages=[
            OptimStage(
                losses={"l2": (L2GradientLoss(), 1.0)},
                rep=PixelRepresentation(min_val=0.0, max_val=1.0),
                driver=GradientDriver(
                    lr=lr,
                    max_iter=max_iter,
                    return_best=True,
                    optimizer_cls=torch.optim.LBFGS,
                    schedule=None,
                ),
                target_source=JointTarget(num_classes=num_classes),
                aggregator=BestLossAggregator(),
            )
        ],
        target_inference=None,
    )
