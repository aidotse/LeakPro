#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""iDLG (Zhao et al. 2020) preset."""
from __future__ import annotations

import torch

from leakpro.attacks.gia_attacks.modular.components.aggregators import BestLossAggregator
from leakpro.attacks.gia_attacks.modular.components.drivers import GradientDriver
from leakpro.attacks.gia_attacks.modular.components.drivers.grad_transforms import GradTransform
from leakpro.attacks.gia_attacks.modular.components.evaluators import ImageEvaluator
from leakpro.attacks.gia_attacks.modular.components.losses import L2GradientLoss
from leakpro.attacks.gia_attacks.modular.components.modalities.image import ImageNormalizationModality
from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.components.target_inference import IDLGInference
from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
from leakpro.attacks.gia_attacks.modular.core.attack import Attack
from leakpro.attacks.gia_attacks.modular.core.interfaces import ModelMode, Schedule, Space
from leakpro.attacks.gia_attacks.modular.core.protocol import FedSGD
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.task import Task
from leakpro.attacks.gia_attacks.modular.core.threat_model import ThreatModel
from leakpro.attacks.gia_attacks.modular.presets._options import pixel_representation


def idlg(
    mean: list[float],
    std: list[float],
    device: torch.device,
    *,
    lr: float = 1.0,
    max_iter: int = 300,
    space: Space = "model_input",
    schedule: Schedule | None = None,
    grad_transforms: tuple[GradTransform, ...] = (),
    patience: int | None = None,
    model_mode: ModelMode = "eval",
) -> Attack:
    """Return an Attack config reproducing the iDLG attack (Zhao et al. 2020).

    Analytically infers labels via IDLGInference before optimisation, then uses
    L2 gradient matching with L-BFGS and no TV regularization.

    Threat model: The attacker sees the client's gradient and knows the model; no other observation field.

    References:
        - Zhao, Mopuri, Bilen, iDLG: Improved Deep Leakage from Gradients, arXiv:2001.02610, 2020.

    Deviations:
        - The image is optimised in the model's normalised input space from N(0, 1) without a box constraint, as in the
          released code.
        - The analytical label rule recovers the label multiset for batches larger than one, not the per-sample
          assignment; the paper derives it for one sample.
        - The client model runs in eval mode; L-BFGS (lr 1.0, 300 steps) is the preset's setting.

    """
    return Attack(
        threat_model=ThreatModel(),
        modality=ImageNormalizationModality(mean=mean, std=std, device=device),
        evaluator=ImageEvaluator(),
        task=Task(CrossEntropy(), FedSGD(), model_mode=model_mode),
        stages=[
            OptimStage(
                losses={"l2": (L2GradientLoss(), 1.0)},
                rep=pixel_representation(space, clip=False),
                driver=GradientDriver(
                    lr=lr,
                    max_iter=max_iter,
                    return_best=True,
                    optimizer_cls=torch.optim.LBFGS,
                    schedule=schedule,
                    grad_transforms=grad_transforms,
                    patience=patience,
                ),
                target_source=FixedTarget(),
                aggregator=BestLossAggregator(),
            )
        ],
        target_inference=IDLGInference(),
    )
