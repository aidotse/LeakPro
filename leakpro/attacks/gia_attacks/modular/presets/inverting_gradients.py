#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Inverting Gradients (Geiping et al. 2020) preset."""
from __future__ import annotations

import torch

from leakpro.attacks.gia_attacks.modular.components.aggregators import BestLossAggregator
from leakpro.attacks.gia_attacks.modular.components.drivers import GradientDriver
from leakpro.attacks.gia_attacks.modular.components.drivers.grad_transforms import GradTransform, SignGradient
from leakpro.attacks.gia_attacks.modular.components.drivers.schedules import MultiStep
from leakpro.attacks.gia_attacks.modular.components.evaluators import ImageEvaluator
from leakpro.attacks.gia_attacks.modular.components.losses import CosineGradientLoss, TVLoss
from leakpro.attacks.gia_attacks.modular.components.modalities.image import ImageNormalizationModality
from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
from leakpro.attacks.gia_attacks.modular.core.attack import Attack
from leakpro.attacks.gia_attacks.modular.core.interfaces import ModelMode, Schedule, Space
from leakpro.attacks.gia_attacks.modular.core.protocol import FedSGD
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.task import Task
from leakpro.attacks.gia_attacks.modular.core.threat_model import ThreatModel
from leakpro.attacks.gia_attacks.modular.presets._options import LabelSource, labels_to_inference, pixel_representation

_DEFAULT_SCHEDULE = MultiStep()  # frozen and stateless, so one shared instance is safe


def inverting_gradients(
    mean: list[float],
    std: list[float],
    device: torch.device,
    *,
    lr: float = 0.1,
    max_iter: int = 4000,
    tv_weight: float = 1e-3,
    labels: LabelSource = "oracle",
    space: Space = "model_input",
    schedule: Schedule | None = _DEFAULT_SCHEDULE,
    grad_transforms: tuple[GradTransform, ...] = (SignGradient(),),
    patience: int | None = None,
    model_mode: ModelMode = "eval",
) -> Attack:
    """Return an Attack config reproducing the Geiping 2020 cosine-matching attack.

    Threat model: The attacker sees the client's gradient and knows the model; no other observation field.

    References:
        - Geiping, Bauermeister, Droge, Moeller, Inverting Gradients - How easy is it to break privacy in federated
          learning?, NeurIPS 2020, arXiv:2003.14053. Signed gradients with Adam, Section 4.

    Deviations:
        - Runs one restart; the paper keeps the best of several.
        - Defaults (4000 steps, lr 0.1, TV weight 1e-3, step decay at 3/8, 5/8, 7/8 of the run) are the reference
          configuration of the LeakPro benchmark; the paper's TV weight depends on the dataset.
        - Labels are supplied by the caller (``target=``) by default; ``labels="idlg"`` infers them.

    """
    return Attack(
        threat_model=ThreatModel(),
        modality=ImageNormalizationModality(mean=mean, std=std, device=device),
        evaluator=ImageEvaluator(),
        task=Task(CrossEntropy(), FedSGD(), model_mode=model_mode),
        stages=[
            OptimStage(
                losses={
                    "grad_match": (CosineGradientLoss(), 1.0),
                    "tv": (TVLoss(space=space), tv_weight),
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
        target_inference=labels_to_inference(labels),
    )
