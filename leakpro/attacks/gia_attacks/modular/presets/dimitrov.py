#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Dimitrov et al. 2022 preset — multi-epoch gradient inversion."""
from __future__ import annotations

import torch

from leakpro.attacks.gia_attacks.modular.components.aggregators import EpochMatchingAggregator
from leakpro.attacks.gia_attacks.modular.components.drivers import GradientDriver
from leakpro.attacks.gia_attacks.modular.components.drivers.grad_transforms import GradTransform
from leakpro.attacks.gia_attacks.modular.components.drivers.schedules import Cosine
from leakpro.attacks.gia_attacks.modular.components.evaluators import ImageEvaluator
from leakpro.attacks.gia_attacks.modular.components.losses import CosineGradientLoss, EpochOrderInvariantPrior, TVLoss
from leakpro.attacks.gia_attacks.modular.components.modalities.image import ImageNormalizationModality
from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
from leakpro.attacks.gia_attacks.modular.core.attack import Attack
from leakpro.attacks.gia_attacks.modular.core.axes import Axis, AxisManager
from leakpro.attacks.gia_attacks.modular.core.interfaces import ModelMode, Schedule, Space
from leakpro.attacks.gia_attacks.modular.core.protocol import FedAvg
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.task import Task
from leakpro.attacks.gia_attacks.modular.core.threat_model import ThreatModel
from leakpro.attacks.gia_attacks.modular.presets._options import LabelSource, labels_to_inference, pixel_representation

_DEFAULT_SCHEDULE = Cosine()  # frozen and stateless, so one shared instance is safe


def dimitrov(
    mean: list[float],
    std: list[float],
    device: torch.device,
    *,
    num_epochs: int = 3,
    lr: float = 1.0,
    max_iter: int = 8000,
    tv_weight: float = 1e-4,
    lambda_inv: float = 1e-6,
    client_lr: float = 0.01,
    client_batch_size: int | None = 1,
    labels: LabelSource = "oracle",
    space: Space = "model_input",
    schedule: Schedule | None = _DEFAULT_SCHEDULE,
    grad_transforms: tuple[GradTransform, ...] = (),
    patience: int | None = None,
    model_mode: ModelMode = "eval",
) -> Attack:
    """Return an Attack config reproducing Dimitrov et al. 2022.

    Inverts multi-epoch update signals (ΔW) by unrolling the epoch axis and
    using EpochMatchingAggregator to align per-epoch reconstructions back to
    a canonical ordering via Hungarian matching.

    Optional regularizers:
      lambda_inv — epoch-order invariant prior weight (0 disables).

    Threat model: The attacker sees the client's weight update and knows the model and the local training configuration
        (epochs, batch size, learning rate).

    References:
        - Dimitrov, Balunovic, Konstantinov, Vechev, Data Leakage in Federated Averaging, TMLR 2022, arXiv:2206.12395.

    Deviations:
        - Epoch alignment uses Hungarian matching of the per-epoch reconstructions and an epoch-order invariant prior, not
          the paper's search over input orderings.
        - Matches updates with cosine distance and TV; lr 1.0, 8000 steps and a client batch size of 1 are the LeakPro
          reference values, and the client model runs in eval mode.
        - Labels are supplied by the caller (``target=``) by default; ``labels="dimitrov"`` uses the paper's label-count
          estimator.

    """
    losses: dict = {
        "grad_match": (CosineGradientLoss(), 1.0),
        "tv": (TVLoss(space=space), tv_weight),
    }
    if lambda_inv > 0:
        losses["epoch_inv"] = (EpochOrderInvariantPrior(), lambda_inv)

    return Attack(
        threat_model=ThreatModel(has_local_hyperparameters=True),
        modality=ImageNormalizationModality(mean=mean, std=std, device=device),
        evaluator=ImageEvaluator(),
        task=Task(
            CrossEntropy(),
            FedAvg(local_epochs=num_epochs, batch_size=client_batch_size, lr=client_lr),
            model_mode=model_mode,
        ),
        target_inference=labels_to_inference(labels, allow=("oracle", "idlg", "dimitrov")),
        axes=AxisManager(Axis("epoch", num_epochs)),
        stages=[
            OptimStage(
                losses=losses,
                rep=pixel_representation(space),
                driver=GradientDriver(
                    lr=lr, max_iter=max_iter, return_best=True,
                    schedule=schedule, grad_transforms=grad_transforms, patience=patience,
                ),
                target_source=FixedTarget(),
                aggregator=EpochMatchingAggregator(),
            )
        ],
    )
