#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""A seed axis works with representations whose decode builds one graph shared by every seed slice.

The gradient driver decodes once per evaluation and backpropagates each slice's loss; the shared decode
graph (here the model-input-to-data map under a data-space prior; a generator in
``test_gan_multi_seed_shared_decode.py``) must survive every slice.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from leakpro.attacks.gia_attacks.modular import Attack, FedSGD, OptimStage, TargetInfo, Task, run_attack, simulate_client
from leakpro.attacks.gia_attacks.modular.components.aggregators import BestLossAggregator
from leakpro.attacks.gia_attacks.modular.components.drivers import GradientDriver
from leakpro.attacks.gia_attacks.modular.components.losses import CosineGradientLoss, TVLoss
from leakpro.attacks.gia_attacks.modular.components.modalities import ImageNormalizationModality
from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.components.representations import PixelRepresentation
from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
from leakpro.attacks.gia_attacks.modular.core.axes import Axis, AxisManager

CPU = torch.device("cpu")


def _run(rep, seeds: int = 2):  # noqa: ANN001, ANN202
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 8 * 8, 5))
    x, y = torch.rand(2, 3, 8, 8), torch.tensor([1, 3])
    attack = Attack(
        modality=ImageNormalizationModality(mean=[0.5] * 3, std=[0.25] * 3, device=CPU),
        task=Task(CrossEntropy(), FedSGD(), model_mode="eval"),
        axes=AxisManager(Axis("seed", seeds)),
        stages=[OptimStage(
            losses={"g": (CosineGradientLoss(), 1.0), "tv": (TVLoss(), 1e-2)}, rep=rep,
            driver=GradientDriver(lr=0.1, max_iter=2), target_source=FixedTarget(), aggregator=BestLossAggregator(),
        )],
    )
    target = TargetInfo(value=y)
    obs = simulate_client(model, x, target, task=attack.task, modality=attack.modality)
    return run_attack(attack, model, obs, tuple(x.shape), CPU, target=target)


@pytest.mark.parametrize(
    "rep_fn", [lambda: PixelRepresentation(space="model_input")], ids=["pixel-model-input-with-data-space-tv"],
)
def test_two_seeds_run_through_a_shared_decode_graph(rep_fn) -> None:  # noqa: ANN001
    state, _ = _run(rep_fn())
    assert state.input.shape == (2, 3, 8, 8)
    assert torch.isfinite(state.input).all()

