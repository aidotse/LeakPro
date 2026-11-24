#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""The approved preset defaults, read off the Attack each preset builds."""
from __future__ import annotations

import pytest
import torch

from leakpro.attacks.gia_attacks.modular import presets
from leakpro.attacks.gia_attacks.modular.components.drivers.grad_transforms import SignGradient
from leakpro.attacks.gia_attacks.modular.components.drivers.schedules import Cosine, MultiStep
from leakpro.attacks.gia_attacks.modular.components.initializers import Gaussian
from leakpro.attacks.gia_attacks.modular.core.protocol import FedAvg, FedSGD
from leakpro.tests.gia import helpers
from leakpro.tests.gia.parity.v1_equivalent import v1_step_milestones

KW = {"mean": helpers.CIFAR_MEAN, "std": helpers.CIFAR_STD, "device": helpers.DEVICE}


def _stage(attack):  # noqa: ANN001, ANN202
    return attack.stages[0]


def _milestones(schedule, n: int) -> list[int]:  # noqa: ANN001
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    return sorted(schedule.build(opt, n).milestones)


def test_inverting_gradients_defaults() -> None:           # R1 R2 R3 R4 R5 R6 R7 R13
    attack = presets.inverting_gradients(**KW)
    stage = _stage(attack)
    assert stage.rep.space == "model_input" and isinstance(stage.rep.init, Gaussian) and stage.rep.clip
    assert stage.driver.grad_transforms == (SignGradient(),)
    assert stage.driver.schedule == MultiStep()
    assert (stage.driver.lr, stage.driver.max_iter, stage.driver.patience) == (0.1, 4000, None)
    assert stage.losses["tv"][1] == 1e-3 and stage.losses["tv"][0].space == "model_input"
    assert attack.target_inference is None                  # oracle labels
    assert isinstance(attack.task.protocol, FedSGD) and attack.task.model_mode == "eval"


def test_idlg_defaults() -> None:                            # R1 R7 R9
    attack = presets.idlg(**KW)
    stage = _stage(attack)
    assert stage.rep.space == "model_input" and stage.rep.clip is False
    assert attack.task.model_mode == "eval"
    assert attack.target_inference is not None               # analytical label inference


def test_huang_defaults() -> None:                           # R1 R3 R4 R5 R6 R7
    attack = presets.huang(**KW)
    stage = _stage(attack)
    assert stage.rep.space == "model_input" and isinstance(stage.rep.init, Gaussian)
    assert (stage.driver.max_iter, stage.losses["tv"][1], stage.losses["bn_stats"][1]) == (10000, 0.052, 1.6e-4)
    assert _milestones(stage.driver.schedule, 4800) == sorted(v1_step_milestones(4800))
    assert stage.driver.grad_transforms == () and stage.driver.patience is None
    assert attack.target_inference is None and attack.task.model_mode == "train"


@pytest.mark.parametrize(("name", "iters"), [("gia_running", 3000), ("gia_estimate", 10000)])
def test_gia_running_and_estimate_defaults(name: str, iters: int) -> None:        # R3 R14
    attack = getattr(presets, name)(**KW)
    stage = _stage(attack)
    assert (stage.driver.max_iter, stage.losses["tv"][1], stage.losses["bn_stats"][1]) == (iters, 0.052, 1.6e-4)
    assert _milestones(stage.driver.schedule, 4800) == sorted(v1_step_milestones(4800))
    assert stage.rep.space == "model_input"
    assert isinstance(attack.task.protocol, FedSGD) and attack.task.model_mode == "train"


def test_yin_defaults() -> None:                             # R1 R5 R6 R10 R11 R12 R13
    attack = presets.yin(**KW)
    stage = _stage(attack)
    assert stage.rep.space == "model_input" and isinstance(stage.rep.init, Gaussian)
    assert (stage.driver.max_iter, stage.driver.grad_transforms, stage.driver.patience) == (8000, (), None)
    assert isinstance(stage.driver.schedule, Cosine)
    assert stage.losses["group"][1] == 1e-2 and attack.axes.get_size("seed") == 4
    assert stage.losses["group"][0].space == "model_input"
    assert stage.losses["l2_image"][0].space == "model_input" and stage.losses["l2_image"][0].reduction == "per_image_sum"
    assert attack.target_inference is None
    assert type(stage.aggregator).__name__ == "MeanAggregator"


def test_dimitrov_defaults() -> None:                        # R1 R5 R6 R8 R13
    attack = presets.dimitrov(**KW)
    stage = _stage(attack)
    assert stage.rep.space == "model_input" and isinstance(stage.rep.init, Gaussian)
    assert (stage.driver.lr, stage.driver.max_iter, stage.driver.patience) == (1.0, 8000, None)
    assert isinstance(stage.driver.schedule, Cosine)
    assert isinstance(attack.task.protocol, FedAvg) and attack.task.protocol.batch_size == 1
    assert attack.task.model_mode == "eval"
    assert attack.target_inference is None
