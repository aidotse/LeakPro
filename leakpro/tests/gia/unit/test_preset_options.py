#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Presets expose the reference options; defaults are checked separately (test_preset_defaults.py)."""
from __future__ import annotations

import pytest

from leakpro.attacks.gia_attacks.modular import presets
from leakpro.attacks.gia_attacks.modular.components.drivers.grad_transforms import GaussianNoise, SignGradient
from leakpro.attacks.gia_attacks.modular.components.drivers.schedules import MultiStep
from leakpro.attacks.gia_attacks.modular.components.target_inference import DimitrovInference, IDLGInference
from leakpro.attacks.gia_attacks.modular.core.protocol import FedAvg
from leakpro.tests.gia import helpers

KW = {"mean": helpers.CIFAR_MEAN, "std": helpers.CIFAR_STD, "device": helpers.DEVICE}
GRADIENT_PRESETS = ["inverting_gradients", "huang", "yin", "dimitrov"]


@pytest.mark.parametrize("name", GRADIENT_PRESETS)
def test_driver_options_reach_the_driver(name: str) -> None:
    transforms = (GaussianNoise(0.1, seed=0), SignGradient())
    attack = getattr(presets, name)(**KW, schedule=MultiStep(), grad_transforms=transforms, patience=7)
    driver = attack.stages[0].driver
    assert driver.schedule == MultiStep()
    assert driver.grad_transforms == transforms
    assert driver.patience == 7


@pytest.mark.parametrize("name", GRADIENT_PRESETS)
def test_model_input_space_uses_a_gaussian_start_and_the_model_input_parameterisation(name: str) -> None:
    rep = getattr(presets, name)(**KW, space="model_input").stages[0].rep
    assert rep.space == "model_input"
    assert type(rep.init).__name__ == "Gaussian"


@pytest.mark.parametrize("name", GRADIENT_PRESETS)
def test_oracle_labels_leave_no_inference_step(name: str) -> None:
    assert getattr(presets, name)(**KW, labels="oracle").target_inference is None
    assert isinstance(getattr(presets, name)(**KW, labels="idlg").target_inference, IDLGInference)


def test_dimitrov_labels_and_client_hyperparameters() -> None:
    attack = presets.dimitrov(**KW, num_epochs=2, client_lr=0.3, client_batch_size=2, labels="dimitrov")
    assert isinstance(attack.target_inference, DimitrovInference)
    protocol = attack.task.protocol
    assert isinstance(protocol, FedAvg)
    assert (protocol.local_epochs, protocol.batch_size, protocol.lr) == (2, 2, 0.3)


def test_unknown_label_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="oracle"):
        presets.huang(**KW, labels="geng")  # type: ignore[arg-type]
