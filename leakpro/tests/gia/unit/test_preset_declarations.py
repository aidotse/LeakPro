#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Every shipped preset states what the attacker knows.

This file covers the image-classification presets. Each feature area declares its presets' builders in
``test_preset_declarations_<area>.py``; ``test_builders_cover_every_preset`` checks them all together.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch

from leakpro.attacks.gia_attacks.modular import presets
from leakpro.attacks.gia_attacks.modular.core.attack import run_attack
from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo
from leakpro.tests.gia import helpers

KW = {"mean": helpers.CIFAR_MEAN, "std": helpers.CIFAR_STD, "device": helpers.DEVICE}

BUILDERS = {
    "dlg": lambda: presets.dlg(**KW, num_classes=helpers.NUM_CLASSES),
    "idlg": lambda: presets.idlg(**KW),
    "inverting_gradients": lambda: presets.inverting_gradients(**KW),
    "huang": lambda: presets.huang(**KW),
    "dimitrov": lambda: presets.dimitrov(**KW),
    "yin": lambda: presets.yin(**KW),
    "see_through_gradients": lambda: presets.see_through_gradients(**KW),
    "combined": lambda: presets.combined(**KW),
    "gia_running": lambda: presets.gia_running(**KW),
    "gia_estimate": lambda: presets.gia_estimate(**KW),
}

SMOKE = {
    "dlg": {"max_iter": 1, "num_classes": helpers.NUM_CLASSES},
    "idlg": {"max_iter": 1},
    "inverting_gradients": {"max_iter": 1},
    "dimitrov": {"max_iter": 1, "num_epochs": 1},
    "combined": {"max_iter": 1, "num_epochs": 1, "num_seeds": 1},
}


def test_builders_cover_every_preset() -> None:
    covered = set()
    for path in Path(__file__).parent.glob("test_preset_declarations*.py"):
        covered |= set(importlib.import_module(f"{__package__}.{path.stem}").BUILDERS)
    assert covered == set(presets.__all__)


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_preset_declares_a_threat_model(name: str) -> None:
    assert BUILDERS[name]().threat_model is not None, f"{name} leaves its threat model undeclared"


@pytest.mark.parametrize("name", sorted(SMOKE))
def test_image_preset_runs_on_what_its_threat_model_grants(name: str) -> None:
    attack = getattr(presets, name)(**KW, **SMOKE[name])
    images, labels = helpers.client_batch()
    obs = helpers.observe(attack, helpers.tiny_bn_cnn().train(), images, labels)
    state, _ = run_attack(
        attack, helpers.tiny_bn_cnn().train(), obs, tuple(images.shape), helpers.DEVICE, target=TargetInfo(value=labels),
    )
    assert torch.isfinite(state.input).all()
