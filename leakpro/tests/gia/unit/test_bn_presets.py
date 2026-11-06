#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Presets with a BN-statistics loss run on what their own threat model grants."""
from __future__ import annotations

import torch

from leakpro.attacks.gia_attacks.modular import presets
from leakpro.attacks.gia_attacks.modular.core.attack import run_attack
from leakpro.attacks.gia_attacks.modular.core.requirements import validate_requirements
from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo
from leakpro.tests.gia import helpers

KW = {"mean": helpers.CIFAR_MEAN, "std": helpers.CIFAR_STD, "device": helpers.DEVICE}


def _observation(attack):  # noqa: ANN001, ANN202
    images, labels = helpers.client_batch()
    return images, labels, helpers.observe(attack, helpers.tiny_bn_cnn().train(), images, labels)


def _run_end_to_end(attack) -> None:  # noqa: ANN001
    images, labels, obs = _observation(attack)
    model = helpers.tiny_bn_cnn().train()
    state, _ = run_attack(attack, model, obs, tuple(images.shape), helpers.DEVICE, target=TargetInfo(value=labels))
    assert torch.isfinite(state.input).all()


def test_huang_runs_on_its_threat_model() -> None:
    _run_end_to_end(presets.huang(**KW, max_iter=1))


def test_yin_runs_on_its_threat_model() -> None:
    _run_end_to_end(presets.yin(**KW, num_seeds=2, max_iter=1))
