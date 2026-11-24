#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""GIA Running / GIA Estimate: threat models, requirement checks and a behaviour check."""
from __future__ import annotations

import dataclasses

import pytest
import torch

from leakpro.attacks.gia_attacks.modular import presets
from leakpro.attacks.gia_attacks.modular.components.drivers.schedules import MultiStep
from leakpro.attacks.gia_attacks.modular.core.attack import run_attack
from leakpro.attacks.gia_attacks.modular.core.requirements import validate_requirements
from leakpro.attacks.gia_attacks.modular.core.state import Context, TargetInfo
from leakpro.tests.gia import helpers
from leakpro.tests.gia.parity.v1_equivalent import v1_step_milestones

KW = {"mean": helpers.CIFAR_MEAN, "std": helpers.CIFAR_STD, "device": helpers.DEVICE}


def _setup(attack):  # noqa: ANN001, ANN202
    images, labels = helpers.client_batch()
    obs = helpers.observe(attack, helpers.tiny_bn_cnn().train(), images, labels)
    return images, labels, obs


def test_running_declares_what_the_inferred_term_needs_and_runs() -> None:
    attack = presets.gia_running(**KW, max_iter=2)
    assert attack.threat_model.has_bn_statistics and attack.threat_model.has_local_hyperparameters
    images, labels, obs = _setup(attack)
    state, _ = run_attack(attack, helpers.tiny_bn_cnn().train(), obs, tuple(images.shape), helpers.DEVICE,
                          target=TargetInfo(value=labels))
    assert torch.isfinite(state.input).all() and state.loss < float("inf")


def test_running_without_the_batch_size_fails_before_running() -> None:
    attack = presets.gia_running(**KW, max_iter=1)
    _, _, obs = _setup(attack)
    obs = dataclasses.replace(obs, batch_size=None)
    with pytest.raises(ValueError, match="batch_size"):
        validate_requirements(attack, obs, proxy_dataloader=None)


def test_estimate_requires_a_proxy_dataloader_and_runs_with_one() -> None:
    attack = presets.gia_estimate(**KW, max_iter=2)
    assert attack.threat_model is not None
    images, labels, obs = _setup(attack)
    with pytest.raises(ValueError, match="proxy"):
        validate_requirements(attack, attack.threat_model.sanitize(obs), proxy_dataloader=None)
    proxy = [(torch.rand(4, *helpers.IMAGE_SHAPE), torch.randint(0, helpers.NUM_CLASSES, (4,)))]
    state, _ = run_attack(attack, helpers.tiny_bn_cnn().train(), obs, tuple(images.shape), helpers.DEVICE,
                          target=TargetInfo(value=labels), proxy_dataloader=proxy)
    assert torch.isfinite(state.input).all()


@pytest.mark.parametrize("name", ["gia_running", "gia_estimate"])
def test_schedules_default_to_reference_milestones_for_4800_steps(name: str) -> None:
    schedule = getattr(presets, name)(**KW).stages[0].driver.schedule
    assert isinstance(schedule, MultiStep)
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    assert sorted(schedule.build(opt, 4800).milestones) == sorted(v1_step_milestones(4800))
