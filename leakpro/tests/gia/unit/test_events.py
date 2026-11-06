#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
import torch

from leakpro.attacks.gia_attacks.modular import presets
from leakpro.attacks.gia_attacks.modular.core.attack import run_attack
from leakpro.attacks.gia_attacks.modular.core.events import StepEvent
from leakpro.attacks.gia_attacks.modular.core.interfaces import Callback
from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo
from leakpro.tests.gia import helpers


class _Events(Callback):
    def __init__(self) -> None:
        self.events: list[StepEvent] = []

    def on_step_end(self, event: StepEvent) -> None:
        self.events.append(event)


def test_step_events_carry_stage_step_loss_and_artifacts() -> None:
    attack = presets.inverting_gradients(mean=helpers.CIFAR_MEAN, std=helpers.CIFAR_STD, device=helpers.DEVICE, max_iter=3)
    model = helpers.tiny_bn_cnn().train()
    images, labels = helpers.client_batch()
    obs = helpers.observe(attack, model, images, labels)
    rec = _Events()
    run_attack(attack, model, obs, tuple(images.shape), helpers.DEVICE, target=TargetInfo(value=labels), callbacks=[rec])
    assert [e.step for e in rec.events] == [0, 1, 2]
    assert {e.stage_idx for e in rec.events} == {0}
    assert all(e.loss == e.loss for e in rec.events)              # finite, not NaN
    assert set(rec.events[0].components) == {"grad_match", "tv"}
    assert rec.events[0].artifacts["pixels"].shape == images.shape
