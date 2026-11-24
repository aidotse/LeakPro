#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Early stopping: the reference stagnation counter, on a run whose loss cannot improve (lr = 0)."""
from __future__ import annotations

import pytest

from leakpro.attacks.gia_attacks.modular.components.drivers import GradientDriver
from leakpro.attacks.gia_attacks.modular.core.attack import run_attack
from leakpro.attacks.gia_attacks.modular.core.events import StepEvent
from leakpro.attacks.gia_attacks.modular.core.interfaces import Callback
from leakpro.tests.gia import helpers
from leakpro.tests.gia.unit.test_grad_transforms import _attack


class _Count(Callback):
    def __init__(self) -> None:
        self.steps = 0

    def on_step_end(self, event: StepEvent) -> None:
        self.steps += 1


def _steps(driver: GradientDriver) -> int:
    attack, shape, target, obs = _attack(driver)
    counter = _Count()
    run_attack(attack, helpers.tiny_bn_cnn().train(), obs, shape, helpers.DEVICE, target=target, callbacks=[counter])
    return counter.steps


def test_a_run_that_never_improves_stops_after_patience_stagnant_steps() -> None:
    # Step 0 sets the best; steps 1..patience are stagnant, so the run is patience + 1 steps long.
    assert _steps(GradientDriver(lr=0.0, max_iter=50, patience=4)) == 5


def test_without_patience_the_run_uses_every_iteration() -> None:
    assert _steps(GradientDriver(lr=0.0, max_iter=12)) == 12


def test_an_improving_run_is_not_stopped_early() -> None:
    assert _steps(GradientDriver(lr=0.05, max_iter=6, patience=6)) == 6


def test_patience_must_be_positive() -> None:
    with pytest.raises(ValueError, match="patience"):
        GradientDriver(patience=0)
