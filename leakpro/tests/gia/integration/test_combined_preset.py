#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""The combined multi-epoch + multi-seed preset collapses both axes whichever order they are declared in."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from leakpro.attacks.gia_attacks.modular.core.attack import run_attack
from leakpro.attacks.gia_attacks.modular.core.axes import Axis, AxisManager
from leakpro.attacks.gia_attacks.modular.core.interfaces import Driver
from leakpro.attacks.gia_attacks.modular.core.result import DriverResult
from leakpro.attacks.gia_attacks.modular.core.simulate import simulate_client
from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo
from leakpro.attacks.gia_attacks.modular.presets.combined_gia import combined
from leakpro.tests.gia import helpers

EPOCHS, SEEDS, BATCH = 3, 2, 2


class _KnownResultDriver(Driver):
    """Returns image ``n`` filled with the value ``n`` on every axis slice; slice (seed 1) has the lowest loss."""

    def __init__(self, axes: AxisManager) -> None:
        self.axes = axes

    def run(self, stage, ctx, *, callbacks=(), stage_idx=0):  # noqa: ANN001, ARG002
        shape = (*self.axes.shape, BATCH, *helpers.IMAGE_SHAPE)
        values = torch.arange(BATCH, dtype=torch.float32).view(*([1] * len(self.axes.shape)), BATCH, 1, 1, 1)
        recon = values.expand(shape).clone()
        seed_losses = torch.tensor([0.9, 0.1])
        return DriverResult(input=recon, target=stage.target_source.current(), artifacts={}, loss=0.1,
                            iterations=1, slice_losses=seed_losses)


def _attack(order):
    attack = combined(helpers.CIFAR_MEAN, helpers.CIFAR_STD, helpers.DEVICE,
                      num_epochs=EPOCHS, num_seeds=SEEDS, max_iter=3)
    sizes = {"epoch": EPOCHS, "seed": SEEDS}
    return replace(attack, axes=AxisManager(*(Axis(name, sizes[name]) for name in order)))


def _observe(attack, model, images, labels):  # noqa: ANN001, ANN202
    return simulate_client(
        model, images.unsqueeze(0).repeat(EPOCHS, 1, 1, 1, 1), TargetInfo(value=labels),
        task=attack.task, modality=attack.modality, input_shape=(BATCH, *helpers.IMAGE_SHAPE),
    )


@pytest.mark.parametrize("order", [("epoch", "seed"), ("seed", "epoch")])
def test_combined_preset_runs_end_to_end_to_the_canonical_shape(order):
    model = helpers.tiny_bn_cnn()
    attack = _attack(order)
    images, labels = helpers.client_batch(BATCH)
    observation = _observe(attack, model, images, labels)

    state, _ = run_attack(attack, model, observation, (BATCH, *helpers.IMAGE_SHAPE), helpers.DEVICE,
                          target=TargetInfo(value=labels))

    assert tuple(state.input.shape) == (BATCH, *helpers.IMAGE_SHAPE)


@pytest.mark.parametrize("order", [("epoch", "seed"), ("seed", "epoch")])
def test_combined_preset_aggregation_does_not_mix_images_across_the_batch(order):
    attack = _attack(order)
    stage = replace(attack.stages[0], driver=_KnownResultDriver(attack.axes))
    attack = replace(attack, stages=[stage])
    model = helpers.tiny_bn_cnn()
    images, labels = helpers.client_batch(BATCH)
    shape = (BATCH, *helpers.IMAGE_SHAPE)

    state, _ = run_attack(attack, model, _observe(attack, model, images, labels), shape, helpers.DEVICE,
                          target=TargetInfo(value=labels))

    assert tuple(state.input.shape) == shape
    for n in range(BATCH):
        torch.testing.assert_close(state.input[n], torch.full(helpers.IMAGE_SHAPE, float(n)))
