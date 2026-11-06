#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""A tiny attack actually reconstructs the client's image: the end-to-end check that the pieces work together."""
import torch
from torch import nn

from leakpro.attacks.gia_attacks.modular import presets
from leakpro.attacks.gia_attacks.modular.components.utils import image_metrics
from leakpro.attacks.gia_attacks.modular.core.attack import run_attack
from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo
from leakpro.tests.gia import helpers


def test_dlg_reconstructs_a_single_image_from_a_small_mlp() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 16), nn.Sigmoid(), nn.Linear(16, helpers.NUM_CLASSES))
    attack = presets.dlg(helpers.CIFAR_MEAN, helpers.CIFAR_STD, helpers.DEVICE, helpers.NUM_CLASSES, max_iter=60)
    images, labels = helpers.client_batch(1)
    observation = helpers.observe(attack, model, images, labels)

    start = run_attack(presets.dlg(helpers.CIFAR_MEAN, helpers.CIFAR_STD, helpers.DEVICE, helpers.NUM_CLASSES,
                                   max_iter=1), model, observation, (1, *helpers.IMAGE_SHAPE), helpers.DEVICE,
                       target=TargetInfo(value=labels))[0]
    state, _ = run_attack(attack, model, observation, (1, *helpers.IMAGE_SHAPE), helpers.DEVICE,
                          target=TargetInfo(value=labels))

    psnr_start = image_metrics.psnr_per_image(start.input.clamp(0, 1), images).item()
    psnr_end = image_metrics.psnr_per_image(state.input.clamp(0, 1), images).item()
    assert psnr_start < 12.0
    assert psnr_end > 30.0
