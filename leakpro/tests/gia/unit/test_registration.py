#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Seed registration before averaging (See Through Gradients)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from leakpro.attacks.gia_attacks.modular import presets
from leakpro.attacks.gia_attacks.modular.components.aggregators import MeanAggregator
from leakpro.attacks.gia_attacks.modular.components.drivers.grad_transforms import GaussianNoise
from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
from leakpro.attacks.gia_attacks.modular.components.utils.registration import align_translation
from leakpro.attacks.gia_attacks.modular.core.axes import Axis, AxisManager
from leakpro.attacks.gia_attacks.modular.core.result import DriverResult
from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo
from leakpro.tests.gia import helpers

SHIFTS = [(0, 0), (2, -1), (-3, 2)]


def _seeds() -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.rand(2, 3, 16, 16, generator=torch.Generator().manual_seed(0))
    return base, torch.stack([base.roll(s, dims=(-2, -1)) for s in SHIFTS])  # [S, N, C, H, W]


def _aggregate(aggregator, seeds):  # noqa: ANN001, ANN202
    result = DriverResult(input=seeds, target=TargetInfo(value=torch.tensor([1, 2])), artifacts={}, loss=0.0,
                          iterations=1, slice_losses=torch.zeros(seeds.shape[0]))
    ctx = SimpleNamespace(axes=AxisManager(Axis("seed", seeds.shape[0])))
    stage = SimpleNamespace(target_source=FixedTarget())
    return aggregator.aggregate(result, ctx, stage).input


def test_align_recovers_a_known_shift() -> None:
    base, seeds = _seeds()
    torch.testing.assert_close(align_translation(base, seeds[1], max_shift=4), base, rtol=0, atol=0)


def test_zero_shift_budget_is_the_identity_and_negative_is_rejected() -> None:
    base, seeds = _seeds()
    assert torch.equal(align_translation(base, seeds[1], max_shift=0), seeds[1])
    with pytest.raises(ValueError, match="max_shift"):
        align_translation(base, seeds[1], max_shift=-1)


def test_registered_mean_recovers_the_base_image_where_the_plain_mean_blurs_it() -> None:
    base, seeds = _seeds()
    torch.testing.assert_close(_aggregate(MeanAggregator(max_shift=4), seeds), base, rtol=0, atol=1e-6)
    assert not torch.allclose(_aggregate(MeanAggregator(), seeds), base, atol=1e-3)


def test_the_yin_preset_wires_noise_and_registration() -> None:
    kw = {"mean": helpers.CIFAR_MEAN, "std": helpers.CIFAR_STD, "device": helpers.DEVICE}
    plain = presets.yin(**kw)
    assert plain.stages[0].driver.grad_transforms == ()
    assert isinstance(plain.stages[0].aggregator, MeanAggregator)
    tuned = presets.yin(**kw, gradient_noise_std=0.2, registration_max_shift=3)
    assert tuned.stages[0].driver.grad_transforms == (GaussianNoise(0.2),)
    assert tuned.stages[0].aggregator.max_shift == 3
