#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Contracts of the diffusion driver's strategies: mean adjustment, noise injection, blending schedules."""
from __future__ import annotations

import pytest
import torch

from leakpro.attacks.gia_attacks.modular.components.diffusion.mean_strategies import MeanAdjustmentStrategy
from leakpro.attacks.gia_attacks.modular.components.diffusion.noise_strategies import NoiseInjectionStrategy
from leakpro.attacks.gia_attacks.modular.components.diffusion.schedules import BlendingSchedule
from leakpro.tests.gia.contract.registry import build, cases

SHAPE = (2, 3, 4, 4)


@pytest.mark.parametrize(("cls", "factory"), cases(MeanAdjustmentStrategy))
def test_mean_adjustment_returns_the_adjusted_mean_and_its_shift(cls, factory) -> None:  # noqa: ANN001
    strategy = build(cls, factory).component
    gen = torch.Generator().manual_seed(8)
    mu, xt = torch.randn(SHAPE, generator=gen), torch.randn(SHAPE, generator=gen)
    goal = torch.randn(SHAPE, generator=gen)
    mu_before = mu.clone()

    adjusted, delta = strategy.adjust(mu, xt, 0.5, 2, 4, lambda x: (x - goal).pow(2).sum())

    assert adjusted.shape == mu.shape and delta.shape == mu.shape
    assert not adjusted.requires_grad and not delta.requires_grad
    torch.testing.assert_close(adjusted - mu, delta)
    torch.testing.assert_close(mu, mu_before, rtol=0, atol=0)
    assert delta.abs().sum() > 0, "a non-constant objective shifts the mean"


@pytest.mark.parametrize(("cls", "factory"), cases(NoiseInjectionStrategy))
def test_noise_has_the_requested_shape_dtype_and_device(cls, factory) -> None:  # noqa: ANN001
    strategy = build(cls, factory).component
    for delta_mu in (torch.zeros(SHAPE), torch.ones(SHAPE)):
        noise = strategy.sample(SHAPE, torch.device("cpu"), torch.float32, delta_mu)
        assert tuple(noise.shape) == SHAPE and noise.dtype == torch.float32 and noise.device.type == "cpu"
        assert torch.isfinite(noise).all()


@pytest.mark.parametrize(("cls", "factory"), cases(BlendingSchedule))
def test_blend_weight_lies_in_the_unit_interval(cls, factory) -> None:  # noqa: ANN001
    schedule = build(cls, factory).component
    for num_timesteps in (1, 4, 100):
        for t in range(num_timesteps):
            assert 0.0 <= schedule.gamma(t, num_timesteps) <= 1.0, (t, num_timesteps)
