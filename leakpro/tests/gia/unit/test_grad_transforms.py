#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Gradient noise and clipping: the reference formula, seeding, order and driver integration."""
from __future__ import annotations

import pytest
import torch

from leakpro.attacks.gia_attacks.modular.components.drivers import GradientDriver
from leakpro.attacks.gia_attacks.modular.components.drivers.grad_transforms import (
    ClipGradNorm,
    GaussianNoise,
    SignGradient,
)
from leakpro.attacks.gia_attacks.modular.components.losses import L2GradientLoss
from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.components.representations.pixel import PixelRepresentation
from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
from leakpro.attacks.gia_attacks.modular.core.attack import Attack, run_attack
from leakpro.attacks.gia_attacks.modular.core.protocol import FedSGD
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo
from leakpro.attacks.gia_attacks.modular.core.task import Task
from leakpro.tests.gia import helpers

CPU = torch.device("cpu")


def _param(grad: torch.Tensor) -> torch.Tensor:
    p = torch.zeros_like(grad, requires_grad=True)
    p.grad = grad.clone()
    return p


def test_noise_follows_reference_formula_and_draw_order() -> None:
    p, q = _param(torch.ones(3, 4)), _param(torch.ones(2))
    torch.manual_seed(5)
    expected_p = torch.ones(3, 4) + torch.randn(3, 4) * 0.2
    expected_q = torch.ones(2) + torch.randn(2) * 0.2
    torch.manual_seed(5)
    GaussianNoise(0.2).bind(CPU)([p, q])
    torch.testing.assert_close(p.grad, expected_p, rtol=0, atol=0)
    torch.testing.assert_close(q.grad, expected_q, rtol=0, atol=0)


def test_seeded_noise_is_reproducible_and_leaves_the_global_rng_alone() -> None:
    a, b, c = (_param(torch.zeros(5)) for _ in range(3))
    state = torch.get_rng_state()
    GaussianNoise(1.0, seed=3).bind(CPU)([a])
    GaussianNoise(1.0, seed=3).bind(CPU)([b])
    GaussianNoise(1.0, seed=4).bind(CPU)([c])
    assert torch.equal(torch.get_rng_state(), state)
    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    assert not torch.equal(a.grad, c.grad)


def test_zero_std_is_a_no_op_that_consumes_no_randomness() -> None:
    p = _param(torch.ones(4))
    state = torch.get_rng_state()
    GaussianNoise(0.0).bind(CPU)([p])
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(p.grad, torch.ones(4))


def test_negative_std_is_rejected() -> None:
    with pytest.raises(ValueError, match="std"):
        GaussianNoise(-0.1)


def test_noise_then_sign_gives_signs_of_the_noise() -> None:
    p = _param(torch.zeros(50))
    for t in (GaussianNoise(1.0, seed=0).bind(CPU), SignGradient()):
        t([p])
    assert set(p.grad.unique().tolist()) <= {-1.0, 1.0}
    assert (p.grad == 0).sum() == 0  # sign(0 + noise): the reference order; sign first would leave raw noise


def test_clip_scales_to_the_max_norm_and_leaves_small_gradients_alone() -> None:
    p, q = _param(torch.full((4,), 3.0)), _param(torch.full((4,), 3.0))
    ClipGradNorm(1.0)([p, q])
    total = torch.sqrt((p.grad**2).sum() + (q.grad**2).sum())
    assert total == pytest.approx(1.0, abs=1e-5)
    small = _param(torch.full((4,), 0.01))
    ClipGradNorm(1.0)([small])
    assert torch.equal(small.grad, torch.full((4,), 0.01))


def _attack(driver: GradientDriver) -> tuple[Attack, tuple[int, ...], TargetInfo, object]:
    images, labels = helpers.client_batch()
    attack = Attack(
        modality=helpers.image_modality(), task=Task(CrossEntropy(), FedSGD()),
        stages=[OptimStage(losses={"l2": (L2GradientLoss(), 1.0)}, rep=PixelRepresentation(), driver=driver,
                           target_source=FixedTarget())],
    )
    obs = helpers.observe(attack, helpers.tiny_bn_cnn().train(), images, labels)
    return attack, tuple(images.shape), TargetInfo(value=labels), obs


def _final(driver: GradientDriver) -> torch.Tensor:
    attack, shape, target, obs = _attack(driver)
    state, _ = run_attack(attack, helpers.tiny_bn_cnn().train(), obs, shape, helpers.DEVICE, target=target)
    return state.input


def test_driver_runs_with_seeded_noise_reproducibly_and_zero_noise_equals_none() -> None:
    plain = _final(GradientDriver(max_iter=3))
    zero = _final(GradientDriver(max_iter=3, grad_transforms=(GaussianNoise(0.0),)))
    noisy_a = _final(GradientDriver(max_iter=3, grad_transforms=(GaussianNoise(0.5, seed=1),)))
    noisy_b = _final(GradientDriver(max_iter=3, grad_transforms=(GaussianNoise(0.5, seed=1),)))
    torch.testing.assert_close(zero, plain, rtol=0, atol=0)
    torch.testing.assert_close(noisy_a, noisy_b, rtol=0, atol=0)
    assert not torch.equal(noisy_a, plain)
