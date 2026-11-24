#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import torch

from leakpro.attacks.gia_attacks.modular.components.initializers import FromState, FromTensor, Gaussian, Uniform

CPU = torch.device("cpu")


def test_uniform_without_seed_matches_the_legacy_global_rng_draw() -> None:
    torch.manual_seed(3)
    expected = torch.empty(2, 3, 4, 4).uniform_(0.0, 1.0)
    torch.manual_seed(3)
    torch.testing.assert_close(Uniform().sample((2, 3, 4, 4), CPU), expected, rtol=0, atol=0)


def test_seeded_initializers_are_reproducible_and_leave_the_global_rng_alone() -> None:
    torch.manual_seed(0)
    before = torch.get_rng_state()
    a = Gaussian(seed=5).sample((4,), CPU)
    b = Gaussian(seed=5).sample((4,), CPU)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), before)


def test_gaussian_moments() -> None:
    x = Gaussian(mean=2.0, std=0.5, seed=0).sample((20000,), CPU)
    assert x.mean().item() == pytest.approx(2.0, abs=0.02)
    assert x.std().item() == pytest.approx(0.5, abs=0.02)


def test_from_tensor_returns_an_independent_copy_and_checks_shape() -> None:
    t = torch.arange(6.0).view(2, 3)
    out = FromTensor(t).sample((2, 3), CPU)
    out += 1
    assert torch.equal(t, torch.arange(6.0).view(2, 3))
    with pytest.raises(ValueError, match=r"\(2, 3\).*\(3, 2\)"):
        FromTensor(t).sample((3, 2), CPU)


def test_from_state_cannot_sample() -> None:
    with pytest.raises(ValueError, match="state"):
        FromState().sample((1,), CPU)


def test_space_is_validated() -> None:
    with pytest.raises(ValueError, match="space"):
        Uniform(space="pixels")
