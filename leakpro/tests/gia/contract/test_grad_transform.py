#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""GradTransform contract: only ``.grad`` changes, and parameters without a gradient are skipped."""
from __future__ import annotations

import pytest
import torch

from leakpro.attacks.gia_attacks.modular.core.interfaces import GradTransform
from leakpro.tests.gia.contract.registry import build, cases

pytestmark = pytest.mark.parametrize(("cls", "factory"), cases(GradTransform))


def test_transform_modifies_only_the_gradients(cls, factory) -> None:  # noqa: ANN001
    transform = build(cls, factory).component.bind(torch.device("cpu"))
    gen = torch.Generator().manual_seed(7)
    with_grad = torch.randn(3, 4, generator=gen).requires_grad_(True)
    with_grad.grad = torch.randn(3, 4, generator=gen)
    without_grad = torch.randn(2, generator=gen).requires_grad_(True)
    data_before = (with_grad.detach().clone(), without_grad.detach().clone())
    grad_storage = with_grad.grad

    transform([with_grad, without_grad])

    torch.testing.assert_close(with_grad.detach(), data_before[0], rtol=0, atol=0)
    torch.testing.assert_close(without_grad.detach(), data_before[1], rtol=0, atol=0)
    assert without_grad.grad is None
    assert with_grad.grad is grad_storage, "the transform rewrites .grad in place"
    assert with_grad.grad.shape == with_grad.shape and torch.isfinite(with_grad.grad).all()
