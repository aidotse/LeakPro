#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Objective contract: a differentiable scalar loss, the same through the module and a functional call."""
from __future__ import annotations

import pytest
import torch
from torch.func import functional_call

from leakpro.attacks.gia_attacks.modular.core.interfaces import MODEL_MODES, Objective
from leakpro.tests.gia.contract.registry import build, cases

pytestmark = pytest.mark.parametrize(("cls", "factory"), cases(Objective))


def test_loss_is_a_scalar_differentiable_in_the_model_input(cls, factory) -> None:  # noqa: ANN001
    case = build(cls, factory)
    x = case.model_input.clone().requires_grad_(True)
    loss = case.component.loss(case.model, x, case.target)
    assert loss.dim() == 0 and torch.isfinite(loss)
    (grad,) = torch.autograd.grad(loss, x)
    assert grad.shape == x.shape and torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_module_call_and_functional_call_give_the_same_loss(cls, factory) -> None:  # noqa: ANN001
    case = build(cls, factory)
    state = {**dict(case.model.named_parameters()), **dict(case.model.named_buffers())}

    def model_fn(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return functional_call(case.model, state, args, kwargs)

    # Same RNG state for both calls, so stochastic layers (dropout) draw the same masks.
    torch.manual_seed(0)
    direct = case.component.loss(case.model, case.model_input, case.target)
    torch.manual_seed(0)
    functional = case.component.loss(model_fn, case.model_input, case.target)
    torch.testing.assert_close(functional, direct)


def test_prepare_model_sets_the_module_mode(cls, factory) -> None:  # noqa: ANN001
    case = build(cls, factory)
    for mode in (*MODEL_MODES, *reversed(MODEL_MODES)):
        case.component.prepare_model(case.model, mode)
        assert case.model.training is (mode == "train"), mode
