#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""MetaOptimizer contract: a step moves trainable parameters against the gradient and leaves frozen ones alone."""
from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from leakpro.attacks.gia_attacks.modular.core.meta import MetaOptimizer
from leakpro.tests.gia.contract.registry import build, cases

pytestmark = pytest.mark.parametrize(("cls", "factory"), cases(MetaOptimizer))


def _params() -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        w=torch.tensor([1.0, -2.0, 3.0], requires_grad=True),
        frozen=torch.tensor([5.0, 6.0]),
    )


def test_step_moves_parameters_against_the_gradient_and_leaves_frozen_ones(cls, factory) -> None:  # noqa: ANN001
    case = build(cls, factory)
    params = _params()
    coefficients = torch.tensor([2.0, -1.0, 0.5])
    loss = (coefficients * params["w"]).sum() + params["frozen"].sum()   # d loss / d w = coefficients
    before = {k: v.detach().clone() for k, v in params.items()}

    updated = case.component.step(loss, params, create_graph=False)

    assert list(updated) == list(params)
    # Each coordinate moves opposite to the sign of its gradient.
    delta = updated["w"].detach() - before["w"]
    assert torch.all(torch.sign(delta) == -torch.sign(coefficients))
    assert torch.all(delta.abs() > 0)
    # Frozen entries are returned untouched; the inputs are not modified in place.
    assert updated["frozen"] is params["frozen"]
    torch.testing.assert_close(updated["frozen"], before["frozen"], rtol=0, atol=0)
    torch.testing.assert_close(params["w"].detach(), before["w"], rtol=0, atol=0)


def test_create_graph_keeps_the_update_differentiable(cls, factory) -> None:  # noqa: ANN001
    case = build(cls, factory)
    x = torch.tensor([0.7], requires_grad=True)
    params = OrderedDict(w=torch.tensor([1.0], requires_grad=True))
    loss = ((params["w"] * x) ** 2).sum()
    updated = case.component.step(loss, params, create_graph=True)
    assert updated["w"].requires_grad
    # The updated parameter depends on x through the gradient.
    assert torch.autograd.grad(updated["w"].sum(), x, allow_unused=True)[0] is not None
