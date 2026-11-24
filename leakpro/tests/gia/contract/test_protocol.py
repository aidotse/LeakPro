#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""ClientProtocol contract: no side effects on the model's weights or mode, determinism, detached payloads."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from leakpro.attacks.gia_attacks.modular.core.interfaces import ClientProtocol
from leakpro.tests.gia import helpers
from leakpro.tests.gia.contract.conftest import client_target
from leakpro.tests.gia.contract.registry import build, cases

pytestmark = pytest.mark.parametrize(("cls", "factory"), cases(ClientProtocol))


def _observe(case, model: nn.Module, *, create_graph: bool = False):  # noqa: ANN001, ANN202
    model_input = helpers.image_modality().to_model_input(helpers.client_batch()[0])
    return case.component.observe(model, case.objective, model_input, client_target(),
                                  create_graph=create_graph, has_epoch_prefix=False)


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_observe_leaves_the_weights_and_mode_unchanged(cls, factory, mode) -> None:  # noqa: ANN001
    case = build(cls, factory)
    model = helpers.tiny_bn_cnn().train(mode == "train")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    _observe(case, model)
    assert model.training is (mode == "train")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_observe_keeps_a_frozen_layer_frozen_and_pays_out_trainable_parameters_only(cls, factory, mode) -> None:  # noqa: ANN001
    case = build(cls, factory)
    model = helpers.tiny_bn_cnn().train(mode == "train")
    model[0].weight.requires_grad_(False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    out = _observe(case, model)
    assert not model[0].weight.requires_grad
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0, msg=name)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert [g.shape for g in out.gradients] == [p.shape for p in trainable]


def test_observe_is_deterministic(cls, factory) -> None:  # noqa: ANN001
    case = build(cls, factory)
    model = helpers.tiny_bn_cnn().train()
    first, second = _observe(case, model), _observe(case, model)
    for a, b in zip(first.gradients, second.gradients):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(first.first_loss, second.first_loss, rtol=0, atol=0)


def test_without_create_graph_the_payload_is_detached(cls, factory) -> None:  # noqa: ANN001
    case = build(cls, factory)
    out = _observe(case, helpers.tiny_bn_cnn().train())
    assert out.gradients
    assert all(g.grad_fn is None and not g.requires_grad for g in out.gradients)
    assert out.first_loss.grad_fn is None


def test_with_create_graph_the_payload_is_differentiable_in_the_input(cls, factory) -> None:  # noqa: ANN001
    case = build(cls, factory)
    model_input = helpers.image_modality().to_model_input(helpers.client_batch()[0]).requires_grad_(True)
    out = case.component.observe(helpers.tiny_bn_cnn().train(), case.objective, model_input, client_target(),
                                 create_graph=True, has_epoch_prefix=False)
    (grad,) = torch.autograd.grad(sum(g.pow(2).sum() for g in out.gradients), model_input)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
