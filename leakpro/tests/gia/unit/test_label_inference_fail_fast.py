#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Label inference fails fast instead of silently returning a different quantity.

- DimitrovInference reconstructs the client weights from a payload that leaves out frozen parameters.
- DimitrovInference refuses a model without a known activation unless the approximation is asked for.
- IDLGInference rejects batches with more samples than classes (it returns one distinct label per sample).
"""
from __future__ import annotations

import logging

import pytest
import torch
from torch import nn

from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.components.target_inference import DimitrovInference, IDLGInference
from leakpro.attacks.gia_attacks.modular.core.protocol import FedAvg, FedSGD
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, Context, TargetInfo
from leakpro.attacks.gia_attacks.modular.core.task import Task

CPU = torch.device("cpu")


def _ctx(model: nn.Module, gradients: list[torch.Tensor], n: int, protocol=None) -> Context:  # noqa: ANN001
    return Context(model=model, observation=ClientObservation(gradients=gradients), modality=None,
                   task=Task(CrossEntropy(), protocol if protocol is not None else FedSGD()),
                   input_shape=(n, 3, 2, 2), device=CPU)


def test_client_params_skip_frozen_parameters() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 6), nn.ReLU(), nn.Linear(6, 4))
    model[1].requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    payload = [torch.full_like(p, 0.5) for p in trainable]
    server = {k: v.data.clone() for k, v in model.named_parameters()}

    client = DimitrovInference()._compute_client_params(payload, model, server, lr=0.1)

    torch.testing.assert_close(client["1.weight"], server["1.weight"])
    torch.testing.assert_close(client["3.weight"], server["3.weight"] - 0.05)
    torch.testing.assert_close(client["3.bias"], server["3.bias"] - 0.05)


def test_client_params_raise_on_a_payload_of_the_wrong_length() -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 4))
    server = {k: v.data.clone() for k, v in model.named_parameters()}
    with pytest.raises(ValueError, match="payload"):
        DimitrovInference()._compute_client_params([torch.zeros(4, 12)], model, server, lr=0.1)


def _no_activation_case() -> Context:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Tanh(), nn.Linear(12, 4))
    x, y = torch.randn(2, 3, 2, 2), torch.tensor([0, 1])
    grads = list(torch.autograd.grad(nn.functional.cross_entropy(model(x), y), list(model.parameters())))
    return _ctx(model, grads, 2)


def test_model_without_known_activation_raises_by_default() -> None:
    with pytest.raises(ValueError, match="no activation of type"):
        DimitrovInference(num_dummy_samples=4).infer(_no_activation_case())


def test_approximation_on_request_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        out = DimitrovInference(num_dummy_samples=4, strict_activation=False).infer(_no_activation_case())
    assert out.value.shape == (2,)
    assert any("activation" in r.getMessage() for r in caplog.records)


def test_idlg_rejects_more_samples_than_classes() -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 4))
    grads = [torch.zeros(4, 12), torch.zeros(4)]
    with pytest.raises(ValueError, match="classes"):
        IDLGInference().infer(_ctx(model, grads, 5))


def test_idlg_recovers_distinct_labels_of_a_batch() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 6))
    x, y = torch.rand(3, 3, 2, 2), torch.tensor([4, 0, 2])
    grads = list(torch.autograd.grad(nn.functional.cross_entropy(model(x), y), list(model.parameters())))
    labels = IDLGInference().infer(_ctx(model, grads, 3)).value
    assert sorted(labels.tolist()) == [0, 2, 4]


def test_fedavg_update_with_frozen_layer_infers_labels() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 4))
    model[1].requires_grad_(False)
    x, y = torch.rand(4, 3, 2, 2), torch.tensor([1, 1, 3, 0])
    protocol = FedAvg(local_epochs=1, lr=0.01)
    out = protocol.observe(model, CrossEntropy(), x, TargetInfo(value=y), create_graph=False, has_epoch_prefix=False)
    labels = DimitrovInference(num_dummy_samples=50).infer(_ctx(model, out.gradients, 4, protocol)).value
    assert labels.shape == (4,)
