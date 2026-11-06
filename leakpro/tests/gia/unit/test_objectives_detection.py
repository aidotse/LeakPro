#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Detection objectives: the training loss a client minimises, given a model call and a box target."""
import pytest
import torch

from leakpro.attacks.gia_attacks.modular.components.objectives import (
    CriterionDetection,
    HuggingFaceDetection,
    TorchvisionDetection,
)
from leakpro.attacks.gia_attacks.modular.core.protocol import FedAvg, FedSGD
from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo


def _identity(x):
    return x


def test_criterion_detection_applies_the_criterion_to_the_encoded_target():
    calls = []

    def criterion(out, encoded):
        calls.append(encoded)
        return torch.stack([out.sum(), encoded.sum()])

    objective = CriterionDetection(criterion, lambda target: target.value * 2.0, sum_output=True)
    x = torch.ones(2, 3)
    loss = objective.loss(_identity, x, TargetInfo(value=torch.tensor([1.0, 2.0])))
    assert loss.item() == pytest.approx(6.0 + 6.0)
    assert torch.equal(calls[0], torch.tensor([2.0, 4.0]))


def test_criterion_detection_prepare_runs_after_the_mode_is_set():
    seen = []
    model = torch.nn.Linear(2, 2)
    objective = CriterionDetection(lambda o, t: o.sum(), lambda t: t.value,
                                   prepare=lambda m, mode: seen.append((m.training, mode)))
    objective.prepare_model(model, "eval")
    assert seen == [(False, "eval")]


def test_detection_objectives_declare_both_protocols() -> None:
    assert TorchvisionDetection.supported_protocols == (FedSGD, FedAvg)
    assert HuggingFaceDetection.supported_protocols == (FedSGD, FedAvg)
    assert FedAvg in CriterionDetection.supported_protocols
