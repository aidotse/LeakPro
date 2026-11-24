#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Objectives: the training loss a client minimises, given a model call and a target."""
import pytest
import torch
import torch.nn.functional as F

from leakpro.attacks.gia_attacks.modular.components.objectives import (
    BinaryCrossEntropy,
    CrossEntropy,
    MeanSquaredError,
)
from leakpro.attacks.gia_attacks.modular.core.protocol import FedAvg, FedSGD
from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo

_MODEL_OUT = torch.tensor([[1.0, 2.0, 3.0]])
_HARD_LABEL = torch.tensor([1])
_SOFT_LOGITS = torch.tensor([[0.5, 1.0, -0.5]])


def _identity(x):
    return x


class TestCrossEntropy:
    def test_hard_label_matches_cross_entropy(self):
        expected = F.cross_entropy(_MODEL_OUT, _HARD_LABEL)
        result = CrossEntropy().loss(_identity, _MODEL_OUT, TargetInfo(value=_HARD_LABEL))
        assert result.item() == pytest.approx(expected.item(), rel=1e-6)

    def test_soft_label_applies_softmax_before_cross_entropy(self):
        expected = F.cross_entropy(_MODEL_OUT, F.softmax(_SOFT_LOGITS, dim=-1))
        result = CrossEntropy().loss(_identity, _MODEL_OUT, TargetInfo(soft=_SOFT_LOGITS))
        assert result.item() == pytest.approx(expected.item(), rel=1e-6)

    def test_sum_reduction_is_the_batch_size_times_the_mean(self):
        target = TargetInfo(value=_HARD_LABEL)
        mean = CrossEntropy().loss(_identity, _MODEL_OUT, target)
        total = CrossEntropy(reduction="sum").loss(_identity, _MODEL_OUT, target)
        assert total.item() == pytest.approx(_MODEL_OUT.shape[0] * mean.item(), rel=1e-6)

    def test_unknown_reduction_is_rejected(self):
        with pytest.raises(ValueError, match="reduction"):
            CrossEntropy(reduction="none")

    def test_soft_takes_priority_over_value(self):
        expected = F.cross_entropy(_MODEL_OUT, F.softmax(_SOFT_LOGITS, dim=-1))
        result = CrossEntropy().loss(_identity, _MODEL_OUT, TargetInfo(value=_HARD_LABEL, soft=_SOFT_LOGITS))
        assert result.item() == pytest.approx(expected.item(), rel=1e-6)

    def test_neither_field_set_raises(self):
        with pytest.raises(ValueError):
            CrossEntropy().loss(_identity, _MODEL_OUT, TargetInfo())


def test_binary_cross_entropy_is_bce_with_logits():
    out, y = torch.randn(6, 1), torch.randint(0, 2, (6,)).float()
    got = BinaryCrossEntropy().loss(_identity, out, TargetInfo(value=y))
    assert torch.allclose(got, F.binary_cross_entropy_with_logits(out.squeeze(-1), y))
    s = BinaryCrossEntropy(reduction="sum").loss(_identity, out, TargetInfo(value=y))
    assert torch.allclose(s, got * 6)


def test_binary_cross_entropy_soft_labels_are_logits():
    out, soft = torch.randn(4, 1), torch.randn(4, 1)
    got = BinaryCrossEntropy().loss(_identity, out, TargetInfo(soft=soft))
    assert torch.allclose(got, F.binary_cross_entropy_with_logits(out.squeeze(-1), torch.sigmoid(soft.squeeze(-1))))


def test_mean_squared_error_is_mse():
    out, y = torch.randn(5, 1), torch.randn(5)
    assert torch.allclose(MeanSquaredError().loss(_identity, out, TargetInfo(value=y)), F.mse_loss(out.squeeze(-1), y))


def test_scalar_objectives_reject_an_unknown_reduction():
    with pytest.raises(ValueError):
        BinaryCrossEntropy(reduction="max")
    with pytest.raises(ValueError):
        MeanSquaredError(reduction="max")


def test_classification_objectives_support_both_protocols():
    for objective in (CrossEntropy, BinaryCrossEntropy, MeanSquaredError):
        assert set(objective.supported_protocols) == {FedSGD, FedAvg}
