#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for L2GradientLoss and CosineGradientLoss.

The driver now pre-computes dummy_obs and passes it to compute() inside a LossInputs struct.
Tests supply all fields directly — no model, no run_attack, no driver needed.
"""
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F


from leakpro.attacks.gia_attacks.modular.components.losses import CosineGradientLoss, L2GradientLoss
from leakpro.attacks.gia_attacks.modular.core.loss_inputs import LossInputs
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, TargetInfo

DEVICE = torch.device("cpu")
_DUMMY_RECON = torch.zeros(1, 3, 4, 4)
_DUMMY_TARGET = TargetInfo(value=torch.tensor([0]))


def _ctx_with_device() -> MagicMock:
    """Minimal Context mock that only supplies device (needed by L2GradientLoss)."""
    ctx = MagicMock()
    ctx.device = DEVICE
    return ctx


def _inputs(
    recon=None,
    target=None,
    dummy_obs=None,
    target_obs=None,
    ctx=None,
) -> LossInputs:
    """Build a LossInputs with test-friendly defaults for fields not under test."""
    return LossInputs(
        recon=recon if recon is not None else _DUMMY_RECON,
        target=target if target is not None else _DUMMY_TARGET,
        target_obs=target_obs if target_obs is not None else ClientObservation(gradients=[]),
        ctx=ctx if ctx is not None else _ctx_with_device(),
        dummy_obs=dummy_obs,
    )


# ── L2GradientLoss ────────────────────────────────────────────────────────────

class TestL2GradientLoss:
    def _compute(self, dummy_grads: list, obs_grads: list) -> torch.Tensor:
        return L2GradientLoss().compute(
            _inputs(
                dummy_obs=ClientObservation(gradients=dummy_grads),
                target_obs=ClientObservation(gradients=obs_grads),
                ctx=_ctx_with_device(),
            )
        )

    def test_identical_gradients_give_zero_loss(self):
        """When dummy == observed gradients, L2GradientLoss must be exactly 0."""
        grads = [torch.tensor([1.0, 2.0, 3.0])]
        result = self._compute(grads, [g.clone() for g in grads])
        assert result.item() == pytest.approx(0.0)

    def test_known_difference_gives_correct_summed_mse(self):
        """L2GradientLoss == sum_layers MSE(dummy, observed, reduction='sum').

        dummy = [2, 3], obs = [0, 0]
        => (2-0)^2 + (3-0)^2 = 4 + 9 = 13.0
        """
        dummy_grads = [torch.tensor([2.0, 3.0])]
        obs_grads   = [torch.tensor([0.0, 0.0])]
        expected = F.mse_loss(dummy_grads[0], obs_grads[0], reduction="sum").item()
        assert expected == pytest.approx(13.0)
        result = self._compute(dummy_grads, obs_grads)
        assert result.item() == pytest.approx(expected)


# ── CosineGradientLoss ────────────────────────────────────────────────────────

class TestCosineGradientLoss:
    def _compute(self, dummy_grads: list, obs_grads: list) -> torch.Tensor:
        return CosineGradientLoss().compute(
            _inputs(
                dummy_obs=ClientObservation(gradients=dummy_grads),
                target_obs=ClientObservation(gradients=obs_grads),
            )
        )

    def test_identical_gradients_give_zero_loss(self):
        """Cosine similarity of identical vectors is 1 => distance is 0."""
        grads = [torch.tensor([1.0, 2.0, 3.0])]
        result = self._compute(grads, [g.clone() for g in grads])
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_gradients_give_loss_of_one(self):
        """Cosine similarity of orthogonal vectors is 0 => distance is 1.

        dummy = [1, 0], obs = [0, 1]  =>  dot = 0  =>  cos_sim = 0  =>  loss = 1 - 0 = 1.
        """
        dummy_grads = [torch.tensor([1.0, 0.0])]
        obs_grads   = [torch.tensor([0.0, 1.0])]
        result = self._compute(dummy_grads, obs_grads)
        assert result.item() == pytest.approx(1.0, abs=1e-6)

    def test_global_flatten_matches_multi_layer(self):
        """All layers are concatenated before cosine distance — layer count doesn't weight."""
        # Two-layer gradient: same total direction, just split across two tensors.
        dummy_grads = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0])]
        obs_grads   = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0])]
        result = self._compute(dummy_grads, obs_grads)
        assert result.item() == pytest.approx(0.0, abs=1e-6)


# ── Image priors: space option and L2 reduction ─────────────────────────────────────────────────

from leakpro.attacks.gia_attacks.modular.components.losses import L2ImageNormLoss, TVLoss  # noqa: E402


def _prior_inputs(x, z):
    return LossInputs(recon=x, target=None, target_obs=None, ctx=None, model_input=z)


def test_tv_in_model_input_space_uses_the_model_input_tensor() -> None:
    x, z = torch.rand(2, 3, 4, 4), torch.randn(2, 3, 4, 4)
    torch.testing.assert_close(TVLoss(space="model_input").compute(_prior_inputs(x, z)), TVLoss().compute(_prior_inputs(z, None)))


def test_model_input_space_without_model_input_raises() -> None:
    with pytest.raises(ValueError, match="model_input"):
        TVLoss(space="model_input").compute(_prior_inputs(torch.rand(1, 3, 4, 4), None))


def test_l2_per_image_sum_matches_reference_formula() -> None:
    z = torch.randn(2, 3, 4, 4)
    expected = z.reshape(2, -1).norm(dim=1).sum() / 16
    got = L2ImageNormLoss(space="model_input", reduction="per_image_sum").compute(_prior_inputs(None, z))
    torch.testing.assert_close(got, expected)
