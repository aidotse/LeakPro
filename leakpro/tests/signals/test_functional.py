#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for functional signals and the membership-direction registry.

Functional signals operate on cached logits/predictions (no model re-query), which is why
LiRA-style attacks use them. Here we cover the classification ``loss`` signal added for MS-LiRA
parity and the ``SIGNAL_MEMBERSHIP_DIRECTION`` orientation table.
"""

import numpy as np
import pytest
from scipy.special import logsumexp

from leakpro.signals import functional


class TestFunctionalLoss:
    """``functional.loss`` should equal per-point cross-entropy from logits + integer labels."""

    def test_multiclass_matches_cross_entropy(self) -> None:
        """CE(z, y) = logsumexp(z) - z[y] for multi-class logits."""
        logits = np.array([[2.0, 1.0, 0.1], [0.0, 1.0, 2.0], [-1.0, 0.5, 0.5]])
        labels = np.array([0, 2, 1], dtype=np.int64)

        expected = logsumexp(logits, axis=1) - logits[np.arange(3), labels]
        np.testing.assert_allclose(functional.loss(logits, labels), expected, rtol=1e-6)

    def test_binary_single_logit_head(self) -> None:
        """A single-logit head is treated as binary: CE = softplus(-z) for y=1, softplus(z) for y=0."""
        logits = np.array([[2.0], [-1.5], [0.0]])
        labels = np.array([1, 0, 1], dtype=np.int64)

        z = logits.ravel()
        # -log sigmoid(z) for y=1; -log(1-sigmoid(z)) for y=0
        expected = np.where(labels == 1, np.logaddexp(0, -z), np.logaddexp(0, z))
        np.testing.assert_allclose(functional.loss(logits, labels), expected, rtol=1e-6)

    def test_lower_loss_for_confident_correct_prediction(self) -> None:
        """A confident correct prediction should have lower loss than an unconfident one."""
        confident = functional.loss(np.array([[10.0, 0.0]]), np.array([0], dtype=np.int64))
        unconfident = functional.loss(np.array([[0.1, 0.0]]), np.array([0], dtype=np.int64))
        assert confident[0] < unconfident[0]

    def test_requires_int64_labels(self) -> None:
        """Labels must be int64 (used for fancy indexing), mirroring rescaled_logits."""
        with pytest.raises(AssertionError):
            functional.loss(np.array([[1.0, 0.0]]), np.array([0], dtype=np.float64))


class TestSignalMembershipDirection:
    """The orientation table encodes whether higher (+1) or lower (-1) means membership."""

    def test_confidence_signals_are_higher_is_member(self) -> None:
        """Confidence-style signals: a higher value indicates membership (+1)."""
        assert functional.SIGNAL_MEMBERSHIP_DIRECTION["rescaled_logits"] == +1
        assert functional.SIGNAL_MEMBERSHIP_DIRECTION["logits"] == +1

    def test_error_and_distance_signals_are_lower_is_member(self) -> None:
        """Error/distance signals: a lower value indicates membership (-1)."""
        for name in ["loss", "mse", "mae", "smape", "rescaled_smape",
                     "seasonality", "trend", "ts2vec", "dtw", "msm"]:
            assert functional.SIGNAL_MEMBERSHIP_DIRECTION[name] == -1, name

    def test_every_direction_is_plus_or_minus_one(self) -> None:
        """Directions are strictly +1 or -1 (used as a multiplicative sign)."""
        assert set(functional.SIGNAL_MEMBERSHIP_DIRECTION.values()) <= {+1, -1}
