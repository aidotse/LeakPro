#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for attack utility functions in leakpro.attacks.utils.utils."""

import numpy as np
import pytest

from leakpro.attacks.utils.utils import gaussian_residual_probability, softmax_logits


def test_gaussian_residual_probability_should_be_one_when_prediction_is_perfect() -> None:
    predictions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    targets = predictions.copy()

    probabilities = gaussian_residual_probability(predictions, targets, sigma2=1.0)

    assert probabilities.shape == (2,)
    assert np.allclose(probabilities, 1.0)


def test_gaussian_residual_probability_should_match_closed_form_when_residual_is_known() -> None:
    # Horizon of 4 with a constant residual of 2.0 -> MSE = 4.0 per point
    predictions = np.zeros((3, 4))
    targets = np.full((3, 4), 2.0)
    sigma2 = 0.5

    probabilities = gaussian_residual_probability(predictions, targets, sigma2)

    expected = np.exp(-4.0 / (2.0 * sigma2))
    assert np.allclose(probabilities, expected)


def test_gaussian_residual_probability_should_decrease_when_residual_grows() -> None:
    targets = np.zeros((3, 5))
    predictions = np.stack([np.full(5, 0.1), np.full(5, 1.0), np.full(5, 10.0)])

    probabilities = gaussian_residual_probability(predictions, targets, sigma2=1.0)

    assert probabilities[0] > probabilities[1] > probabilities[2]
    assert np.all(probabilities > 0.0)
    assert np.all(probabilities <= 1.0)


def test_gaussian_residual_probability_should_average_over_all_output_dims_when_multivariate() -> None:
    # Multivariate forecast (n, horizon, variables): MSE averages over both trailing dims
    rng = np.random.default_rng(0)
    predictions = rng.normal(size=(6, 4, 3))
    targets = rng.normal(size=(6, 4, 3))
    sigma2 = 2.0

    probabilities = gaussian_residual_probability(predictions, targets, sigma2)

    expected = np.exp(-np.mean((predictions - targets) ** 2, axis=(1, 2)) / (2.0 * sigma2))
    assert probabilities.shape == (6,)
    assert np.allclose(probabilities, expected)


def test_gaussian_residual_probability_should_raise_when_sigma2_is_not_positive() -> None:
    predictions = np.zeros((2, 3))
    targets = np.ones((2, 3))

    with pytest.raises(ValueError, match="sigma2 must be positive"):
        gaussian_residual_probability(predictions, targets, sigma2=0.0)
    with pytest.raises(ValueError, match="sigma2 must be positive"):
        gaussian_residual_probability(predictions, targets, sigma2=-1.0)


def test_gaussian_residual_probability_should_raise_when_shapes_differ() -> None:
    predictions = np.zeros((2, 3))
    targets = np.zeros((2, 4))

    with pytest.raises(AssertionError):
        gaussian_residual_probability(predictions, targets, sigma2=1.0)


def test_softmax_logits_should_sum_to_one_per_point() -> None:
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(5, 10))

    probabilities = softmax_logits(logits, temp=2.0)

    assert probabilities.shape == (5, 10)
    assert np.allclose(np.sum(probabilities, axis=1), 1.0)
