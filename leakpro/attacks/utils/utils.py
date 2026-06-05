#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions for attacks."""

import numpy as np
from torch import cat, exp, from_numpy, max, sigmoid, sum

from leakpro.signals.functional import mse


def gaussian_residual_probability(predictions: np.ndarray, targets: np.ndarray, sigma2: float) -> np.ndarray:
    """Compute a Gaussian likelihood surrogate exp(-MSE/(2*sigma2)) for continuous-output models.

    Plays the role the softmax probability of the true class plays for classifiers:
    a per-point "probability of the true output given the model" for regression and
    forecasting targets. The Gaussian normalizing constant is intentionally dropped;
    when sigma2 is shared across the target and all reference models it cancels in
    likelihood ratios such as RMIA's p(x|theta)/p(x). Values lie in (0, 1].

    Args:
    ----
        predictions ( len(dataset) x ... ): Model predictions (e.g. forecast horizons).
        targets ( len(dataset) x ... ): Ground-truth values, same shape as predictions.
        sigma2 (float): Residual variance; acts as the regression analogue of the
            softmax temperature. Must be positive.

    Returns:
    -------
        Per-point probabilities of shape ( len(dataset), ).

    """
    if sigma2 <= 0:
        raise ValueError(f"sigma2 must be positive, got {sigma2}")
    return np.exp(-mse(predictions, targets) / (2.0 * sigma2))


def softmax_logits(logits: np.ndarray, temp:float=1.0, dimension:int=-1) -> np.ndarray:
    """Rescale logits to (0, 1).

    Args:
    ----
        logits ( len(dataset) x ... x nb_classes ): Logits to be rescaled.
        temp (float): Temperature for softmax.
        dimension (int): Dimension to apply softmax.

    """
    # If the number of classes is 1, apply sigmoid to return a matrix of [1 - p, p]
    if logits.shape[dimension] == 1:
        logits = from_numpy(logits)
        positive_confidence = sigmoid(logits / temp)  # Apply sigmoid to get the probability of class 1
        zero_confidence = 1 - positive_confidence     # Probability of class 0
        confidences = cat([zero_confidence, positive_confidence], dim=dimension)  # Stack both confidences
        return confidences.numpy()

    logits = from_numpy(logits) / temp
    logits = logits - max(logits, dim=dimension, keepdim=True).values
    logits = exp(logits)
    logits = logits/sum(logits, dim=dimension, keepdim=True)
    return logits.numpy()

