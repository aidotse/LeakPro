#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions for attacks."""

import numpy as np
from torch import cat, exp, from_numpy, max, sigmoid, sum

from leakpro.signals.functional import mae, mse


def gaussian_residual_probability(predictions: np.ndarray, targets: np.ndarray, sigma2: float) -> np.ndarray:
    """Compute a Gaussian likelihood surrogate exp(-MSE/(2*sigma2)) for continuous-output models.

    Plays the role the softmax probability of the true class plays for classifiers:
    a per-point "probability of the true output given the model" for regression and
    forecasting targets. The Gaussian noise model is implied by training with MSELoss.
    The normalizing constant is intentionally dropped; when sigma2 is shared across the
    target and all reference models it cancels in likelihood ratios such as RMIA's
    p(x|theta)/p(x). Values lie in (0, 1].

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


def laplace_residual_probability(predictions: np.ndarray, targets: np.ndarray, scale: float) -> np.ndarray:
    """Compute a Laplace likelihood surrogate exp(-MAE/scale) for L1-trained models.

    The Laplace noise model is implied by training with L1Loss, just as the Gaussian
    is implied by MSELoss; using the Gaussian surrogate for an L1-trained model would
    score it under a mismatched noise model. The normalizing constant is dropped for
    the same reason as in gaussian_residual_probability. Values lie in (0, 1].

    Args:
    ----
        predictions ( len(dataset) x ... ): Model predictions (e.g. forecast horizons).
        targets ( len(dataset) x ... ): Ground-truth values, same shape as predictions.
        scale (float): Laplace scale b (MLE: mean absolute residual). Must be positive.

    Returns:
    -------
        Per-point probabilities of shape ( len(dataset), ).

    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    return np.exp(-mae(predictions, targets) / scale)


def huber_energy(predictions: np.ndarray, targets: np.ndarray, delta: float) -> np.ndarray:
    """Compute the per-point mean Huber energy of the residuals.

    Elementwise rho_delta(r) = 0.5*r^2 for |r| <= delta, delta*(|r| - 0.5*delta) otherwise
    (torch.nn.HuberLoss convention; SmoothL1Loss(beta) is rho_beta scaled by 1/beta, a
    constant factor absorbed by the likelihood scale), averaged over all output dims.

    Args:
    ----
        predictions ( len(dataset) x ... ): Model predictions (e.g. forecast horizons).
        targets ( len(dataset) x ... ): Ground-truth values, same shape as predictions.
        delta (float): Threshold between the quadratic core and the linear tails. Must be positive.

    Returns:
    -------
        Per-point energies of shape ( len(dataset), ).

    """
    assert predictions.shape == targets.shape
    if delta <= 0:
        raise ValueError(f"delta must be positive, got {delta}")
    residuals = np.abs(predictions - targets)
    elementwise = np.where(residuals <= delta, 0.5 * residuals**2, delta * (residuals - 0.5 * delta))
    return np.mean(elementwise, axis=tuple(range(1, predictions.ndim)))


def huber_residual_probability(predictions: np.ndarray, targets: np.ndarray, scale: float, delta: float) -> np.ndarray:
    """Compute a Huber likelihood surrogate exp(-huber_energy/scale) for Huber/SmoothL1-trained models.

    The Huber density (Gaussian core, Laplace tails) is the noise model implied by
    training with HuberLoss/SmoothL1Loss. For residuals within delta the exponent is
    0.5*r^2/scale, so scale plays exactly the sigma^2 role of the Gaussian surrogate.
    The normalizing constant is dropped as in the other surrogates. Values lie in (0, 1].

    Args:
    ----
        predictions ( len(dataset) x ... ): Model predictions (e.g. forecast horizons).
        targets ( len(dataset) x ... ): Ground-truth values, same shape as predictions.
        scale (float): Variance of the quadratic core. Must be positive.
        delta (float): Huber threshold (HuberLoss.delta or SmoothL1Loss.beta). Must be positive.

    Returns:
    -------
        Per-point probabilities of shape ( len(dataset), ).

    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    return np.exp(-huber_energy(predictions, targets, delta) / scale)


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

