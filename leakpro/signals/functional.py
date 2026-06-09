"""Function versions of the signal classes.

These functions take in logits and targets in general, and are mainly used in
combination with shadow models where logits are precomputed and cached.
"""

import numpy as np
import torch
from joblib import Parallel, delayed
from sktime.distances import dtw_distance
from torch import cuda
from ts2vec import TS2Vec

from leakpro.signals.utils.msm import mv_msm_distance


def logits(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return the logits unchanged (identity signal).

    Args:
        logits (np.ndarray): The cached model logits.
        targets (np.ndarray): Unused; present for signal-function signature consistency.

    Returns:
        np.ndarray: The logits unchanged.

    """
    _ = targets  # not used, present here for function consistency by convention.
    return logits


def rescaled_logits(logits: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
    """Rescale the logits to a range of [0, 1].

    Args:
        logits (np.ndarray): The logits to be rescaled.
        true_labels (np.ndarray): The true labels for the logits.

    Returns:
        np.ndarray: The rescaled logits.

    """
    assert true_labels.dtype == np.int64
    if logits.shape[1] == 1:
        def sigmoid(z:np.ndarray) -> np.ndarray:
            return 1/(1 + np.exp(-z))
        positive_class_prob = sigmoid(logits).reshape(-1, 1)
        predictions = np.concatenate([1 - positive_class_prob, positive_class_prob], axis=1)
    else:
        predictions = logits - np.max(logits, axis=1, keepdims=True)
        predictions = np.exp(predictions)
        predictions = predictions / np.sum(predictions, axis=1, keepdims=True)
    count = predictions.shape[0]
    y_true = predictions[np.arange(count), true_labels]
    predictions[np.arange(count), true_labels] = 0
    y_wrong = np.sum(predictions, axis=1)
    return np.log(y_true+1e-45) - np.log(y_wrong+1e-45)


def loss(logits: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
    """Per-point cross-entropy loss computed from cached logits and integer class labels.

    Functional counterpart of the ``ModelLoss`` signal for the common classification case:
    the negative log-likelihood of the true class, ``-log softmax(logits)[y]``. Unlike
    ``ModelLoss`` (which re-runs the model's own ``loss_fn``), this works directly on cached
    logits, so it only matches models trained with cross-entropy. For regression/forecasting
    targets use ``mse``/``mae``/``smape`` instead — those are the model's loss.

    Args:
        logits (np.ndarray): The logits, shape (n_points, n_classes); a single-column array is
            treated as a binary logit head.
        true_labels (np.ndarray): Integer class labels, shape (n_points,).

    Returns:
        np.ndarray: Per-point cross-entropy loss, shape (n_points,).

    """
    assert true_labels.dtype == np.int64
    if logits.shape[1] == 1:
        def sigmoid(z:np.ndarray) -> np.ndarray:
            return 1/(1 + np.exp(-z))
        positive_class_prob = sigmoid(logits).reshape(-1, 1)
        predictions = np.concatenate([1 - positive_class_prob, positive_class_prob], axis=1)
    else:
        predictions = logits - np.max(logits, axis=1, keepdims=True)
        predictions = np.exp(predictions)
        predictions = predictions / np.sum(predictions, axis=1, keepdims=True)
    count = predictions.shape[0]
    p_true = predictions[np.arange(count), true_labels]
    return -np.log(p_true + 1e-45)


def mse(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-point mean squared error between logits and targets."""
    assert logits.shape == targets.shape
    return np.mean((logits - targets)**2, axis=tuple(range(1, logits.ndim)))


def mae(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-point mean absolute error between logits and targets."""
    assert logits.shape == targets.shape
    return np.mean(np.abs(logits - targets), axis=tuple(range(1, logits.ndim)))


def smape(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-point symmetric mean absolute percentage error between logits and targets."""
    assert logits.shape == targets.shape
    numerator = np.abs(logits - targets)
    denominator = np.abs(logits) + np.abs(targets) + 1e-30
    fraction = numerator / denominator
    return np.mean(fraction, axis=tuple(range(1, logits.ndim)))


def rescaled_smape(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-point SMAPE mapped through a logit transform to spread values over the real line."""
    assert logits.shape == targets.shape
    numerator = np.abs(logits - targets)
    denominator = np.abs(logits) + np.abs(targets) + 1e-30
    fraction = numerator / denominator
    smape_loss = np.mean(fraction, axis=tuple(range(1, logits.ndim)))
    return np.log(smape_loss / (1 - smape_loss + 1e-30))


def seasonality(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-point distance between the seasonality (2D-DFT) coefficients of logits and targets."""
    assert logits.shape == targets.shape
    if logits.ndim == 2:  # expand dims if times series are univariate
        logits = np.expand_dims(logits, axis=2)
        targets = np.expand_dims(targets, axis=2)
    def get_seasonality_coefficients(y: np.ndarray) -> np.ndarray:
        """Extract seasonality coefficients from a multivariate time series using 2D DFT."""
        z = np.fft.fft(y, axis=2)      # column-wise 1D-DFT over variables
        return np.fft.fft(z, axis=1)   # row-wise 1D-DFT over horizon
    seasonality_pred = get_seasonality_coefficients(logits)
    seasonality_true = get_seasonality_coefficients(targets)
    return np.linalg.norm(seasonality_true - seasonality_pred, axis=(1, 2))


def trend(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-point distance between the polynomial-trend coefficients of logits and targets."""
    assert logits.shape == targets.shape
    if logits.ndim == 2:  # expand dims if times series are univariate
        logits = np.expand_dims(logits, axis=2)
        targets = np.expand_dims(targets, axis=2)
    def get_trend_coefficients(y: np.ndarray, polynomial_degree: int = 4) -> np.ndarray:
        """Extract trend coefficients from a multivariate time series by fitting a polynomial to each variable."""
        horizon = y.shape[1]
        t = np.arange(horizon) / horizon
        vander = np.vander(t, polynomial_degree, increasing=True)  # Vandermonde matrix of specified degree
        return np.linalg.inv(vander.T @ vander) @ vander.T @ y     # least squares solution to polynomial fit
    trend_pred = get_trend_coefficients(logits)
    trend_true = get_trend_coefficients(targets)
    return np.linalg.norm(trend_true - trend_pred, axis=(1, 2))


def ts2vec(logits: np.ndarray, targets: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Per-point distance between TS2Vec embeddings of logits and targets."""
    assert logits.shape == targets.shape
    if logits.ndim == 2:  # expand dims if times series are univariate
        logits = np.expand_dims(logits, axis=2)
        targets = np.expand_dims(targets, axis=2)
    device = "cuda:0" if cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic = False
    ts2vec_model = TS2Vec(input_dims = targets.shape[-1], device = device, batch_size = batch_size)
    ts2vec_model.fit(targets)
    logits_encoded = ts2vec_model.encode(logits, encoding_window="full_series", batch_size=batch_size)
    targets_encoded = ts2vec_model.encode(targets, encoding_window="full_series", batch_size=batch_size)
    return np.linalg.norm(logits_encoded - targets_encoded, axis=1)


def dtw(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-point Dynamic Time Warping distance between logits and targets."""
    assert logits.shape == targets.shape
    distances = Parallel(n_jobs=-1)(
        delayed(dtw_distance)(logits[i], targets[i]) for i in range(len(logits))
    )
    return np.array(distances)


def msm(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-point Move-Split-Merge distance between logits and targets."""
    assert logits.shape == targets.shape
    distances = Parallel(n_jobs=-1)(
        delayed(mv_msm_distance)(logits[i], targets[i]) for i in range(len(logits))
    )
    return np.array(distances)


# Membership-inference orientation of each signal under the offline LiRA tail test.
#   +1: a HIGHER signal value indicates membership (e.g. confidence/rescaled logits).
#   -1: a LOWER  signal value indicates membership (errors/distances — members fit better).
# Used to orient signals so the one-sided Gaussian tail (norm.logcdf) is evaluated in the
# correct direction; see multi_signal_lira.py and the offline score in arXiv:2509.04169 Eq. (2).
SIGNAL_MEMBERSHIP_DIRECTION = {
    "logits": +1,
    "rescaled_logits": +1,
    "loss": -1,
    "mse": -1,
    "mae": -1,
    "smape": -1,
    "rescaled_smape": -1,
    "seasonality": -1,
    "trend": -1,
    "ts2vec": -1,
    "dtw": -1,
    "msm": -1,
}
