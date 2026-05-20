"""Function versions of the signal classes. These functions take in logits and targets in general.
   These functions are mainly used in combination with shadow modeles where logits are precomputed and cashed."""

import numpy as np
import torch

from joblib import Parallel, delayed
from torch import cuda
from ts2vec import TS2Vec
from sktime.distances import dtw_distance
from leakpro.signals.utils.msm import mv_msm_distance

def logits(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    _ = targets  # not used, present here for function consistency by convention. 
    return logits

def rescaled_logits(logits: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
    """Rescale the logits to a range of [0, 1].

    Args:
        logits (np.ndarray): The logits to be rescaled.
        true_label (np.ndarray): The true labels for the logits.

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

def mse(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    assert logits.shape == targets.shape
    return np.mean((logits - targets)**2, axis=tuple(range(1, logits.ndim)))

def mae(logits: np.ndarray, targets: np.ndarray) -> np.ndarray: 
    assert logits.shape == targets.shape
    return np.mean(np.abs(logits - targets), axis=tuple(range(1, logits.ndim)))

def smape(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    assert logits.shape == targets.shape
    numerator = np.abs(logits - targets)
    denominator = np.abs(logits) + np.abs(targets) + 1e-30
    fraction = numerator / denominator
    return np.mean(fraction, axis=tuple(range(1, logits.ndim)))

def rescaled_smape(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    assert logits.shape == targets.shape
    numerator = np.abs(logits - targets)
    denominator = np.abs(logits) + np.abs(targets) + 1e-30
    fraction = numerator / denominator
    smape_loss = np.mean(fraction, axis=tuple(range(1, logits.ndim)))
    return np.log(smape_loss / (1 - smape_loss + 1e-30))

def seasonality(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    assert logits.shape == targets.shape
    if logits.ndim == 2:  # expand dims if times series are univariate
        logits = np.expand_dims(logits, axis=2)
        targets = np.expand_dims(targets, axis=2)
    def get_seasonality_coefficients(Y: np.ndarray) -> np.ndarray:
        """Extract seasonality coefficients from a multivariate time series using 2D DFT."""
        Z = np.fft.fft(Y, axis=2)      # column-wise 1D-DFT over variables
        return np.fft.fft(Z, axis=1)   # row-wise 1D-DFT over horizon
    seasonality_pred = get_seasonality_coefficients(logits)
    seasonality_true = get_seasonality_coefficients(targets)
    return np.linalg.norm(seasonality_true - seasonality_pred, axis=(1, 2))

def trend(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    assert logits.shape == targets.shape
    if logits.ndim == 2:  # expand dims if times series are univariate
        logits = np.expand_dims(logits, axis=2)
        targets = np.expand_dims(targets, axis=2)
    def get_trend_coefficients(Y: np.ndarray, polynomial_degree: int = 4) -> np.ndarray:
        """Extract trend coefficients from a multivariate time series by fitting a polynomial to each variable."""
        horizon = Y.shape[1]
        t = np.arange(horizon) / horizon
        P = np.vander(t, polynomial_degree, increasing=True)    # Vandermonde matrix of specified degree  # noqa: N806
        return np.linalg.inv(P.T @ P) @ P.T @ Y                 # least squares solution to polynomial fit
    trend_pred = get_trend_coefficients(logits)
    trend_true = get_trend_coefficients(targets)
    return np.linalg.norm(trend_true - trend_pred, axis=(1, 2))

def ts2vec(logits: np.ndarray, targets: np.ndarray, batch_size: int = 256) -> np.ndarray:
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
    assert logits.shape == targets.shape
    distances = Parallel(n_jobs=-1)(
        delayed(dtw_distance)(logits[i], targets[i]) for i in range(len(logits))
    )
    return np.array(distances)

def msm(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    assert logits.shape == targets.shape 
    distances = Parallel(n_jobs=-1)(
        delayed(mv_msm_distance)(logits[i], targets[i]) for i in range(len(logits))
    )
    return np.array(distances)