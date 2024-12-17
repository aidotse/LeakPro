"""Utility functions for attacks."""

import numpy as np
from torch import cat, exp, from_numpy, max, sigmoid, sum


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

