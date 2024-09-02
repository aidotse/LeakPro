"""Utility functions for attacks."""

import numpy as np
from torch import exp, from_numpy, max, sum


def softmax_logits(logits: np.ndarray, temp:float=1.0, dimension:int=-1) -> np.ndarray:
    """Rescale logits to (0, 1).

    Args:
    ----
        logits ( len(dataset) x ... x nb_classes ): Logits to be rescaled.
        temp (float): Temperature for softmax.
        dimension (int): Dimension to apply softmax.

    """
    logits = from_numpy(logits) / temp
    logits = logits - max(logits, dim=dimension, keepdim=True).values
    logits = exp(logits)
    logits = logits/sum(logits, dim=dimension, keepdim=True)
    return logits.numpy()

