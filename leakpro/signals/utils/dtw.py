"""Dynamic Time Warping distance for multivariate time-series."""

import numpy as np


def mv_dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the Dynamic Time Warping distance between multivariate time-series x and y.

    This is the "dependent" DTW variant (DTW_D): the local cost between two time points is
    the squared Euclidean distance summed over all variables, and the distance is the sum of
    local costs along the optimal warping path. No square root is taken, and the warping path
    is unconstrained (no Sakoe-Chiba or Itakura bounding).

    Note the axis convention, which matches :func:`leakpro.signals.utils.msm.mv_msm_distance`:
    axis 0 is time and axis 1 indexes the variables. Univariate series may be passed as 1d
    arrays of shape (n,).

    Args:
    ----
        x: First time series of shape (n, num_variables).
        y: Second time series of shape (m, num_variables).

    Returns:
    -------
        The DTW distance between x and y.

    """
    # Flatten the variable axis so univariate (n,) and multivariate (n, num_variables) inputs
    # take the same path; the squared differences are summed over variables either way.
    x_2d = x.reshape(x.shape[0], -1)
    y_2d = y.reshape(y.shape[0], -1)
    if x_2d.shape[1] != y_2d.shape[1]:
        raise ValueError(
            f"x and y must have the same number of variables, got {x_2d.shape[1]} and {y_2d.shape[1]}."
        )

    n, m = x_2d.shape[0], y_2d.shape[0]
    local_cost = np.sum((x_2d[:, None, :] - y_2d[None, :, :]) ** 2, axis=-1)

    # Padded by one row/column so the (0, 0) boundary needs no special-casing in the main loop.
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    # Wavefront (anti-diagonal) traversal. cost[a, b] depends only on cost[a-1, b] and
    # cost[a, b-1] (both on anti-diagonal a+b-1) and cost[a-1, b-1] (on a+b-2), so every cell
    # on an anti-diagonal is independent of the others and the whole diagonal resolves in one
    # vectorised step. That is O(n+m) numpy calls rather than O(n*m) Python iterations, which
    # matters for long forecasting horizons.
    for diagonal in range(2, n + m + 1):
        a_first = max(1, diagonal - m)
        a_last = min(n, diagonal - 1)
        if a_first > a_last:
            continue
        a = np.arange(a_first, a_last + 1)
        b = diagonal - a
        cost[a, b] = local_cost[a - 1, b - 1] + np.minimum(
            np.minimum(cost[a - 1, b], cost[a, b - 1]), cost[a - 1, b - 1]
        )

    return float(cost[n, m])
