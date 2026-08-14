"""Move-Split-Merge distance for multivariate time-series."""

import numpy as np


def mv_msm_cost(a: np.ndarray, b: np.ndarray, c: np.ndarray, cost: float = 1.0) -> np.ndarray:
    """Compute the MSM cost function element-wise."""
    mask = (b <= a) & (a <= c) | (c <= a) & (a <= b)
    return np.where(mask, cost, cost + np.minimum(np.abs(a - b), np.abs(a - c)))


def _univariate_msm_distance(x: np.ndarray, y: np.ndarray, cost: float) -> float:
    """Compute the standard (univariate) MSM distance between 1d series x and y.

    Args:
    ----
        x: First time series of shape (n,).
        y: Second time series of shape (m,).
        cost: The MSM constant c charged for every split and merge operation.

    Returns:
    -------
        The MSM distance between x and y.

    """
    n, m = x.shape[0], y.shape[0]

    # Every transition cost is independent of the alignment, so all three cost matrices can be
    # built up front with numpy instead of inside the dynamic-programming loop. Rows/columns
    # that no move can reach are left at infinity so the recurrence needs no boundary cases.
    match = np.abs(x[:, None] - y[None, :])
    split = np.full((n, m), np.inf)
    split[1:, :] = mv_msm_cost(x[1:, None], x[:-1, None], y[None, :], cost)
    merge = np.full((n, m), np.inf)
    merge[:, 1:] = mv_msm_cost(y[None, 1:], x[:, None], y[None, :-1], cost)

    accumulated = np.full((n, m), np.inf)
    accumulated[0, 0] = match[0, 0]

    # Wavefront (anti-diagonal) traversal, as in mv_dtw_distance: cell (i, j) depends only on
    # (i-1, j-1), (i-1, j) and (i, j-1), so every cell on an anti-diagonal is independent of the
    # others and resolves in one vectorised step. O(n+m) numpy calls rather than O(n*m) Python
    # iterations, which matters for long forecasting horizons.
    for diagonal in range(1, n + m - 1):
        i = np.arange(max(0, diagonal - m + 1), min(n - 1, diagonal) + 1)
        j = diagonal - i
        i_prev = np.maximum(i - 1, 0)
        j_prev = np.maximum(j - 1, 0)

        # A match consumes one point from each series, so it only exists off both borders.
        cost_match = np.where(
            (i > 0) & (j > 0), accumulated[i_prev, j_prev] + match[i, j], np.inf
        )
        # The infinite border rows/columns of split/merge rule out the other two moves there.
        cost_split = accumulated[i_prev, j] + split[i, j]
        cost_merge = accumulated[i, j_prev] + merge[i, j]

        accumulated[i, j] = np.minimum(np.minimum(cost_match, cost_split), cost_merge)

    return float(accumulated[-1, -1])


def mv_msm_distance(x: np.ndarray, y: np.ndarray, cost: float = 1.0) -> float:
    """Compute the Move-Split-Merge distance between multivariate time-series x and y.

    This is the "independent" multivariate variant (MSM_I): the standard univariate MSM distance
    (Stefan, Athitsos and Das, 2013) is computed for each variable separately, each with its own
    alignment path, and the per-variable distances are summed. Univariate input therefore gives
    exactly the standard MSM distance.

    The alternative published variant is "dependent" MSM (MSM_D), which treats each timestamp as
    one vector-valued point sharing a single alignment path. That is deliberately not what this
    function does. Note that :func:`leakpro.signals.utils.dtw.mv_dtw_distance` is the dependent
    variant of DTW, so the two distances do not treat variables the same way.

    Args:
    ----
        x: First time series of shape (n, num_variables).
        y: Second time series of shape (m, num_variables).
        cost: The MSM constant c charged for every split and merge operation.

    Returns:
    -------
        The MSM distance between x and y.

    """
    # Univariate series may arrive as (n,) rather than (n, 1); both take the same path.
    x_2d = x.reshape(x.shape[0], -1)
    y_2d = y.reshape(y.shape[0], -1)
    if x_2d.shape[1] != y_2d.shape[1]:
        raise ValueError(
            f"x and y must have the same number of variables, got {x_2d.shape[1]} and {y_2d.shape[1]}."
        )

    return float(sum(
        _univariate_msm_distance(x_2d[:, variable], y_2d[:, variable], cost)
        for variable in range(x_2d.shape[1])
    ))
