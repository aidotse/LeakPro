#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Attack-success metrics for the frontier's privacy axis.

The campaign optimizes TPR at FPR = 1% (a powered proxy with enough events per
evaluation for the surrogate to see signal); TPR at lower FPR levels is a
validation-time statistic, reported with Clopper-Pearson intervals on the
final frontier points only.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta


@dataclass(frozen=True)
class AttackScores:
    """Membership scores produced by one attack run (higher = more member-like)."""

    member_scores: np.ndarray
    nonmember_scores: np.ndarray

    def __post_init__(self) -> None:
        """Coerce score arrays to 1-D float arrays and validate them."""
        for field in ("member_scores", "nonmember_scores"):
            arr = np.asarray(getattr(self, field), dtype=float)
            if arr.ndim != 1 or arr.size == 0:
                raise ValueError(f"{field} must be a non-empty 1-D array, got shape {arr.shape}.")
            object.__setattr__(self, field, arr)


def tpr_at_fpr(scores: AttackScores, fpr: float) -> tuple[float, int, int]:
    """TPR at a fixed FPR, with the event counts the estimate rests on.

    The decision threshold is the smallest score whose empirical FPR does not
    exceed ``fpr`` (conservative under ties).

    Returns:
        (tpr, n_true_positives, n_members) — the counts let callers judge the
        estimate's resolution before believing it.

    """
    if not 0.0 < fpr < 1.0:
        raise ValueError(f"fpr must be in (0, 1), got {fpr}.")
    nonmember = np.sort(scores.nonmember_scores)[::-1]
    # Highest threshold admitting at most floor(fpr * n) nonmembers.
    n_admit = int(np.floor(fpr * nonmember.size))
    if n_admit == 0:
        threshold = np.inf if nonmember.size == 0 else nonmember[0]
        tpr = float(np.mean(scores.member_scores > threshold))
    else:
        threshold = nonmember[n_admit - 1]
        tpr = float(np.mean(scores.member_scores >= threshold))
    k = int(round(tpr * scores.member_scores.size))
    return tpr, k, int(scores.member_scores.size)


def clopper_pearson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact binomial confidence interval for a proportion k/n."""
    if not 0 <= k <= n or n <= 0:
        raise ValueError(f"Require 0 <= k <= n and n > 0, got k={k}, n={n}.")
    alpha = 1.0 - confidence
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lower, upper
