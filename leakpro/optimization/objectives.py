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

    The threshold is the smallest score value whose realized FPR — counted with
    the same ``>=`` rule used for members — does not exceed ``fpr``. Tied
    nonmember scores are the case that matters: DP-SGD saturates model outputs,
    producing large tied blocks, and a threshold placed inside such a block
    admits the whole block. If admitting a block would push the realized FPR
    over ``fpr``, the threshold moves above it, so the realized FPR can only be
    at or below the requested one, never over. With all-distinct scores this
    reduces exactly to the classic floor(fpr * n) rule.

    Returns:
        (tpr, n_true_positives, n_members) — the counts let callers judge the
        estimate's resolution before believing it.

    """
    if not 0.0 < fpr < 1.0:
        raise ValueError(f"fpr must be in (0, 1), got {fpr}.")
    member = scores.member_scores
    nonmember = scores.nonmember_scores
    n_admit = int(np.floor(fpr * nonmember.size))

    if n_admit == 0 or nonmember.size == 0:
        threshold = np.inf if nonmember.size == 0 else float(nonmember.max())
        tpr = float(np.mean(member > threshold))
    else:
        # admitted(v) = #{nm >= v} for each distinct value v, descending.
        ascending = np.sort(nonmember)
        distinct = np.unique(ascending)[::-1]
        admitted = nonmember.size - np.searchsorted(ascending, distinct, side="left")
        within = np.nonzero(admitted <= n_admit)[0]
        if within.size == 0:
            # Even the top tie-block alone exceeds the budget: only scores
            # strictly above every nonmember can be counted.
            tpr = float(np.mean(member > float(distinct[0])))
        else:
            threshold = float(distinct[within[-1]])
            tpr = float(np.mean(member >= threshold))
    k = int(round(tpr * member.size))
    return tpr, k, int(member.size)


def clopper_pearson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact binomial confidence interval for a proportion k/n."""
    if not 0 <= k <= n or n <= 0:
        raise ValueError(f"Require 0 <= k <= n and n > 0, got k={k}, n={n}.")
    alpha = 1.0 - confidence
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lower, upper
