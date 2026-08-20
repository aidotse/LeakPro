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
from typing import NamedTuple

import numpy as np
import torch
from scipy.stats import beta
from sklearn.metrics import roc_curve

from leakpro.signals.functional import rescaled_logits


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


class TPRAtFPR(NamedTuple):
    """A TPR measurement together with the operating point it was taken at.

    ``tpr`` alone cannot be interpreted: a 0.0 from a genuinely private model and
    a 0.0 from an estimate the audit set could not resolve are the same three
    characters in a log. ``realized_fpr`` and ``threshold`` name the discrete
    operating point actually available in the data, and ``warning`` is set when
    that point falls materially short of what was asked for.
    """

    tpr: float
    events: int
    n_members: int
    realized_fpr: float
    threshold: float
    warning: str | None


# Below this fraction of the requested FPR, the nearest achievable operating
# point is too far away for the interpolated value to be treated as measured.
_RESOLUTION_RATIO = 0.5


def tpr_at_fpr(scores: AttackScores, fpr: float) -> TPRAtFPR:
    """TPR at a fixed FPR, with the operating point the estimate rests on.

    Uses the interpolated (equivalently, randomized-threshold) ROC: the TPR is
    read off the ROC curve at exactly ``fpr``, interpolating between the two
    achievable operating points that bracket it.

    On distinct scores this is never worse than the classic rule of thresholding
    at the floor(fpr * n)-th nonmember score, and usually equal. It can be
    strictly higher when member scores fall in the gap between that nonmember
    and the next one down: those members can be counted without admitting any
    further nonmember, so the realized FPR is unchanged and the TPR is higher.
    Taking the better of two operating points at the same realized FPR is the
    right call here — an attack result is a lower bound on risk, so the tightest
    bound a real adversary could achieve is the honest number to report.

    Interpolation rather than a discrete threshold, because tied scores are the
    normal case here, not an edge case: DP-SGD saturates model outputs and
    the signal saturates, so a degenerate model can produce
    large exactly-tied blocks. A rule restricted to observed distinct values is
    a step function of the data — a tie block spanning the budget forces the
    threshold above it and discards every member inside, so two extra tied
    nonmembers can move the answer from 5% to 0%. That artifact lands at TPR 0,
    which is unbeatable on the privacy axis and therefore guaranteed to sit on
    the Pareto front. Interpolation is continuous and returns chance-level TPR
    for a no-signal attack, which is the truthful answer.

    Returns:
        A :class:`TPRAtFPR`. ``events`` and ``n_members`` let callers judge the
        estimate's resolution; ``realized_fpr`` is the FPR of the nearest
        achievable point at or below ``fpr``, and ``warning`` is set when that
        point is far enough below ``fpr`` that the interpolation, not the data,
        is carrying the answer.

    """
    if not 0.0 < fpr < 1.0:
        raise ValueError(f"fpr must be in (0, 1), got {fpr}.")
    member, nonmember = scores.member_scores, scores.nonmember_scores

    labels = np.concatenate([np.ones(member.size, dtype=int), np.zeros(nonmember.size, dtype=int)])
    values = np.concatenate([member, nonmember])
    # drop_intermediate=False is required, not cosmetic: sklearn otherwise prunes
    # "collinear" points, which can remove the exact operating point at the
    # requested FPR and make the interpolated value disagree with the classic
    # floor(fpr * n) rule on distinct scores.
    fpr_curve, tpr_curve, thresholds = roc_curve(labels, values, drop_intermediate=False)

    tpr = float(np.interp(fpr, fpr_curve, tpr_curve))

    # The nearest operating point the data actually offers at or below `fpr`.
    at_or_below = np.nonzero(fpr_curve <= fpr)[0]
    idx = int(at_or_below[-1]) if at_or_below.size else 0
    realized_fpr = float(fpr_curve[idx])
    threshold = float(thresholds[idx])

    warning = None
    if realized_fpr < fpr * _RESOLUTION_RATIO:
        warning = (
            f"Nearest achievable FPR is {realized_fpr:.2%} against a requested {fpr:.2%}: the score "
            "distribution is too coarse (large tied blocks or too few nonmembers) for this operating "
            "point to be measured directly, so the reported TPR is interpolated, not observed."
        )

    k = int(round(tpr * member.size))
    return TPRAtFPR(tpr, k, int(member.size), realized_fpr, threshold, warning)



@torch.no_grad()
def confidence_signal(  # noqa: PLR0913
    model: "torch.nn.Module",
    x: "torch.Tensor",
    y: "torch.Tensor",
    device: str,
    output_kind: str = "logits",
    batch: int = 1024,
) -> np.ndarray:
    """Per-point membership signal: LeakPro's ``rescaled_logits`` over batched inference.

    The signal itself is not reimplemented here — it is
    ``leakpro.signals.functional.rescaled_logits``, Carlini's phi from LiRA, so
    the campaign and the attack stack score points identically. This function
    only adds the batched forward pass and maps ``output_kind`` onto the shape
    that function expects.

    ``output_kind``:
        "logits"        raw multiclass logits (CrossEntropyLoss models);
        "binary_logits" one raw logit column (BCEWithLogitsLoss models);
        "binary_probs"  one sigmoid probability column (BCELoss models) — the
                        probability is converted back to a logit first, because
                        ``rescaled_logits`` applies the sigmoid itself.
    """
    valid = ("logits", "binary_logits", "binary_probs")
    if output_kind not in valid:
        raise ValueError(f"output_kind must be one of {valid}, got '{output_kind}'.")

    outs = []
    for i in range(0, len(x), batch):
        out = model(x[i:i + batch].to(device)).cpu().numpy()
        labels = y[i:i + batch].reshape(-1).numpy().astype(np.int64)
        if output_kind == "binary_probs":
            # rescaled_logits sigmoids a single column, so hand it a logit.
            p = np.clip(out.reshape(-1, 1), 1e-12, 1 - 1e-12)
            out = np.log(p) - np.log1p(-p)
        elif output_kind == "binary_logits":
            out = out.reshape(-1, 1)
        outs.append(rescaled_logits(out, labels))
    return np.concatenate(outs)


def clopper_pearson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact binomial confidence interval for a proportion k/n."""
    if not 0 <= k <= n or n <= 0:
        raise ValueError(f"Require 0 <= k <= n and n > 0, got k={k}, n={n}.")
    alpha = 1.0 - confidence
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lower, upper
