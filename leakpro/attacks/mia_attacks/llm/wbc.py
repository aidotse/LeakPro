#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""WBC: Window-Based Comparison membership inference against fine-tuned causal language models.

Chen, Du, Zhang, Kundu, Fleming, Ribeiro & Li, *Window-based Membership Inference Attacks Against
Fine-tuned Large Language Models*, arXiv:2601.02751.

The paper works with per-token *losses* ``l = -log p`` and defines ``Delta_j = l^R_j - l^T_j``
(reference minus target). Since ``l = -lp`` that is exactly ``lp^T_j - lp^R_j`` — the same
quantity EZ-MIA calls ``delta``. Then (paper §4.1.3–4.1.4):

    S_i(w)    = sum_{j=i}^{i+w-1} Delta_j                    windowed sum, n-w+1 windows
    T_sign(w) = (1 / (n-w+1)) * #{ i : S_i(w) > 0 }           fraction of windows favouring membership
    w_k       = round( w_min * (w_max / w_min)^((k-1)/(|W|-1)) )   geometric grid, duplicates kept
    S_WBC     = mean_k T_sign(w_k)                            in [0, 1], higher = member

Paper defaults: ``w_min = 2``, ``w_max = 40``, ``|W| = 10``, sign aggregation.

The paper evaluates on sequences of >= 512 tokens and never defines the case ``n < w``. Here a
window size that does not fit a sequence contributes nothing to that sequence's mean; if *no*
configured window fits, the sequence is scored with a single window of size ``n``.
"""

from typing import List, Literal, Optional

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from leakpro.attacks.mia_attacks.llm.abstract_llm_mia import AbstractLLMMIA, LLMAttackConfig
from leakpro.reporting.mia_result import MIAResult
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger

Aggregation = Literal["sign", "mean", "median", "min"]


class WBCConfig(LLMAttackConfig):
    """Configuration for WBC. Defaults are the paper's (§5.1, §5.3.2)."""

    w_min: int = Field(default=2, ge=1, description="Smallest window size")
    w_max: int = Field(default=40, ge=1, description="Largest window size")
    n_windows: int = Field(default=10, ge=2, description="Number of geometrically spaced window sizes |W|")
    aggregation: Aggregation = Field(default="sign", description="Per-window statistic; the paper's ablation (§5.3.3)")

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _check_grid(self: Self) -> Self:
        if self.w_max < self.w_min:
            raise ValueError(f"w_max ({self.w_max}) must be >= w_min ({self.w_min})")
        return self


def geometric_windows(w_min: int, w_max: int, n_windows: int) -> List[int]:
    """Paper eq. 12: ``round(w_min * (w_max/w_min)^((k-1)/(|W|-1)))`` for ``k = 1..|W|``. Duplicates are kept."""
    k = np.arange(n_windows, dtype=np.float64)
    ws = np.round(w_min * (w_max / w_min) ** (k / (n_windows - 1)))
    return [int(w) for w in ws]


def window_stat(delta: np.ndarray, lengths: np.ndarray, w: Optional[int], aggregation: str) -> np.ndarray:
    """Per-sequence statistic over all length-``w`` windows that fit inside each sequence.

    Args:
    ----
        delta: ``(N, T)`` per-token ``lp^T - lp^R`` with 0 at invalid positions.
        lengths: ``(N,)`` number of valid positions per row.
        w: Window size. ``None`` means one window spanning the whole valid prefix of each row.
        aggregation: ``sign`` (fraction of windows with positive sum), ``mean``, ``median`` or ``min``
            of the window sums.

    Returns:
    -------
        ``(N,)`` statistic, ``nan`` for rows in which no window of size ``w`` fits.

    """
    n, t = delta.shape
    if w is None:
        in_range = np.arange(t)[None, :] < lengths[:, None]
        total = np.where(in_range, delta, 0.0).sum(axis=1)
        return (total > 0).astype(np.float64) if aggregation == "sign" else total.astype(np.float64)

    if w > t:
        return np.full(n, np.nan)
    csum = np.concatenate([np.zeros((n, 1)), np.cumsum(delta, axis=1)], axis=1)  # (N, T+1)
    sums = csum[:, w:] - csum[:, :-w]                                           # (N, T-w+1), start i covers i..i+w-1
    starts = np.arange(t - w + 1)
    valid = starts[None, :] <= (lengths - w)[:, None]                            # window must end inside the sequence
    count = valid.sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        if aggregation == "sign":
            stat = ((sums > 0) & valid).sum(axis=1) / count
        elif aggregation == "mean":
            stat = np.where(valid, sums, 0.0).sum(axis=1) / count
        elif aggregation == "median":
            stat = np.nanmedian(np.where(valid, sums, np.nan), axis=1)
        elif aggregation == "min":
            stat = np.nanmin(np.where(valid, sums, np.nan), axis=1)
        else:
            raise ValueError(f"Unknown WBC aggregation: {aggregation}")
    return np.where(count > 0, stat, np.nan)


def wbc_scores(delta: np.ndarray, lengths: np.ndarray, windows: List[int], aggregation: str) -> np.ndarray:
    """Ensemble the per-window statistics (paper eq. 13), with the short-sequence rule from the module docstring."""
    per_window = np.stack([window_stat(delta, lengths, w, aggregation) for w in windows])  # (|W|, N)
    with np.errstate(invalid="ignore"):
        scores = np.nanmean(per_window, axis=0)  # windows that don't fit a row are skipped for that row
    short = np.isnan(scores)
    if short.any():
        scores[short] = window_stat(delta[short], lengths[short], None, aggregation)
    return scores


class AttackWBC(AbstractLLMMIA):
    """Window-Based Comparison membership inference (WBC)."""

    AttackConfig = WBCConfig

    def description(self: Self) -> dict:
        """Return a description of the attack."""
        return {
            "title_str": "WBC (Window-Based Comparison)",
            "reference": "Chen, Y., Du, Y., Zhang, K., Kundu, A., Fleming, C., Ribeiro, B., Li, N. Window-based "
                         "Membership Inference Attacks Against Fine-tuned Large Language Models. arXiv:2601.02751 (2026).",
            "summary": "Reference-based MIA for fine-tuned causal LMs that replaces the global average loss difference "
                       "with a sign-based vote over sliding windows of geometrically spaced sizes.",
            "detailed": "For each audited sequence: (1) compute per-token losses under the fine-tuned target and a "
                        "frozen pretrained reference; (2) for each window size in a geometric grid, slide a window "
                        "across the sequence and record whether the reference loss exceeds the target loss inside "
                        "it; (3) the fraction of windows favouring the target is the per-size statistic; (4) average "
                        "over window sizes. Sign-based voting is robust to the long-tailed loss-difference outliers "
                        "that dominate global averages.",
        }

    def prepare_attack(self: Self) -> None:
        """Run the two forward passes over the audit set."""
        self._require_references(1)
        logger.info("WBC: extracting per-token evidence for target and reference")
        self.evidence_set = self.evidence(self.audit_dataset["data"])

    def run_attack(self: Self) -> MIAResult:
        """Reduce the stored evidence to WBC scores and package them as a MIAResult."""
        target, reference = self.evidence_set.target, self.evidence_set.ref(0)
        delta = np.where(target.mask, target.logprob - reference.logprob, 0.0)
        windows = geometric_windows(self.configs.w_min, self.configs.w_max, self.configs.n_windows)
        logger.info(f"WBC: window sizes {windows}, aggregation={self.configs.aggregation}")

        scores = wbc_scores(delta, target.lengths, windows, self.configs.aggregation)
        if not np.all(np.isfinite(scores)):
            raise RuntimeError("WBC produced non-finite scores; this is a bug in the window handling")

        return MIAResult.from_full_scores(
            true_membership=self.membership_labels,
            signal_values=scores,
            result_name="WBC",
            metadata=self.configs.model_dump(),
        )
