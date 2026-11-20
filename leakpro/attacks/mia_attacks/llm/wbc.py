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

Paper defaults: ``w_min = 2``, ``w_max = 40``, ``|W| = 10``, sign aggregation. **Caution:** the paper
*text* gives eq. 12, but the reference codebase (github.com/Stry233/WBC) has no formula at all — its
``configs/example.yaml`` lists the window sizes verbatim: ``[2, 3, 4, 6, 9, 13, 18, 25, 32, 40]``.
``geometric_windows(2, 40, 10)`` computes ``[2, 3, 4, 5, 8, 11, 15, 21, 29, 40]`` instead, and neither
``round``, ``floor`` nor ``ceil`` on eq. 12 reproduces the config list, so this is not a rounding
difference. Set ``window_lengths`` explicitly (below) to reproduce the reference implementation's exact
numbers rather than relying on the formula to match them.

**Short sequences — matches the reference codebase's clamping, not a skip-then-fallback:**
``attacks/wbc.py``'s ``_compute_window_score`` in github.com/Stry233/WBC clamps
``effective_window_size = min(window_size, min_length)`` per sequence, so *every* configured window
size always contributes a score for every sequence — never skipped, and no separate whole-sequence
fallback exists. Implemented the same way here: distinct configured window sizes can clamp to the
same effective size for a short row (contributing the same value more than once to that row's mean
over window sizes) — that duplication is intentional, matching the reference behaviour exactly.
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

    w_min: int = Field(default=2, ge=1, description="Smallest window size (ignored if window_lengths is set)")
    w_max: int = Field(default=40, ge=1, description="Largest window size (ignored if window_lengths is set)")
    n_windows: int = Field(
        default=10, ge=2, description="Number of geometrically spaced window sizes |W| (ignored if window_lengths is set)"
    )
    window_lengths: Optional[List[int]] = Field(
        default=None,
        description="Explicit window sizes, used verbatim instead of w_min/w_max/n_windows's geometric "
                    "formula. Set this to a reference implementation's config list (WBC: "
                    "[2,3,4,6,9,13,18,25,32,40]) to reproduce its exact numbers -- the formula does not.",
    )
    aggregation: Aggregation = Field(default="sign", description="Per-window statistic; the paper's ablation (§5.3.3)")

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _check_grid(self: Self) -> Self:
        if self.window_lengths is not None:
            if len(self.window_lengths) == 0:
                raise ValueError("window_lengths, if given, must be non-empty")
            if any(w < 1 for w in self.window_lengths):
                raise ValueError(f"window_lengths must all be >= 1, got {self.window_lengths}")
        elif self.w_max < self.w_min:
            raise ValueError(f"w_max ({self.w_max}) must be >= w_min ({self.w_min})")
        return self


def geometric_windows(w_min: int, w_max: int, n_windows: int) -> List[int]:
    """Paper eq. 12: ``round(w_min * (w_max/w_min)^((k-1)/(|W|-1)))`` for ``k = 1..|W|``. Duplicates are kept."""
    k = np.arange(n_windows, dtype=np.float64)
    ws = np.round(w_min * (w_max / w_min) ** (k / (n_windows - 1)))
    return [int(w) for w in ws]


def _reduce_window_sums(sums: np.ndarray, aggregation: str) -> float:
    """One row's sliding-window sums (already sized to that row's own clamped window) -> one statistic."""
    if aggregation == "sign":
        return float(np.mean(sums > 0))
    if aggregation == "mean":
        return float(np.mean(sums))
    if aggregation == "median":
        return float(np.median(sums))
    if aggregation == "min":
        return float(np.min(sums))
    raise ValueError(f"Unknown WBC aggregation: {aggregation}")


def window_stat(delta_row: np.ndarray, length: int, w: int, aggregation: str) -> float:
    """One row, one window size -> one statistic (paper's ``S_i(w)``, reduced by ``aggregation``).

    ``w`` is clamped down to ``length`` when the row is shorter than the configured window size --
    matches the reference codebase's ``_compute_window_score`` (github.com/Stry233/WBC,
    ``attacks/wbc.py``): every configured window size always contributes a score, never skipped.

    Args:
    ----
        delta_row: ``(T,)`` per-token ``lp^T - lp^R`` for one sequence; only the first ``length``
            positions are read (later ones may be arbitrary padding).
        length: Number of valid positions in `delta_row`.
        w: Configured window size (clamped to `length` internally if too large).
        aggregation: ``sign`` (fraction of windows with positive sum), ``mean``, ``median`` or ``min``.

    Returns:
    -------
        One float statistic for this row and window size.

    """
    if length == 0:
        return 0.0
    eff_w = min(w, length)
    csum = np.concatenate([[0.0], np.cumsum(delta_row[:length])])
    sums = csum[eff_w:] - csum[:-eff_w]                     # (length-eff_w+1,) window sums, start i covers i..i+eff_w-1
    return _reduce_window_sums(sums, aggregation)


def wbc_scores(delta: np.ndarray, lengths: np.ndarray, windows: List[int], aggregation: str) -> np.ndarray:
    """Ensemble the per-window-size statistics (paper eq. 13): the mean of `window_stat` over `windows`, per row."""
    n = delta.shape[0]
    scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        length = int(lengths[i])
        scores[i] = float(np.mean([window_stat(delta[i], length, w, aggregation) for w in windows]))
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
        """Run the two forward passes over the (possibly `max_samples`-subsampled) audit set."""
        self._require_references(1)
        indices, self._audit_labels = self._audit_indices_and_labels()
        logger.info(f"WBC: extracting per-token evidence for target and reference ({len(indices)} sequences)")
        self.evidence_set = self.evidence(indices)

    def run_attack(self: Self) -> MIAResult:
        """Reduce the stored evidence to WBC scores and package them as a MIAResult."""
        target, reference = self.evidence_set.target, self.evidence_set.ref(0)
        delta = np.where(target.mask, target.logprob - reference.logprob, 0.0)
        windows = (self.configs.window_lengths if self.configs.window_lengths is not None
                  else geometric_windows(self.configs.w_min, self.configs.w_max, self.configs.n_windows))
        logger.info(f"WBC: window sizes {windows}, aggregation={self.configs.aggregation}")

        scores = wbc_scores(delta, target.lengths, windows, self.configs.aggregation)
        if not np.all(np.isfinite(scores)):
            raise RuntimeError("WBC produced non-finite scores; this is a bug in the window handling")

        result = MIAResult.from_full_scores(
            true_membership=self._audit_labels,
            signal_values=scores,
            result_name="WBC",
            metadata=self.configs.model_dump(),
        )
        return self._attach_bootstrap_if_configured(result, self._audit_labels, scores)
