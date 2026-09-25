#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""EZ-MIA: Error Zone membership inference against fine-tuned causal language models.

Ilić, Stanojević & Cvejoski, *Powerful Training-Free Membership Inference Against Fine-Tuned
Autoregressive Language Models*, arXiv:2601.12104.

For a sequence ``x`` with per-token log-probs ``lp^T`` under the fine-tuned target and ``lp^R``
under a frozen pretrained reference (paper §3.3–3.4):

    delta_j = lp^T_j - lp^R_j
    E       = { j : argmax_v p_T(v | x_<j) != x_j }        error positions (top-1)
    P       = sum_{j in E} max(delta_j, 0)                  probability mass moved *up* by fine-tuning
    N       = sum_{j in E} |min(delta_j, 0)|                mass moved *down*
    EZ(x)   = P / N                                         higher = member

Edge cases (paper appendix E.5 / E.6): a sequence with ``N == 0`` (all movement upward), or fewer than
``min_error_positions`` error positions, is treated as the strongest possible member signal and ranked
above every ordinary score via :func:`~leakpro.attacks.mia_attacks.llm.abstract_llm_mia.rank_top` --
not scored ``0.0``. For an audit tool, scoring an actually-memorised sequence as a non-member is the
worse failure mode, so ambiguous cases default to "member". ``min_error_positions`` (default 2) and
``ignore_first_position`` (default True, one token of left context is too little to trust) are
:class:`EZMIAConfig` fields, not hard-coded, so a run under either choice is reproducible from its
config alone.

The paper's own released code (github.com/JetBrains-Research/ez-mia, read for comparison only --
nothing is ported from it, and it ships no license file) instead scores both edge cases as a plain
``0.0``. An isolated ablation (same trained target and population, only the scoring rule swapped)
found this makes almost no practical difference: on both WikiText and XSum, zero sequences in a
20,000-sequence audit ever hit ``N == 0`` or fewer than 2 error positions, and the two rules' AUC/TPR
agreed to within single-seed noise. The large gap this repo previously (dead02b5) attributed to this
choice was actually caused by two unrelated fixes landing in the same commit (a disjoint validation
split for checkpoint selection, and `prepare_target.py`'s `prefix` chunking mode) -- not by this one.
"""

import warnings
from typing import Literal, NamedTuple

import numpy as np
from pydantic import ConfigDict, Field

from leakpro.attacks.mia_attacks.llm.abstract_llm_mia import AbstractLLMMIA, LLMAttackConfig, rank_top
from leakpro.reporting.mia_result import MIAResult
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger

Aggregation = Literal["ratio", "log_ratio", "positive_fraction", "difference", "mean_delta", "median_delta"]


class EZMIAConfig(LLMAttackConfig):
    """Configuration for EZ-MIA.

    ``aggregation`` exposes the paper's ablation (appendix E.3). ``ratio`` is the paper's score;
    ``log_ratio`` is its monotone transform (identical ROC); the others are alternatives the paper
    reports as within ~0.01 AUC.
    """

    aggregation: Aggregation = Field(default="ratio", description="How P and N (or delta on E) become a score")
    min_error_positions: int = Field(default=2, ge=0,
        description="Rows with fewer error positions than this cannot support a trustworthy ratio and "
                    "are ranked as members (paper appendix E.5), same as N == 0 (appendix E.6).")
    ignore_first_position: bool = Field(default=True,
        description="Exclude position 0 from the error-position set E -- one token of left context is "
                    "too little to trust. The paper's appendix never mentions this either way.")

    model_config = ConfigDict(extra="forbid")


class EZScores(NamedTuple):
    """``ez_scores``'s return value: the finite scores plus which rows were forced to the top."""

    scores: np.ndarray
    forced: np.ndarray


def ez_scores(
    delta: np.ndarray,
    error: np.ndarray,
    aggregation: str,
    min_error_positions: int = 2,
    ignore_first_position: bool = True,
) -> EZScores:
    """Compute EZ-MIA scores from per-token deltas and the error-position mask.

    Args:
    ----
        delta: ``(N, T)`` ``lp^T - lp^R``; values outside ``error`` are ignored.
        error: ``(N, T)`` bool, True at error positions (already restricted to valid tokens).
        aggregation: One of :data:`Aggregation`.
        min_error_positions: Rows with fewer error positions than this are forced to the top (see
            :class:`EZMIAConfig`).
        ignore_first_position: If True, position 0 is never counted as an error position.

    Returns:
    -------
        :class:`EZScores`: ``scores`` are ``(N,)`` finite, higher = member; ``forced`` is ``(N,)``
        bool, True where ``N == 0`` or fewer than ``min_error_positions`` error positions forced that
        row to rank as a member (paper appendix E.5 / E.6) rather than reflecting an ordinary ratio.

    """
    error = error.copy()
    if ignore_first_position and error.shape[1] > 0:
        error[:, 0] = False

    sel = np.where(error, delta, 0.0)
    P = np.clip(sel, 0.0, None).sum(axis=1)  # noqa: N806
    N = np.abs(np.clip(sel, None, 0.0)).sum(axis=1)  # noqa: N806
    n_err = error.sum(axis=1)
    forced = (n_err < min_error_positions) | (N == 0)

    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # forced rows are re-ranked by rank_top below regardless
        if aggregation == "ratio":
            raw = P / N
        elif aggregation == "log_ratio":
            raw = np.log(P) - np.log(N)
        elif aggregation == "positive_fraction":
            raw = P / (P + N)
        elif aggregation == "difference":
            raw = P - N
        elif aggregation == "mean_delta":
            raw = sel.sum(axis=1) / n_err
        elif aggregation == "median_delta":
            raw = np.nanmedian(np.where(error, delta, np.nan), axis=1)
        else:
            raise ValueError(f"Unknown EZ-MIA aggregation: {aggregation}")

    return EZScores(scores=rank_top(raw, force_top=forced, tiebreak=P), forced=forced)


class AttackEZMIA(AbstractLLMMIA):
    """Error Zone membership inference (EZ-MIA)."""

    AttackConfig = EZMIAConfig

    def description(self: Self) -> dict:
        """Return a description of the attack."""
        return {
            "title_str": "EZ-MIA (Error Zone membership inference)",
            "reference": "Ilić, D., Stanojević, D., Cvejoski, K. Powerful Training-Free Membership Inference Against "
                         "Fine-Tuned Autoregressive Language Models. arXiv:2601.12104 (2026).",
            "summary": "Training-free reference-based MIA for fine-tuned causal LMs that measures the directional "
                       "imbalance of log-probability shifts at the tokens the target mispredicts.",
            "detailed": "For each audited sequence: (1) compute per-token log-probabilities under the fine-tuned "
                        "target and a frozen pretrained reference (two forward passes, no training); (2) keep only "
                        "error positions, where the target's top-1 prediction is wrong; (3) split the target-minus-"
                        "reference shifts there into upward mass P and downward mass N; (4) score P/N — memorisation "
                        "pushes the true token up even where the model still fails, so members show P >> N. "
                        "Sequences with fewer than min_error_positions error positions, or N = 0, are ranked as "
                        "members (paper appendix E.5/E.6) rather than scored on an untrustworthy ratio.",
        }

    def prepare_attack(self: Self) -> None:
        """Run the two forward passes over the (possibly `max_samples`-subsampled) audit set."""
        self._require_references(1)
        indices, self._audit_labels = self._audit_indices_and_labels()
        logger.info(f"EZ-MIA: extracting per-token evidence for target and reference ({len(indices)} sequences)")
        self.evidence_set = self.evidence(indices)

    def run_attack(self: Self) -> MIAResult:
        """Reduce the stored evidence to EZ scores and package them as a MIAResult."""
        target, reference = self.evidence_set.target, self.evidence_set.ref(0)
        delta = target.logprob - reference.logprob
        error = (target.argmax != target.token_ids) & target.mask
        scores, forced = ez_scores(
            delta, error, self.configs.aggregation,
            min_error_positions=self.configs.min_error_positions,
            ignore_first_position=self.configs.ignore_first_position,
        )

        n_forced = int(forced.sum())
        if n_forced:
            logger.info(f"EZ-MIA: {n_forced}/{len(scores)} sequences had fewer than "
                        f"{self.configs.min_error_positions} error positions or N = 0; ranked as members")

        result = MIAResult.from_full_scores(
            true_membership=self._audit_labels,
            signal_values=scores,
            result_name="EZ-MIA",
            metadata=self.configs.model_dump(),
        )
        return self._attach_bootstrap_if_configured(result, self._audit_labels, scores)
