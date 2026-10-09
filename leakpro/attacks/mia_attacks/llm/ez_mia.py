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

Edge cases (paper appendix E.5 / E.6) -- two *independent* rules, neither gated by the other:

- ``n_err == 0`` (no error positions at all): the paper's own strongest possible member signal
  (E.5), forced to the top regardless of aggregation. This is a structural fact about the row, not
  a "too little evidence" judgment, so it does not depend on ``min_error_positions``.
- ``N == 0`` *with at least one error position*: only meaningful for aggregations whose formula is
  genuinely undefined/infinite there (``ratio``: ``P / 0``; ``log_ratio``: ``log(P) - log(0)``).
  ``positive_fraction`` already evaluates to its own maximum (``1.0``) at ``N == 0`` without forcing;
  ``difference``/``mean_delta``/``median_delta`` have perfectly ordinary finite values there and are
  never forced by it (paper E.6 is specifically about the ``P / N -> inf`` case).

Neither rule is about "too few error positions" as a general reliability floor -- ``min_error_positions``
(default 2) is a *separate* knob for that: a row with ``1 <= n_err < min_error_positions`` and ``N != 0``
has some evidence, just not enough to trust a ratio built from one or two positions, so it is scored a
plain ``0.0`` (the reference code's floor), *not* forced to the top -- a lone error whose mass moved
entirely downward (``P == 0``, the attack's own non-member signal) must not outrank a genuine member.
``ignore_first_position`` (default True, one token of left context is too little to trust) and
``min_error_positions`` are :class:`EZMIAConfig` fields, not hard-coded, so a run under either choice
is reproducible from its config alone.

The paper's own released code (github.com/JetBrains-Research/ez-mia, read for comparison only --
nothing is ported from it, and it ships no license file) scores *every* row with ``N == 0`` or
``n_err < 2`` as a plain ``0.0``, with no separate "zero error positions is the strongest signal"
rule. For an audit tool, scoring an actually-memorised sequence (one with literally no errors, or
no downward movement at all) as a non-member is the worse failure mode, so this module keeps the
paper's two structural force-to-member cases and only applies the reference code's floor to the
genuinely-ambiguous remainder.

One-toggle-at-a-time ablation (``examples/mia/llm_mia/ez_mia_ablation.py``: same target, same
evidence, only the scoring rule or only ``ignore_first_position`` changed), GPT-2 full fine-tune,
20,000 audited rows each on WikiText and XSum:

- edge-case policy (this module's vs the reference code's): 0 rows hit any edge case on either
  dataset, so scores and metrics are identical;
- ``ignore_first_position`` False instead of True: most rows change score (WikiText 17,228, XSum
  17,956) but the metrics barely move -- WikiText AUC 0.9817 -> 0.9818, TPR@1%FPR 0.590 -> 0.597;
  XSum AUC 0.9895 -> 0.9896, TPR@1%FPR 0.778 -> 0.777.

An earlier, retracted ablation reported small metric differences between the two scoring rules despite
zero forced rows; that difference came from ``ignore_first_position``, which it had changed at the same
time.
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

# N == 0 (all movement upward, at least one error position) is only genuinely undefined/infinite for
# these two -- P / 0 and log(P) - log(0). positive_fraction already peaks at 1.0 there; difference /
# mean_delta / median_delta have ordinary finite values. See ez_scores' docstring.
_UNDEFINED_AT_N_ZERO = frozenset({"ratio", "log_ratio"})


class EZMIAConfig(LLMAttackConfig):
    """Configuration for EZ-MIA.

    ``aggregation`` exposes the paper's ablation (appendix E.3). ``ratio`` is the paper's score;
    ``log_ratio`` is its monotone transform (identical ROC); the others are alternatives the paper
    reports as within ~0.01 AUC.
    """

    aggregation: Aggregation = Field(default="ratio", description="How P and N (or delta on E) become a score")
    min_error_positions: int = Field(default=2, ge=0,
        description="Rows with fewer (but at least 1) error positions than this cannot support a "
                    "trustworthy ratio and score a plain 0.0 (reference-code floor), unless they "
                    "already hit the paper's own E.5 (zero errors) or E.6 (N == 0) member cases, "
                    "which this threshold does not override.")
    ignore_first_position: bool = Field(default=True,
        description="Exclude position 0 from the error-position set E -- one token of left context is "
                    "too little to trust. The paper's appendix never mentions this either way.")

    model_config = ConfigDict(extra="forbid")


class EZScores(NamedTuple):
    """``ez_scores``'s return value: the finite scores plus which rows got special-cased, and how."""

    scores: np.ndarray
    forced: np.ndarray        # ranked as members (paper E.5/E.6) -- not a reflection of the raw score
    insufficient: np.ndarray  # scored a plain 0.0 -- too few error positions to trust, not a member signal


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
        :class:`EZScores`: ``scores`` are ``(N,)`` finite, higher = member. ``forced`` is True where
        ``n_err == 0`` or (``N == 0`` and the aggregation is undefined there) ranked that row as a
        member (paper appendix E.5 / E.6), regardless of the raw score. ``insufficient`` is True
        where ``1 <= n_err < min_error_positions`` and the row was *not* otherwise forced -- scored a
        plain ``0.0``, too little evidence to trust rather than a member signal (see the module
        docstring). The two are mutually exclusive.

    """
    error = error.copy()
    if ignore_first_position and error.shape[1] > 0:
        error[:, 0] = False

    sel = np.where(error, delta, 0.0)
    P = np.clip(sel, 0.0, None).sum(axis=1)  # noqa: N806
    N = np.abs(np.clip(sel, None, 0.0)).sum(axis=1)  # noqa: N806
    n_err = error.sum(axis=1)

    no_errors = n_err == 0  # paper E.5: always the strongest signal, independent of aggregation
    n_zero_forced = (N == 0) & (n_err > 0) & (aggregation in _UNDEFINED_AT_N_ZERO)  # paper E.6
    forced = no_errors | n_zero_forced
    insufficient = (n_err > 0) & (n_err < min_error_positions) & ~forced

    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # forced/insufficient rows are overwritten below regardless
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

    raw = np.asarray(raw, dtype=np.float64)
    raw[insufficient] = 0.0

    # Lexicographic tiebreak among forced rows: no_errors (E.5, the paper's strongest case) ranks
    # above n_zero_forced (E.6) regardless of P; P only breaks ties within each bucket.
    p_offset = (np.nanmax(P[np.isfinite(P)]) + 1.0) if np.isfinite(P).any() else 1.0
    tiebreak = P + np.where(no_errors, p_offset, 0.0)

    return EZScores(scores=rank_top(raw, force_top=forced, tiebreak=tiebreak), forced=forced, insufficient=insufficient)


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
                        "Sequences with zero error positions, or N = 0 (all movement upward), are ranked as members "
                        "(paper appendix E.5/E.6); sequences with too few error positions to trust a ratio but not "
                        "otherwise covered score a plain 0.0 instead.",
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
        scores, forced, insufficient = ez_scores(
            delta, error, self.configs.aggregation,
            min_error_positions=self.configs.min_error_positions,
            ignore_first_position=self.configs.ignore_first_position,
        )

        n_forced = int(forced.sum())
        if n_forced:
            logger.info(f"EZ-MIA: {n_forced}/{len(scores)} sequences had zero error positions or N = 0; "
                        "ranked as members")
        n_insufficient = int(insufficient.sum())
        if n_insufficient:
            logger.info(f"EZ-MIA: {n_insufficient}/{len(scores)} sequences had fewer than "
                        f"{self.configs.min_error_positions} error positions; scored 0.0 (too little evidence)")

        result = MIAResult.from_full_scores(
            true_membership=self._audit_labels,
            signal_values=scores,
            result_name="EZ-MIA",
            metadata=self.configs.model_dump(),
        )
        return self._attach_bootstrap_if_configured(result, self._audit_labels, scores)
