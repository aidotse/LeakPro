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
    E       = { j : argmax_v p_T(v | x_<j) != x_j, j != 0 }  error positions (top-1), excluding the
                                                              first prediction (one token of context)
    P       = sum_{j in E} max(delta_j, 0)                  probability mass moved *up* by fine-tuning
    N       = sum_{j in E} |min(delta_j, 0)|                mass moved *down*
    EZ(x)   = P / N                                         higher = member

Edge cases: the paper's appendix (E.5 / E.6) says a sequence with ``N == 0`` (all movement upward) or
no error positions is the strongest possible member signal and should rank as a member. The paper's
own released reference implementation (github.com/JetBrains-Research/ez-mia, ``mia/ez_score.py``) does
not do this: it scores both that case *and* any sequence with fewer than 2 error positions as a plain
``0.0`` -- a low/neutral score, not an automatic top rank -- and always excludes position 0 from ``E``
(``ignore_bos``). This module follows the released code, since that is what reproduces the paper's
tables; verified directly against the reference repo, not just its appendix prose.
"""

import warnings
from typing import Literal

import numpy as np
from pydantic import ConfigDict, Field

from leakpro.attacks.mia_attacks.llm.abstract_llm_mia import AbstractLLMMIA, LLMAttackConfig
from leakpro.reporting.mia_result import MIAResult
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger

Aggregation = Literal["ratio", "log_ratio", "positive_fraction", "difference", "mean_delta", "median_delta"]

# Reference repo's `min_tokens` default (mia/ez_score.py): fewer error positions than this and the
# ratio is not trusted, regardless of aggregation.
MIN_ERROR_POSITIONS = 2


class EZMIAConfig(LLMAttackConfig):
    """Configuration for EZ-MIA.

    ``aggregation`` exposes the paper's ablation (appendix E.3). ``ratio`` is the paper's score;
    ``log_ratio`` is its monotone transform (identical ROC); the others are alternatives the paper
    reports as within ~0.01 AUC.
    """

    aggregation: Aggregation = Field(default="ratio", description="How P and N (or delta on E) become a score")

    model_config = ConfigDict(extra="forbid")


def ez_scores(delta: np.ndarray, error: np.ndarray, aggregation: str) -> np.ndarray:
    """Compute EZ-MIA scores from per-token deltas and the error-position mask.

    Args:
    ----
        delta: ``(N, T)`` ``lp^T - lp^R``; values outside ``error`` are ignored.
        error: ``(N, T)`` bool, True at error positions (already restricted to valid tokens).
        aggregation: One of :data:`Aggregation`.

    Returns:
    -------
        ``(N,)`` finite scores, higher = member. Position 0 of every row is excluded from ``E``
        (``ignore_bos``, matching the reference implementation). Rows with fewer than
        :data:`MIN_ERROR_POSITIONS` error positions, or ``N == 0``, score ``0.0`` -- not forced to
        the top: the paper's own reference code treats these as too little evidence to trust, not
        as a guaranteed member signal.

    """
    error = error.copy()
    if error.shape[1] > 0:
        error[:, 0] = False  # ignore_bos: the first prediction has only one token of left context

    sel = np.where(error, delta, 0.0)
    P = np.clip(sel, 0.0, None).sum(axis=1)  # noqa: N806
    N = np.abs(np.clip(sel, None, 0.0)).sum(axis=1)  # noqa: N806
    n_err = error.sum(axis=1)
    degenerate = (n_err < MIN_ERROR_POSITIONS) | (N == 0)

    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # degenerate rows are overwritten below regardless
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
    raw[degenerate] = 0.0

    # log_ratio can still legitimately be -inf on an ordinary row (P == 0, N > 0, enough error
    # positions) -- log(P/N) with P == 0. That is a real, weakest-signal score, not degenerate; place
    # every such row just below the lowest ordinary score, preserving ties between them.
    neg_inf = np.isneginf(raw)
    if neg_inf.any():
        finite = raw[~neg_inf]
        raw[neg_inf] = (finite.min() - 1.0) if finite.size else -1.0
    return raw


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
                        "error positions after the first token, where the target's top-1 prediction is wrong; "
                        "(3) split the target-minus-reference shifts there into upward mass P and downward mass N; "
                        "(4) score P/N — memorisation pushes the true token up even where the model still fails, so "
                        "members show P >> N. Sequences with fewer than 2 error positions, or N = 0, score 0.0 "
                        "(too little evidence to trust), matching the paper's own released reference code.",
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
        scores = ez_scores(delta, error, self.configs.aggregation)

        error_no_bos = error.copy()
        if error_no_bos.shape[1] > 0:
            error_no_bos[:, 0] = False
        sel = np.where(error_no_bos, delta, 0.0)
        n_degenerate = int((
            (error_no_bos.sum(1) < MIN_ERROR_POSITIONS) | (np.abs(np.clip(sel, None, 0.0)).sum(1) == 0)
        ).sum())
        if n_degenerate:
            logger.info(f"EZ-MIA: {n_degenerate}/{len(scores)} sequences had fewer than {MIN_ERROR_POSITIONS} error "
                        "positions or N = 0; scored 0.0")

        result = MIAResult.from_full_scores(
            true_membership=self._audit_labels,
            signal_values=scores,
            result_name="EZ-MIA",
            metadata=self.configs.model_dump(),
        )
        return self._attach_bootstrap_if_configured(result, self._audit_labels, scores)
