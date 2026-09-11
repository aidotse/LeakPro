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

Edge cases the paper fixes (appendix E.5 / E.6): if ``N == 0`` (all movement upward) or the sequence
has no error positions, it is the strongest possible member signal and is classified as a member.
Those rows are routed through :func:`~leakpro.attacks.mia_attacks.llm.abstract_llm_mia.rank_top`
so they rank above every ordinary score without emitting ``inf``.
"""

import warnings
from typing import Literal

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
        ``(N,)`` finite scores, higher = member. Rows with no error positions or ``N == 0`` are
        forced above all others via :func:`rank_top`.

    """
    sel = np.where(error, delta, 0.0)
    P = np.clip(sel, 0.0, None).sum(axis=1)  # noqa: N806
    N = np.abs(np.clip(sel, None, 0.0)).sum(axis=1)  # noqa: N806
    n_err = error.sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows are handled by rank_top below
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

    # N == 0 with P > 0 → EZ = +inf → member; no error positions → member (paper E.5 / E.6).
    force_top = (N == 0) | (n_err == 0)
    return rank_top(raw, force_top=force_top, tiebreak=P)


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
                        "Sequences with N = 0 or no error positions are classified as members.",
        }

    def prepare_attack(self: Self) -> None:
        """Run the two forward passes over the audit set."""
        self._require_references(1)
        logger.info("EZ-MIA: extracting per-token evidence for target and reference")
        self.evidence_set = self.evidence(self.audit_dataset["data"])

    def run_attack(self: Self) -> MIAResult:
        """Reduce the stored evidence to EZ scores and package them as a MIAResult."""
        target, reference = self.evidence_set.target, self.evidence_set.ref(0)
        delta = target.logprob - reference.logprob
        error = (target.argmax != target.token_ids) & target.mask
        scores = ez_scores(delta, error, self.configs.aggregation)

        n_forced = int(((np.abs(np.clip(np.where(error, delta, 0.0), None, 0.0)).sum(1) == 0) | (error.sum(1) == 0)).sum())
        if n_forced:
            logger.info(f"EZ-MIA: {n_forced}/{len(scores)} sequences had N = 0 or no error positions; ranked as members")

        return MIAResult.from_full_scores(
            true_membership=self.membership_labels,
            signal_values=scores,
            result_name="EZ-MIA",
            metadata=self.configs.model_dump(),
        )
