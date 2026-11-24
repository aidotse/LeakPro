#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Label recovery accuracy for attacks that infer or optimise the labels."""
from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from torch import Tensor

from leakpro.attacks.gia_attacks.modular.components.utils.matching import hungarian_align
from leakpro.attacks.gia_attacks.modular.core.evaluation import EvalResult
from leakpro.attacks.gia_attacks.modular.core.interfaces import Evaluator

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.state import Context, State


class LabelEvaluator(Evaluator):
    """Scores the labels an attack recovered against the ground truth.

    The ground truth is not part of the attack's state, so the caller sets
    ``state.artifacts[GT_KEY]`` (a ``[N]`` long tensor) after ``run_attack()`` and before
    ``evaluate()``. The recovered labels come from ``state.target`` (``value``, else the argmax of
    ``soft``). Missing state, target or ground truth gives an empty ``EvalResult`` so the
    evaluator composes inside a ``CompositeEvaluator``.

    Metrics:
      accuracy: Fraction of positions labelled correctly after the reconstructed images are
        Hungarian-aligned to the ground-truth images, as ``ImageEvaluator`` does.
      multiset_accuracy: Order-free overlap of predicted and true labels,
        ``sum(min(count_pred, count_true)) / N``; what analytical inference recovers for batches.
    """

    GT_KEY = "ground_truth_labels"

    def evaluate(
        self,
        recon: Tensor,
        target: Tensor,
        *,
        ctx: Context | None = None,  # noqa: ARG002 -- interface signature
        state: State | None = None,
    ) -> EvalResult:
        """Reports label accuracy of the reconstruction's inferred labels against the ground truth."""
        if state is None or state.target is None:
            return EvalResult(metrics={})
        truth = state.artifacts.get(self.GT_KEY)
        predicted = state.target.value
        if predicted is None and state.target.soft is not None:
            predicted = state.target.soft.argmax(dim=-1)
        if truth is None or predicted is None:
            return EvalResult(metrics={})

        truth = truth.detach().cpu()
        predicted = predicted.detach().cpu()
        n = truth.numel()
        if not (predicted.numel() == n == recon.shape[0]):
            raise ValueError(
                f"LabelEvaluator needs one predicted label, one true label and one reconstruction row per sample, got "
                f"{predicted.numel()} predicted, {n} true and {recon.shape[0]} reconstruction rows."
            )
        if recon.shape[0] > 1:
            _, col_ind = hungarian_align(target, recon)
            # Row i of the aligned reconstruction is reconstructed row col_ind[i].
            aligned = predicted[col_ind]
        else:
            aligned = predicted
        accuracy = (aligned == truth).float().mean().item()
        overlap = sum((Counter(predicted.tolist()) & Counter(truth.tolist())).values())
        return EvalResult(metrics={"accuracy": accuracy, "multiset_accuracy": overlap / n})
