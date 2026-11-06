#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Merge independent per-modality evaluators into one EvalResult.

Detection needs two unrelated scores at once — pixel fidelity and annotation
recovery — which is exactly the split ``Evaluator`` was introduced for
(``core/interfaces.py::Evaluator``). Rather than writing a bespoke
"detection evaluator" that hardcodes PSNR/SSIM + box IoU together,
``CompositeEvaluator`` composes any named set of existing evaluators, each
scoped to its own key prefix.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

from leakpro.attacks.gia_attacks.modular.core.evaluation import EvalResult
from leakpro.attacks.gia_attacks.modular.core.interfaces import Evaluator

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.state import Context, State


class CompositeEvaluator(Evaluator):
    """Runs several evaluators against the same (recon, target) and merges results.

    Every sub-evaluator receives the exact same ``recon``/``target``/``ctx``/
    ``state`` — it's each evaluator's own job to read only what it needs from
    them (e.g. ``BoxEvaluator`` ignores ``recon``/``target`` themselves and
    reads boxes from ``state.target``/``state.artifacts``). Metric and extra
    keys are prefixed with ``"<name>."`` so two sub-evaluators can't collide.
    """

    def __init__(self, evaluators: dict[str, Evaluator]) -> None:
        self.evaluators = evaluators

    def evaluate(
        self,
        recon: Tensor,
        target: Tensor,
        *,
        ctx: Context | None = None,
        state: State | None = None,
    ) -> EvalResult:
        """Runs every wrapped evaluator and merges their metrics and extras."""
        metrics: dict[str, float] = {}
        extras: dict[str, object] = {}
        for name, evaluator in self.evaluators.items():
            sub = evaluator.evaluate(recon, target, ctx=ctx, state=state)
            metrics.update({f"{name}.{k}": v for k, v in sub.metrics.items()})
            extras.update({f"{name}.{k}": v for k, v in sub.extras.items()})
        return EvalResult(metrics=metrics, extras=extras)
