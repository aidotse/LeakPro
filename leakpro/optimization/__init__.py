#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""PET optimization: trace the utility-vs-attack-success frontier over PET hyperparameters."""

from leakpro.optimization.campaign import Campaign, EvaluationRecord
from leakpro.optimization.frontier import pareto_front, plot_frontier
from leakpro.optimization.knobs import Knob, KnobSpace, default_dpsgd_space
from leakpro.optimization.objectives import (
    AttackScores,
    clopper_pearson_ci,
    confidence_signal,
    tpr_at_fpr,
)
from leakpro.optimization.training import (
    PETRecipe,
    build_campaign_fns,
    make_opacus_compatible,
    train_with_dpsgd,
)
from leakpro.optimization.validation import proxy_agreement, resolution_warning, validate_frontier

__all__ = [
    "AttackScores",
    "Campaign",
    "EvaluationRecord",
    "Knob",
    "KnobSpace",
    "PETRecipe",
    "build_campaign_fns",
    "clopper_pearson_ci",
    "confidence_signal",
    "default_dpsgd_space",
    "make_opacus_compatible",
    "pareto_front",
    "plot_frontier",
    "proxy_agreement",
    "resolution_warning",
    "tpr_at_fpr",
    "train_with_dpsgd",
    "validate_frontier",
]
