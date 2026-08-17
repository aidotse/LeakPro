#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""PET optimization: trace the utility-vs-attack-success frontier over PET hyperparameters."""

from leakpro.optimization.campaign import Campaign, EvaluationRecord
from leakpro.optimization.frontier import pareto_front, plot_frontier
from leakpro.optimization.knobs import Knob, KnobSpace, default_dpsgd_space
from leakpro.optimization.objectives import AttackScores, clopper_pearson_ci, tpr_at_fpr
from leakpro.optimization.validation import proxy_agreement, resolution_warning, validate_frontier

__all__ = [
    "AttackScores",
    "Campaign",
    "EvaluationRecord",
    "Knob",
    "KnobSpace",
    "clopper_pearson_ci",
    "default_dpsgd_space",
    "pareto_front",
    "plot_frontier",
    "proxy_agreement",
    "resolution_warning",
    "tpr_at_fpr",
    "validate_frontier",
]
