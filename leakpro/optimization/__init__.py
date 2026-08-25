#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""PET optimization: find the utility-vs-attack-success frontier over PET hyperparameters.

The search proposes DP-SGD configurations with Optuna (Bayesian optimization),
trains a target for each, and measures its privacy with LeakPro's own RMIA attack
via :func:`leakpro.optimization.audit.run_rmia_audit`. Nothing about the attack is
reimplemented here — the privacy axis is the real RMIA result's TPR at a fixed FPR.
"""

from leakpro.optimization.audit import (
    TABULATED_FPRS,
    clopper_pearson_ci,
    resolved_proxy_tpr,
    run_rmia_audit,
    tpr_at_fixed_fpr,
)
from leakpro.optimization.frontier import pareto_trials, plot_frontier
from leakpro.optimization.knobs import Knob, KnobSpace, default_dpsgd_space
from leakpro.optimization.search import ObjectiveResult, optimize
from leakpro.optimization.validation import proxy_agreement, validate_frontier

__all__ = [
    "TABULATED_FPRS",
    "Knob",
    "KnobSpace",
    "ObjectiveResult",
    "clopper_pearson_ci",
    "default_dpsgd_space",
    "optimize",
    "pareto_trials",
    "plot_frontier",
    "proxy_agreement",
    "resolved_proxy_tpr",
    "run_rmia_audit",
    "tpr_at_fixed_fpr",
    "validate_frontier",
]
