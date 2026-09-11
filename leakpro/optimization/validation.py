#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Validation pass over a finished optimization run.

The optimization loop uses a powered proxy: TPR at FPR = 1%, measured with the number
of shadow models the loop can afford. Two questions remain once it finishes, and
both are answered here by re-auditing the frontier configurations with a stronger
attack (more shadow models):

1. **What is the risk at the FPR levels we actually report?** Read at the
   validation FPRs straight from the re-audit's ``fixed_fpr_table`` — the same
   reporting path the loop used, so the numbers are directly comparable.
2. **Was the proxy legitimate?** If ranking configurations by TPR@1% disagrees
   with ranking them by a tail FPR, the loop optimized the wrong thing.
   ``proxy_agreement`` measures that rank correlation on the re-audited points.
"""

from collections.abc import Callable

import numpy as np
from optuna.study import Study
from scipy.stats import spearmanr

from leakpro.optimization.audit import tpr_at_fixed_fpr
from leakpro.optimization.frontier import pareto_trials
from leakpro.reporting.mia_result import MIAResult
from leakpro.utils.logger import logger


def _resolved_config(trial) -> dict:  # noqa: ANN001
    """The trial's full configuration: searched + fixed knobs.

    ``trial.params`` holds only the searched knobs — a knob pinned via
    ``KnobSpace.fixed`` consumes no search dimension and is absent from it.
    Anything keyed on the full configuration (per-trial artifact directories,
    re-audits) must use the ``config`` user attribute the search recorded;
    ``trial.params`` is only a fallback for studies from older runs.
    """
    return trial.user_attrs.get("config", trial.params)


def validate_frontier(
    study: Study,
    revalidate_fn: Callable[[dict[str, float]], MIAResult],
    report_fprs: tuple[float, ...] = (0.001, 0.01),
) -> list[dict]:
    """Re-audit the Pareto configurations and report TPR at the given FPR levels.

    Args:
        study: A finished optimization study.
        revalidate_fn: config -> :class:`MIAResult`, expected to run a stronger
            audit (more shadow models, larger audit set) than the loop did.
        report_fprs: FPR levels to report a TPR at.

    Returns:
        One record per frontier configuration, each with the loop's own
        (utility, TPR@proxy) and the revalidated TPRs, plus ``selection_bias``:
        the gap between the revalidated TPR at the proxy FPR and the loop's TPR,
        i.e. how optimistic the search was for that point.

    """
    proxy_fpr = study.user_attrs.get("proxy_fpr", 0.01)
    front = pareto_trials(study)
    logger.info(f"Validating {len(front)} frontier points at FPR levels {report_fprs}.")

    validated = []
    for trial in front:
        result = revalidate_fn(_resolved_config(trial))
        tprs = {f"tpr_at_{fpr}": tpr_at_fixed_fpr(result, fpr) for fpr in report_fprs}
        loop_tpr = float(trial.values[1])
        revalidated_at_proxy = tpr_at_fixed_fpr(result, proxy_fpr)
        validated.append({
            "config": _resolved_config(trial),
            "loop_utility": float(trial.values[0]),
            "loop_tpr": loop_tpr,
            "proxy_fpr": proxy_fpr,
            "revalidated": tprs,
            # Positive means the loop under-reported attack success for this
            # config, i.e. the frontier point was optimistic.
            "selection_bias": revalidated_at_proxy - loop_tpr,
        })
        logger.info(
            f"  trial {trial.number}: loop TPR@{proxy_fpr:.0%}={loop_tpr:.4f}, "
            + ", ".join(f"TPR@{fpr:.1%}={v:.4f}" for fpr, v in zip(report_fprs, tprs.values()))
        )
    return validated


def proxy_agreement(
    study: Study,
    revalidate_fn: Callable[[dict[str, float]], MIAResult],
    proxy_fpr: float = 0.01,
    target_fpr: float = 0.001,
    n_configs: int = 8,
) -> dict:
    """Check whether ranking by the proxy FPR agrees with ranking by a tail FPR.

    Re-audits ``n_configs`` trials spread across the observed proxy-TPR range (not
    only the frontier) and correlates their TPR@proxy_fpr with their
    TPR@target_fpr. Both TPRs come from the *same* re-audit, so the only thing
    differing between the two rankings is the FPR level.

    A Spearman rho well below ~0.7 means the loop optimized a quantity that does
    not order configurations the way the reported metric does.
    """
    scored = [t for t in study.trials if t.values is not None]
    if len(scored) < 2:
        raise ValueError("Need at least two finished trials to check proxy agreement.")

    ordered = sorted(scored, key=lambda t: t.values[1])
    step = max(1, len(ordered) // n_configs)
    picks = ordered[::step][:n_configs]
    logger.info(f"Proxy agreement: re-auditing {len(picks)} configs spanning the proxy-TPR range.")

    proxy_tprs, target_tprs, pairs = [], [], []
    for trial in picks:
        result = revalidate_fn(_resolved_config(trial))
        proxy = tpr_at_fixed_fpr(result, proxy_fpr)
        target = tpr_at_fixed_fpr(result, target_fpr)
        proxy_tprs.append(proxy)
        target_tprs.append(target)
        pairs.append({
            "trial": trial.number,
            f"tpr_at_{proxy_fpr}": proxy,
            f"tpr_at_{target_fpr}": target,
            "loop_tpr": float(trial.values[1]),
        })

    rho, pvalue = spearmanr(proxy_tprs, target_tprs)
    # spearmanr returns NaN when a ranking is constant (e.g. every re-audit lands
    # at TPR 0 because the tail FPR is unresolvable). That is inconclusive, not a
    # pass — surface it rather than reporting a NaN as if it were a number.
    if np.isnan(rho):
        warning = (f"Spearman rho is undefined: at least one ranking is constant, usually because the target "
                   f"FPR {target_fpr:.2%} produced no resolvable TPR differences. The proxy check is INCONCLUSIVE.")
        logger.warning(f"Proxy agreement: {warning}")
        rho_out, p_out = None, None
    else:
        logger.info(f"Proxy agreement: Spearman rho = {rho:.3f} (p = {pvalue:.3g}) over {len(picks)} configs.")
        rho_out, p_out, warning = float(rho), float(pvalue), None

    return {
        "proxy_fpr": proxy_fpr,
        "target_fpr": target_fpr,
        "spearman_rho": rho_out,
        "p_value": p_out,
        "n_configs": len(picks),
        "warning": warning,
        "pairs": pairs,
    }
