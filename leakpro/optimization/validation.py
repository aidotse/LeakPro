#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Validation pass over a finished campaign.

The campaign optimizes a powered proxy (TPR at FPR = 1%). Two questions remain
open once it finishes, and both are answered here:

1. **What is the risk at the FPR level we actually report?** Tail statistics
   are re-measured on the frontier points only, with a stronger attack and a
   larger audit set — never by re-thresholding the proxy's scores, which do not
   have the resolution for it.
2. **Was the proxy allowed in the first place?** If ranking configurations by
   TPR@1% disagrees with ranking them by the reported FPR level, the campaign
   optimized the wrong thing. ``proxy_agreement`` measures that rank
   correlation; a low value is a finding, not a formality.
"""

from collections.abc import Callable

import numpy as np
from scipy.stats import spearmanr

from leakpro.optimization.campaign import EvaluationRecord
from leakpro.optimization.frontier import pareto_front
from leakpro.optimization.objectives import AttackScores, clopper_pearson_ci, tpr_at_fpr
from leakpro.utils.logger import logger

MIN_EVENTS = 10


def resolution_warning(n_nonmembers: int, fpr: float) -> str | None:
    """Warn when an audit set is too small for the requested FPR level to mean anything."""
    expected = fpr * n_nonmembers
    if expected < MIN_EVENTS:
        return (
            f"FPR = {fpr:.2%} on {n_nonmembers} nonmembers puts the threshold at "
            f"~{expected:.1f} scores: the estimate is essentially unresolved. "
            f"Use >= {int(MIN_EVENTS / fpr)} nonmembers, or report a higher FPR."
        )
    return None


def validate_frontier(
    records: list[EvaluationRecord],
    revalidate_fn: Callable[[dict[str, float]], AttackScores],
    report_fprs: tuple[float, ...] = (0.001, 0.01),
) -> list[EvaluationRecord]:
    """Re-measure the Pareto points with a stronger attack, at the reported FPR levels.

    Args:
        records: all evaluations from the campaign.
        revalidate_fn: config -> AttackScores, expected to use a stronger attack
            (more reference models) and a larger audit set than the loop did.
        report_fprs: FPR levels to report; each gets a TPR, event counts and a
            Clopper-Pearson 95% interval.

    Returns:
        The frontier records, each extended with a ``validation`` block.

    """
    front = pareto_front(records)
    logger.info(f"Validating {len(front)} frontier points at FPR levels {report_fprs}.")

    validated = []
    for record in front:
        scores = revalidate_fn(record["config"])
        n_nonmembers = len(scores.nonmember_scores)
        results = {}
        for fpr in report_fprs:
            tpr, k, n = tpr_at_fpr(scores, fpr)
            low, high = clopper_pearson_ci(k, n)
            results[f"tpr_at_{fpr}"] = {
                "tpr": tpr, "events": k, "n_members": n, "ci95": [low, high],
                "warning": resolution_warning(n_nonmembers, fpr),
            }
            logger.info(
                f"  config {record['index']}: TPR@{fpr:.1%} = {tpr:.4f} "
                f"[{low:.4f}, {high:.4f}] ({k}/{n} events)"
            )
        out = EvaluationRecord(record)
        out["validation"] = {"n_nonmembers": n_nonmembers, **results}
        validated.append(out)
    return validated


def proxy_agreement(
    records: list[EvaluationRecord],
    revalidate_fn: Callable[[dict[str, float]], AttackScores],
    proxy_fpr: float = 0.01,
    target_fpr: float = 0.001,
    n_configs: int = 8,
) -> dict:
    """Check whether ranking by the proxy FPR agrees with ranking by the reported FPR.

    Spreads ``n_configs`` evaluations deterministically across the observed
    proxy-TPR range (so the check is not confined to the frontier), re-attacks
    each, and correlates the two rankings.

    Returns a dict with Spearman rho, its p-value, and the per-config TPR pairs.
    A rho well below ~0.7 means the loop optimized a quantity that does not
    order configurations the way the reported metric does.
    """
    scored = [r for r in records if r.get("attack_tpr") is not None]
    if len(scored) < 2:
        raise ValueError("Need at least two scored evaluations to check proxy agreement.")

    ordered = sorted(scored, key=lambda r: r["attack_tpr"])
    picks = np.unique(np.linspace(0, len(ordered) - 1, min(n_configs, len(ordered))).astype(int))
    logger.info(f"Proxy agreement: re-attacking {len(picks)} configs spanning the proxy-TPR range.")

    proxy_tprs, target_tprs, pairs = [], [], []
    for i in picks:
        record = ordered[i]
        scores = revalidate_fn(record["config"])
        target_tpr, k, n = tpr_at_fpr(scores, target_fpr)
        proxy_tprs.append(record["attack_tpr"])
        target_tprs.append(target_tpr)
        pairs.append({
            "index": record["index"],
            f"proxy_tpr_at_{proxy_fpr}": record["attack_tpr"],
            f"tpr_at_{target_fpr}": target_tpr,
            "events": k, "n_members": n,
        })

    rho, pvalue = spearmanr(proxy_tprs, target_tprs)
    warning = resolution_warning(len(scores.nonmember_scores), target_fpr)
    if warning:
        logger.warning(f"Proxy agreement: {warning}")

    # spearmanr returns NaN when either ranking is constant (e.g. every
    # re-attack lands at TPR 0 because the target FPR is unresolvable). A NaN
    # must not pass silently: it would disable the "rho well below 0.7 is a
    # finding" check while looking like a number.
    if np.isnan(rho):
        undefined = ("Spearman rho is undefined: at least one ranking is constant. This usually means the "
                     f"target FPR {target_fpr:.2%} produced no resolvable TPR differences; the proxy check "
                     "is INCONCLUSIVE, not passed.")
        logger.warning(f"Proxy agreement: {undefined}")
        warning = f"{warning} {undefined}" if warning else undefined
        rho_out, p_out = None, None
    else:
        logger.info(f"Proxy agreement: Spearman rho = {rho:.3f} (p = {pvalue:.3g}) over {len(picks)} configs.")
        rho_out, p_out = float(rho), float(pvalue)

    return {
        "proxy_fpr": proxy_fpr,
        "target_fpr": target_fpr,
        "spearman_rho": rho_out,
        "p_value": p_out,
        "n_configs": len(picks),
        "resolution_warning": warning,
        "pairs": pairs,
    }
