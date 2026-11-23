#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Bridge from a trained target to LeakPro's own RMIA audit.

This module is the whole point of the package's integration into LeakPro. The
privacy axis of the frontier is measured by
:class:`leakpro.attacks.mia_attacks.rmia.AttackRMIA`, run through the ordinary
:class:`leakpro.leakpro.LeakPro` pipeline — never by an attack reimplemented
here. The optimizer (``leakpro.optimization.search``) supplies configurations;
this module turns "a target model saved on disk" into "a TPR at a fixed FPR from
the real RMIA attack".

Shadow-model mimicry is a property of that pipeline, not something added here:
:class:`leakpro.attacks.utils.shadow_model_handler.ShadowModelHandler` trains the
RMIA reference (shadow) models through the *same* input handler and the *same*
stored training configuration as the target — optimizer and batch size from the
target metadata, DP-SGD noise and clipping from the target's ``dpsgd`` config.
One dimension is NOT automatic: under balanced sampling every reference trains
on ``len(shadow_population) // 2`` points, so the references match the target's
training-set size — and hence its DP-SGD sampling rate ``q = B/N`` and step
count — only when the audit population is exactly twice the target's training
set. Callers must enforce that coupling (the CIFAR example does); otherwise the
RMIA null distribution comes from a different noise-per-example regime than the
candidate being measured.
"""

import numpy as np
from scipy.stats import beta

from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.leakpro import LeakPro
from leakpro.reporting.mia_result import MIAResult
from leakpro.utils.logger import logger

RMIA_RESULT_NAME = "RMIA"

# The FPR levels MIAResult tabulates (its fixed_fpr_table also holds 0%, which
# is not a usable operating point). A proxy_fpr outside this set cannot be read
# back from the table at all.
TABULATED_FPRS = (0.0001, 0.001, 0.01, 0.1)


def run_rmia_audit(
    handler_cls: type[AbstractInputHandler],
    config_path: str,
    model_handler: type[AbstractInputHandler] | None = None,
) -> MIAResult:
    """Run a LeakPro audit and return its RMIA result.

    The audit config at ``config_path`` must list the ``rmia`` attack. Every
    attack in the config is executed; this helper returns the RMIA one and errors
    if it is absent, rather than silently scoring with whatever ran.

    Args:
        handler_cls: The user's data (and, if ``model_handler`` is None, model)
            input handler class.
        config_path: Path to the audit YAML for this configuration.
        model_handler: Optional separate handler providing ``train``/``eval``.

    Returns:
        The :class:`MIAResult` produced by ``AttackRMIA``.

    """
    leakpro = LeakPro(handler_cls, config_path, model_handler=model_handler)
    results = leakpro.run_audit()

    rmia = [r for r in results if getattr(r, "result_name", "").upper() == RMIA_RESULT_NAME]
    if not rmia:
        ran = [getattr(r, "result_name", "?") for r in results]
        raise ValueError(
            f"The audit at {config_path} produced no RMIA result (ran: {ran}). "
            "Add `- attack: rmia` to the audit config's attack_list."
        )
    if len(rmia) > 1:
        logger.warning(f"Audit produced {len(rmia)} RMIA results; using the first.")
    return rmia[0]


def tpr_at_fixed_fpr(result: MIAResult, fpr: float = 0.01) -> float:
    """Read TPR at a fixed FPR from an :class:`MIAResult`'s fixed-FPR table.

    Uses the table the reporting stack already computed
    (``MIAResult.fixed_fpr_table``) rather than re-deriving TPR from the raw
    scores, so the optimizer and any LeakPro report agree on the number.

    Args:
        result: An MIAResult with a populated ``fixed_fpr_table``.
        fpr: The false-positive rate to read the TPR at (default 1%).

    Returns:
        TPR as a fraction in [0, 1].

    """
    table = result.fixed_fpr_table
    if not table:
        raise ValueError(
            "MIAResult has no fixed_fpr_table (the attack produced no ROC). "
            "TPR at a fixed FPR cannot be read from this result."
        )
    # Keys are formatted "TPR@<p>%FPR" with <p> a percent; match numerically so a
    # formatting change in the reporting stack cannot silently break the lookup.
    target_percent = fpr * 100.0
    for key, value in table.items():
        try:
            percent = float(key.removeprefix("TPR@").removesuffix("%FPR"))
        except ValueError:
            continue
        if abs(percent - target_percent) < 1e-9:
            return float(value)
    raise ValueError(
        f"No TPR at FPR={fpr:.4%} in the fixed-FPR table (available: {sorted(table)}). "
        "MIAResult reports TPR at FPR levels {0%, 0.01%, 0.1%, 1%, 10%}."
    )


def interpolated_tpr_at_fpr(result: MIAResult, fpr: float) -> float | None:
    """Randomized-threshold TPR at exactly ``fpr``, by linear interpolation on the ROC.

    The fixed-FPR table reports the TPR at the largest *achievable* FPR at or
    below the target — under tied scores that operating point can sit well below
    the requested one, and the TPR shortfall from reading the curve there is
    unbounded (it is the member mass inside the ROC segment straddling the
    target). Interpolating across that segment is the conventional randomized-
    threshold reading of "TPR at x% FPR": continuous in the scores, evaluated at
    the same FPR for every result, and identical to the vertex value whenever a
    vertex exists at ``fpr``. The trivial endpoints (0, 0) and (1, 1) — always
    reject, always admit — are added before interpolating.

    Returns None when the result has no ROC.
    """
    if result.fpr is None or result.tpr is None:
        return None
    f = np.r_[0.0, np.asarray(result.fpr, dtype=float), 1.0]
    t = np.r_[0.0, np.asarray(result.tpr, dtype=float), 1.0]
    return float(np.interp(fpr, f, t))


def resolved_proxy_tpr(result: MIAResult, proxy_fpr: float) -> tuple[float | None, float | None, bool]:
    """The search objective: TPR measured at exactly the proxy FPR, or None to prune.

    The objective is the *interpolated* TPR (see :func:`interpolated_tpr_at_fpr`)
    so that every trial is measured at the same operating point. Reading the
    fixed-FPR table instead would report each trial at its own realized FPR
    anywhere below the proxy — an error that only points down, on a minimized
    axis, and that grows with the noise multiplier being searched, so TPE would
    treat the most-saturated (worst-measured) trials as the most private ones.
    Callers wanting parity with LeakPro reports should record
    :func:`tpr_at_fixed_fpr` alongside, as an extra field, not as the objective.

    A degenerate audit (no ROC at all — every score saturated to one value)
    still returns ``tpr=None``: there is no measurement to interpolate, and the
    trial must be pruned rather than scored.

    Returns:
        ``(tpr, realized_fpr, degenerate)`` — ``realized_fpr`` is the largest
        achievable FPR at or below ``proxy_fpr`` (None when there is no ROC),
        recorded so the resolution behind each number stays auditable.

    """
    degenerate = not result.fixed_fpr_table
    realized_fpr = None
    if result.fpr is not None:
        fpr = np.asarray(result.fpr, dtype=float)
        at_or_below = fpr[fpr <= proxy_fpr]
        realized_fpr = float(at_or_below.max()) if at_or_below.size else 0.0
    if degenerate:
        return None, realized_fpr, degenerate
    return interpolated_tpr_at_fpr(result, proxy_fpr), realized_fpr, degenerate


def clopper_pearson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact binomial confidence interval for a proportion k/n.

    Useful for putting an interval on a reported TPR@FPR (k = TPR-implied events,
    n = number of members). It captures binomial sampling error over audit points
    only, not target-training or reference-model randomness.
    """
    if not 0 <= k <= n or n <= 0:
        raise ValueError(f"Require 0 <= k <= n and n > 0, got k={k}, n={n}.")
    alpha = 1.0 - confidence
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lower, upper
