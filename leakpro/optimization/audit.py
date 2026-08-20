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
target metadata, DP-SGD noise and clipping from the target's ``dpsgd`` config —
so every reference model matches the candidate target it is used to attack.
"""

from scipy.stats import beta

from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.leakpro import LeakPro
from leakpro.reporting.mia_result import MIAResult
from leakpro.utils.logger import logger

RMIA_RESULT_NAME = "RMIA"


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
