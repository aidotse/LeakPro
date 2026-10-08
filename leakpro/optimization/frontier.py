#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Pareto-front extraction and plotting for privacy-utility optimization runs.

The frontier is Optuna's own non-dominated set (``study.best_trials``); this
module only exposes it under a task-specific name and plots it. Each trial's
values are ``(utility, attack_tpr)`` in the order the search registered its
objectives (maximize utility, minimize TPR).
"""

from pathlib import Path

from optuna.study import Study
from optuna.trial import FrozenTrial


def pareto_trials(study: Study) -> list[FrozenTrial]:
    """The non-dominated trials: maximize utility, minimize attack TPR."""
    return list(study.best_trials)


def _utility_tpr(trial: FrozenTrial) -> tuple[float, float]:
    """Unpack a trial's (utility, tpr) objective values."""
    utility, tpr = trial.values
    return float(utility), float(tpr)


def plot_frontier(study: Study, path: str | Path) -> Path:
    """Scatter all finished trials, highlight the Pareto front, save to ``path``."""
    import sys

    import matplotlib

    # Only force the headless backend when pyplot has not been imported yet, so a
    # notebook calling this keeps its interactive backend.
    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scored = [t for t in study.trials if t.values is not None]
    if not scored:
        raise ValueError("No trials with objective values to plot.")
    front = sorted(pareto_trials(study), key=lambda t: t.values[0])
    proxy_fpr = study.user_attrs.get("proxy_fpr", 0.01)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        [t.values[1] for t in scored],
        [t.values[0] for t in scored],
        s=25, color="#9aa5b1", label="evaluated configs",
    )
    ax.plot(
        [t.values[1] for t in front],
        [t.values[0] for t in front],
        "o-", color="#d1495b", label="Pareto front",
    )
    ax.set_xlabel(f"Attack TPR @ FPR = {proxy_fpr:.0%} (lower is safer)")
    ax.set_ylabel("Utility (higher is better)")
    ax.set_title("Utility vs. attack success")
    ax.legend()
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
