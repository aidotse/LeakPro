#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Pareto-front extraction and plotting for frontier campaigns."""

from pathlib import Path

from leakpro.optimization.campaign import EvaluationRecord


def pareto_front(records: list[EvaluationRecord]) -> list[EvaluationRecord]:
    """Non-dominated set: maximize utility, minimize attack TPR.

    Records whose attack was skipped (no ``attack_tpr``) cannot sit on the
    frontier and are excluded.
    """
    scored = [r for r in records if r.get("attack_tpr") is not None]
    front = []
    for candidate in scored:
        dominated = any(
            (other["utility"] >= candidate["utility"] and other["attack_tpr"] <= candidate["attack_tpr"])
            and (other["utility"] > candidate["utility"] or other["attack_tpr"] < candidate["attack_tpr"])
            for other in scored
        )
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda r: r["utility"])


def plot_frontier(records: list[EvaluationRecord], path: str | Path) -> Path:
    """Scatter all evaluations, highlight the Pareto front, save to ``path``."""
    import sys

    import matplotlib

    # Only force the headless backend when pyplot has not been imported yet:
    # matplotlib.use() would otherwise silently hijack an interactive session's
    # backend for the rest of the process (e.g. a notebook calling this).
    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scored = [r for r in records if r.get("attack_tpr") is not None]
    if not scored:
        raise ValueError("No records with attack results to plot.")
    front = pareto_front(records)
    proxy_fpr = scored[0].get("proxy_fpr", 0.01)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        [r["attack_tpr"] for r in scored],
        [r["utility"] for r in scored],
        s=25, color="#9aa5b1", label="evaluated configs",
    )
    ax.plot(
        [r["attack_tpr"] for r in front],
        [r["utility"] for r in front],
        "o-", color="#d1495b", label="Pareto front",
    )
    for record in front:
        low, high = record.get("attack_tpr_ci95", (None, None))
        if low is not None:
            ax.plot([low, high], [record["utility"]] * 2, color="#d1495b", alpha=0.4, lw=1)
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
