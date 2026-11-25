#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Sweep driver shared by the tabular leakage-factor experiments.

Each experiment supplies a grid of config overrides; this module runs them, writes ``results.json`` (one row per
configuration, one value per seed), the raw trial table (``trials.csv``, what ``--replot`` reads), aggregates
mean +/- 95% CI (std with ddof=1, ci95 = 1.96 * std / sqrt(n)) and draws the plots.
"""
from __future__ import annotations

import argparse
import itertools
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from examples.gia.common import cli, results

DEFAULT_SEEDS = [7, 13, 42]
RESULT_METRICS = ("tableak_acc", "exact_row_recovery", "prior_accuracy", "random_accuracy", "gain_over_prior")
QUICK_GRID_POINTS = 3


def base_config(args: argparse.Namespace, **overrides) -> dict:  # noqa: ANN003
    """The configuration every trial starts from; ``overrides`` set the swept axes."""
    cfg = {"model": "small", "batch_size": 8, "lr": 0.06, "lr_softmax": 0.01, "max_iter": args.max_iter, "epochs": args.epochs,
           "protocol": "fedsgd", "label_mode": "known", "random_trials": args.random_trials}
    cfg.update(overrides)
    return cfg


def grid(**axes) -> list[dict]:  # noqa: ANN003
    """Cartesian product of the axes as a list of dicts."""
    keys = list(axes)
    return [dict(zip(keys, vals)) for vals in itertools.product(*axes.values())]


def parse_args(
    description: str,
    argv: list[str] | None = None,
    extra: Callable[[argparse.ArgumentParser], None] | None = None,
    default_variants: tuple[str, ...] = ("raw_onehot",),
) -> argparse.Namespace:
    """The shared CLI plus the sweep flags; ``--quick`` swaps in a synthetic table, 1 seed and a tiny grid."""
    def add(p: argparse.ArgumentParser) -> None:
        p.add_argument("--datasets", nargs="+", default=["adult", "california"], choices=["adult", "california"])
        p.add_argument("--variants", nargs="+", default=list(default_variants), choices=["raw_onehot", "softmax"])
        p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
        p.add_argument("--max-iter", type=int, default=3000, help="Attack iterations.")
        p.add_argument("--epochs", type=int, default=1, help="Training epochs before the attack ('exposure').")
        p.add_argument("--random-trials", type=int, default=32, help="Random-baseline trials.")
        if extra:
            extra(p)

    args = cli.parse(description, argv, extra=add)
    if args.quick:
        args.datasets, args.seeds, args.max_iter, args.random_trials = ["synthetic"], args.seeds[:1], 30, 4
    return args


def ci95(x: pd.Series) -> float:
    """Half-width of the 95% confidence interval of the mean."""
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def aggregate(df: pd.DataFrame, by: list[str], metrics: list[str]) -> pd.DataFrame:
    """Mean, std, ci95 and count of every metric per group."""
    metrics = [m for m in metrics if m in df]
    out = df.groupby(by, dropna=False)[metrics].agg(["mean", "std", ci95, "count"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    return out.reset_index()


def plot_lines(df: pd.DataFrame, path: Path, *, x: str, hue: str | None, metrics: list[str], title: str,
               xlog: bool = False) -> None:
    """One column per dataset, one row per metric; points are means with 95% CI, prior and random baselines as grey lines."""
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    datasets = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(len(metrics), len(datasets), figsize=(5 * len(datasets), 3.4 * len(metrics)), squeeze=False)
    for c, ds in enumerate(datasets):
        sub = df[df["dataset"] == ds]
        for r, metric in enumerate(metrics):
            ax = axes[r][c]
            if metric not in sub or sub[metric].isna().all():
                ax.set_visible(False)
                continue
            for h, g in (sub.groupby(hue) if hue else [(None, sub)]):
                a = aggregate(g, [x], [metric]).sort_values(x)
                ax.errorbar(a[x].astype(str) if not pd.api.types.is_numeric_dtype(a[x]) else a[x], a[f"{metric}_mean"],
                            yerr=a[f"{metric}_ci95"], marker="o", capsize=3, label=str(h) if hue else metric)
            if metric == "tableak_acc":
                for base, style in (("prior_accuracy", "--"), ("random_accuracy", ":")):
                    if base in sub:
                        ax.axhline(sub[base].mean(), color="gray", linestyle=style, label=base.replace("_accuracy", ""))
            if xlog and pd.api.types.is_numeric_dtype(sub[x]):
                ax.set_xscale("log", base=2)
            ax.set_title(f"{ds}: {metric}" if r == 0 else metric, fontsize=10)
            ax.set_xlabel(x)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)
            if r == 0 and c == len(datasets) - 1:
                ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved plot: {path}")


def run_experiment(
    args: argparse.Namespace,
    configs: list[dict],
    report: Callable[[pd.DataFrame, Path], None],
    axes: list[str],
) -> dict:
    """Run (or ``--replot``) an experiment. ``axes`` are the config keys that vary; ``report(df, out)`` plots and prints."""
    out = Path(args.out)
    if args.replot:
        report(pd.read_csv(out / "trials.csv"), out)
        return {}

    from examples.gia.common.tabular_runner import run_trial, set_device  # noqa: PLC0415

    set_device(cli.device_for(args))
    if args.quick:
        configs = configs[:QUICK_GRID_POINTS]
    rows, t0 = [], time.time()
    for i, cfg in enumerate(configs, 1):
        row = run_trial(cfg)
        rows.append(row)
        shown = ", ".join(f"{k}={cfg[k]}" for k in axes if k in cfg)
        print(f"[{i}/{len(configs)}] {shown} -> tableak={row['tableak_acc']:.3f} prior={row.get('prior_accuracy', float('nan')):.3f}"
              f" ({time.time() - t0:.0f}s)")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "trials.csv", index=False)
    by_label: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        label = ", ".join(f"{k}={row[k]}" for k in axes if k != "seed")
        cell = by_label.setdefault(label, {m: [] for m in RESULT_METRICS if m in row})
        for m in cell:
            cell[m].append(float(row[m]))
    results.save(out, by_label, {"command": args.command, "quick": args.quick, "seeds": args.seeds, "max_iter": args.max_iter,
                                 "epochs": args.epochs, "datasets": args.datasets})
    results.emit(by_label)
    report(df, out)
    return by_label


def summary(df: pd.DataFrame, by: list[str], metrics: list[str]) -> pd.DataFrame:
    """Compact table: one 'mean +/- ci95' cell per metric."""
    agg = aggregate(df, by, metrics)
    out = agg[by].copy()
    out["n"] = agg[f"{[m for m in metrics if m in df][0]}_count"].astype(int)
    for m in metrics:
        if m in df:
            out[m] = [f"{a:.3f}+-{b:.3f}" if pd.notna(a) else "-" for a, b in zip(agg[f"{m}_mean"], agg[f"{m}_ci95"])]
    return out
