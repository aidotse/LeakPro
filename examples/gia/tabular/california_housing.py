#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Gradient inversion of California Housing records: what does one client's gradient reveal about their blocks?

A small MLP is trained on the (standardised) housing table to predict the median house value; the attacker intercepts the
gradient of one client batch and reconstructs the private rows. Reports how close each feature is, in standard deviations
and in original units. California Housing is downloaded once through scikit-learn; ``--quick`` uses a synthetic regression
table of the same shape and skips the original-unit table.
"""
from __future__ import annotations

import numpy as np
import torch

from examples.gia.common import cli, results, runner, tabular_runner
from examples.gia.common.tabular_data import LOADERS
from leakpro.attacks.gia_attacks.modular import RunMetadata, TargetInfo, run_attack, simulate_client
from leakpro.attacks.gia_attacks.modular.components.evaluators import TabularEvaluator
from leakpro.attacks.gia_attacks.modular.presets import tabular

METRICS = ("reconstruction_accuracy", "continuous_accuracy", "mae_sigma", "rmse_sigma", "prior_accuracy", "random_accuracy")


def add_args(p) -> None:
    p.add_argument("--batch-size", type=int, default=4, help="Rows in the client batch.")
    p.add_argument("--epochs", type=int, default=5, help="Training epochs before the gradient is intercepted.")
    p.add_argument("--iterations", type=int, default=None, help="Attack iterations (default 3000, quick 40).")


def main(argv: list[str] | None = None) -> dict:
    """Train the victim MLP, attack one client batch and report per-feature reconstruction error."""
    a = cli.parse(__doc__, argv, extra=add_args)
    if a.replot:
        print(results.load(a.out)["rows"])
        return {}
    device = cli.device_for(a)
    tabular_runner.set_device(device)
    ds = tabular_runner.get_dataset("synthetic_regression" if a.quick else "california")
    model, util = tabular_runner.pretrained(ds.name, tuple(sorted(tabular_runner.PRESETS["small"].items())), a.seed,
                                             1 if a.quick else a.epochs)
    print("victim test metrics:", {k: round(v, 3) for k, v in util.items()})

    n = a.num_images or a.batch_size
    idx = torch.tensor(np.random.default_rng(a.seed).choice(len(ds.x_train), size=n, replace=False))
    x, y = ds.x_train[idx].to(device), ds.y_train[idx].to(device)
    evaluator = TabularEvaluator(ds.schema(), std_source="unit", matching="l1", reference_rows=ds.x_train,
                                 random_trials=4 if a.quick else 32, random_seed=a.seed)
    it = a.iterations or runner.iters(a, 3000, 40)
    atk = tabular(**ds.preset_kwargs(), task_type="regression", label_mode="known", lr=0.01, max_iter=it,
                  model_mode="eval", evaluator=evaluator)
    obs = simulate_client(model, x, TargetInfo(value=y), task=atk.task, modality=atk.modality, input_shape=tuple(x.shape))
    state, ctx = run_attack(atk, model=model, observation=obs, input_shape=tuple(x.shape), device=device,
                            target=TargetInfo(value=y), metadata=RunMetadata(seed=a.seed))
    result = evaluator.evaluate(state.input.detach(), x.detach(), ctx=ctx, state=state)
    metrics = result.metrics
    print(f"row accuracy {metrics['reconstruction_accuracy']:.1f}%  (prior {100 * metrics.get('prior_accuracy', 0):.1f}%)")

    if not a.quick:
        true, recon = result.extras["matched_true"].numpy(), result.extras["matched_recon"].numpy()
        sigma = np.array([ds.col_stds[i] for i in ds.continuous_cols])
        mu = np.array([ds.col_means[i] for i in ds.continuous_cols])
        print(f"\n{'feature':12s} {'true (orig)':>12s} {'recon (orig)':>13s} {'mean abs err (orig)':>20s}")
        for i, name in enumerate(ds.feature_names):
            t, r = true[:, i] * sigma[i] + mu[i], recon[:, i] * sigma[i] + mu[i]
            print(f"{name:12s} {t.mean():12.3f} {r.mean():13.3f} {np.abs(t - r).mean():20.3f}")
    rows = {"California Housing" if not a.quick else "synthetic": {m: [float(metrics[m])] for m in METRICS if m in metrics}}
    return runner.finish(a, rows, batch_size=n, iterations=it, dataset=ds.name, epochs=a.epochs, **{f"victim_{k}": v for k, v in util.items()})


if __name__ == "__main__":
    main()
