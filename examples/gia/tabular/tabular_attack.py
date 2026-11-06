#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Gradient inversion of mixed tabular data: recover a client's rows (continuous plus one-hot categorical) from one gradient.

The private rows are synthetic: continuous features are standard normal, two categorical features are one-hot blocks.
Compares the softmax-relaxation attack with known labels and with iDLG label inference, and the raw one-hot attack
(raw one-hot slots, cosine only). Full: 40 continuous features and two 30-way categoricals; ``--quick``: 4 and two 3-way.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from examples.gia.common import cli, results, runner
from leakpro.attacks.gia_attacks.modular import RunMetadata, TargetInfo, run_attack, simulate_client
from leakpro.attacks.gia_attacks.modular.components.evaluators import TabularEvaluator
from leakpro.attacks.gia_attacks.modular.components.tabular import TabularSchema
from leakpro.attacks.gia_attacks.modular.presets import tabular, tabular_raw_onehot

METRICS = ("reconstruction_accuracy", "continuous_accuracy", "categorical_accuracy", "exact_row_recovery")


def add_args(p) -> None:
    p.add_argument("--batch-size", type=int, default=1, help="Rows in the client batch.")
    p.add_argument("--iterations", type=int, default=None, help="Attack iterations (default 2000, quick 40).")


def make_mlp(num_features: int, num_classes: int, hidden: int = 64) -> nn.Module:
    """A two-hidden-layer MLP with BatchNorm, attacked in eval mode."""
    return nn.Sequential(nn.Linear(num_features, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Linear(hidden, num_classes))


def main(argv: list[str] | None = None) -> dict:
    """Attack one synthetic batch with each variant and report the TabLeak-style metrics."""
    a = cli.parse(__doc__, argv, extra=add_args)
    if a.replot:
        print(results.load(a.out)["rows"])
        return {}
    device = cli.device_for(a)
    torch.manual_seed(a.seed)
    n_cont, cat_sizes, classes = (4, [3, 3], 10) if a.quick else (40, [30, 30], 20)
    cont = list(range(n_cont))
    groups, start = [], n_cont
    for k in cat_sizes:
        groups.append((start, k))
        start += k
    num_features = start

    def draw(n: int) -> torch.Tensor:
        return torch.cat([torch.randn(n, n_cont), *[F.one_hot(torch.randint(0, k, (n,)), k).float() for k in cat_sizes]], dim=1)

    n = a.num_images or a.batch_size
    x = draw(n).to(device)
    labels = torch.randint(0, classes, (n,)).to(device)
    model = make_mlp(num_features, classes).to(device).eval()
    schema = TabularSchema(num_features=num_features, continuous_cols=cont, categorical_col_groups=groups)
    it = a.iterations or runner.iters(a, 2000, 40)
    layout = {"continuous_cols": cont, "categorical_col_groups": groups, "num_features": num_features,
              "num_classes": classes}
    attacks = {
        "softmax, known labels": (lambda ev: tabular(**layout, label_mode="known", lr=0.1, max_iter=it, model_mode="eval",
                                                    evaluator=ev), True),
        "softmax, iDLG labels": (lambda ev: tabular(**layout, label_mode="idlg", lr=0.1, max_iter=it, model_mode="eval",
                                                   evaluator=ev), False),
        "raw one-hot": (lambda ev: tabular_raw_onehot(**layout, task_type="multiclass", lr=0.06, max_iter=it,
                                                      evaluator=ev), True),
    }
    # Public rows from the same distribution (drawn after the victim batch): the baseline for relative_gain.
    reference = draw(256)
    rows = {}
    for name, (factory, oracle) in attacks.items():
        evaluator = TabularEvaluator(schema, std_source="unit", reference_rows=reference)
        atk = factory(evaluator)
        target = TargetInfo(value=labels)
        obs = simulate_client(model, x, target, task=atk.task, modality=atk.modality, input_shape=tuple(x.shape))
        state, ctx = run_attack(atk, model=model, observation=obs, input_shape=tuple(x.shape), device=device,
                                target=target if oracle else None, metadata=RunMetadata(seed=a.seed))
        metrics = evaluator.evaluate(state.input.detach(), x.detach(), ctx=ctx, state=state).metrics
        rows[name] = {m: [float(metrics[m])] for m in METRICS if m in metrics}
        print(f"{name}: row accuracy {metrics['reconstruction_accuracy']:.1f}%, relative gain {metrics['relative_gain']:.2f}")
    return runner.finish(a, rows, batch_size=n, iterations=it)


if __name__ == "__main__":
    main()
