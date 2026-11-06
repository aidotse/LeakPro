#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Leakage at different stages of training.

"Exposure" is the number of central training epochs before the gradient is intercepted (milestones 0, 1, 5, 10,
25); there is no multi-client simulation. Open datasets are downloaded on first use; ``--quick`` uses a synthetic table.
"""
from __future__ import annotations

from examples.gia.common import tabular_sweep as sweep


def extra(p) -> None:
    p.add_argument("--exposures", nargs="+", type=int, default=[0, 1, 5, 10, 25])
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 32])


def report(df, out) -> None:
    sweep.plot_lines(df, out / "training_exposure.png", x="epochs", hue="batch_size",
                     metrics=["tableak_acc", "exact_row_recovery"], title="Leakage vs training exposure (epochs)")
    print(sweep.summary(df, ["dataset", "variant", "epochs", "batch_size"],
                        ["tableak_acc", "exact_row_recovery", "prior_accuracy", "test_roc_auc", "test_r2"]).to_string())


def main(argv: list[str] | None = None) -> dict:
    args = sweep.parse_args(__doc__, argv, extra)
    exps = [0, 1] if args.quick else args.exposures
    cfgs = [sweep.base_config(args, **g) for g in sweep.grid(
        dataset=args.datasets, variant=args.variants, epochs=exps, batch_size=args.batch_sizes, seed=args.seeds)]
    return sweep.run_experiment(args, cfgs, report, ["dataset", "variant", "epochs", "batch_size", "seed"])


if __name__ == "__main__":
    main()
