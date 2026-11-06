#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Leakage vs batch size (FedSGD, labels known) on Adult and California Housing.

Compares the raw one-hot attack (raw one-hot slots, cosine only) with the softmax-relaxation attack. One central model
stands in for the client population. ``--quick`` uses a synthetic table.
"""
from __future__ import annotations

from examples.gia.common import tabular_sweep as sweep


def extra(p) -> None:
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    p.add_argument("--models", nargs="+", default=["small"], choices=["small", "medium", "large"])


def report(df, out) -> None:
    sweep.plot_lines(df, out / "batch_sizes.png", x="batch_size", hue="variant", xlog=True,
                     metrics=["tableak_acc", "exact_row_recovery"], title="Leakage vs batch size (FedSGD, labels known)")
    print(sweep.summary(df, ["dataset", "variant", "batch_size"],
                        ["tableak_acc", "continuous_accuracy", "categorical_accuracy", "emr_90", "exact_row_recovery",
                         "prior_accuracy", "random_accuracy", "gain_over_prior", "test_roc_auc", "test_r2"]).to_string())


def main(argv: list[str] | None = None) -> dict:
    args = sweep.parse_args(__doc__, argv, extra, default_variants=("raw_onehot", "softmax"))
    cfgs = [sweep.base_config(args, **g) for g in sweep.grid(dataset=args.datasets, variant=args.variants, model=args.models,
                                                            batch_size=args.batch_sizes, seed=args.seeds)]
    return sweep.run_experiment(args, cfgs, report, ["dataset", "variant", "model", "batch_size", "seed"])


if __name__ == "__main__":
    main()
