#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Effect of the attacker not knowing the labels (FedSGD).

Label modes: ``known`` (oracle), ``zeros`` (the "label unknown" mode: all labels 0) and ``joint`` (labels
optimised alongside the data). Open datasets are downloaded on first use; ``--quick`` uses a synthetic table.
"""
from __future__ import annotations

from examples.gia.common import tabular_sweep as sweep


def extra(p) -> None:
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 32])
    p.add_argument("--label-modes", nargs="+", default=["known", "zeros", "joint"], choices=["known", "zeros", "joint"])


def report(df, out) -> None:
    sweep.plot_lines(df, out / "label_knowledge.png", x="label_mode", hue="batch_size",
                     metrics=["tableak_acc", "exact_row_recovery"], title="Leakage with known vs unknown labels")
    print(sweep.summary(df, ["dataset", "variant", "label_mode", "batch_size"],
                        ["tableak_acc", "exact_row_recovery", "prior_accuracy", "gain_over_prior"]).to_string())


def main(argv: list[str] | None = None) -> dict:
    args = sweep.parse_args(__doc__, argv, extra)
    cfgs = [sweep.base_config(args, **g) for g in sweep.grid(dataset=args.datasets, variant=args.variants,
                                                            label_mode=args.label_modes, batch_size=args.batch_sizes,
                                                            seed=args.seeds)]
    return sweep.run_experiment(args, cfgs, report, ["dataset", "variant", "label_mode", "batch_size", "seed"])


if __name__ == "__main__":
    main()
