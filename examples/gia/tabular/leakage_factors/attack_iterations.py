#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Does a larger attack budget help? Leakage vs attack iterations at two batch sizes (FedSGD).

Open datasets (Adult, California Housing) are downloaded on first use; ``--quick`` uses a synthetic table.
"""
from __future__ import annotations

from examples.gia.common import tabular_sweep as sweep


def extra(p) -> None:
    p.add_argument("--iterations", nargs="+", type=int, default=[1000, 3000, 10000])
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 32])


def report(df, out) -> None:
    sweep.plot_lines(df, out / "attack_iterations.png", x="max_iter", hue="batch_size", xlog=True,
                     metrics=["tableak_acc", "exact_row_recovery"], title="Leakage vs attack iterations")
    print(sweep.summary(df, ["dataset", "variant", "max_iter", "batch_size"],
                        ["tableak_acc", "exact_row_recovery", "prior_accuracy", "attack_seconds"]).to_string())


def main(argv: list[str] | None = None) -> dict:
    args = sweep.parse_args(__doc__, argv, extra)
    its = [20, 40, 60] if args.quick else args.iterations
    cfgs = [sweep.base_config(args, dataset=g["dataset"], variant=g["variant"], max_iter=g["iters"], batch_size=g["batch_size"],
                              seed=g["seed"])
            for g in sweep.grid(dataset=args.datasets, variant=args.variants, iters=its, batch_size=args.batch_sizes,
                                seed=args.seeds)]
    return sweep.run_experiment(args, cfgs, report, ["dataset", "variant", "max_iter", "batch_size", "seed"])


if __name__ == "__main__":
    main()
