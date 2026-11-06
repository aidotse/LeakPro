#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""FedAvg leakage: local epochs x client batch size (client dataset capped).

The attacker sees the update of one client holding ``--client-cap`` rows (default 32) after ``local_epochs`` epochs of
mini-batch SGD, so local steps = local_epochs * ceil(cap / batch_size). Try ``--client-cap 16`` and ``64`` for the
sensitivity study. ``--quick`` uses a synthetic table.
"""
from __future__ import annotations

import numpy as np

from examples.gia.common import tabular_sweep as sweep


def extra(p) -> None:
    p.add_argument("--client-cap", type=int, default=32)
    p.add_argument("--local-epochs", nargs="+", type=int, default=[1, 4])
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 16, 32])
    p.add_argument("--meta-lr", type=float, default=0.01, help="Client learning rate.")


def report(df, out) -> None:
    df["local_steps"] = df["local_epochs"] * np.ceil(df["client_cap"] / df["batch_size"])
    sweep.plot_lines(df, out / "fedavg.png", x="batch_size", hue="local_epochs",
                     metrics=["tableak_acc", "exact_row_recovery"], title="FedAvg leakage: local epochs vs batch size")
    print(sweep.summary(df, ["dataset", "variant", "local_epochs", "batch_size", "local_steps"],
                        ["tableak_acc", "exact_row_recovery", "prior_accuracy", "gain_over_prior"]).to_string())


def main(argv: list[str] | None = None) -> dict:
    args = sweep.parse_args(__doc__, argv, extra)
    cfgs = [sweep.base_config(args, **g, protocol="fedavg", client_cap=args.client_cap, meta_lr=args.meta_lr)
            for g in sweep.grid(dataset=args.datasets, variant=args.variants, local_epochs=args.local_epochs,
                                batch_size=args.batch_sizes, seed=args.seeds)]
    return sweep.run_experiment(args, cfgs, report, ["dataset", "variant", "local_epochs", "batch_size", "seed"])


if __name__ == "__main__":
    main()
