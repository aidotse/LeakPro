#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Which architecture factors drive tabular leakage? Width x depth x normalisation x dropout x activation (FedSGD, MLPs).

Reduced 48-combination grid by default; prints the per-factor marginal means. Open datasets (Adult, California Housing)
are downloaded on first use; ``--quick`` uses a synthetic table.
"""
from __future__ import annotations

from examples.gia.common import tabular_sweep as sweep

FACTORS = ["d_hidden", "n_layers", "norm", "dropout", "activation"]


def extra(p) -> None:
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--d-hidden", nargs="+", type=int, default=[32, 128])
    p.add_argument("--n-layers", nargs="+", type=int, default=[1, 3])
    p.add_argument("--norms", nargs="+", default=["layernorm", "batchnorm", "none"])
    p.add_argument("--dropouts", nargs="+", type=float, default=[0.0, 0.1])
    p.add_argument("--activations", nargs="+", default=["relu", "gelu"])


def report(df, out) -> None:
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    datasets = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(len(datasets), len(FACTORS), figsize=(3.2 * len(FACTORS), 3 * len(datasets)), squeeze=False)
    for r, ds in enumerate(datasets):
        sub = df[df["dataset"] == ds]
        for c, factor in enumerate(FACTORS):
            a = sweep.aggregate(sub, [factor], ["tableak_acc"])
            axes[r][c].bar(a[factor].astype(str), a["tableak_acc_mean"], yerr=a["tableak_acc_ci95"], capsize=3)
            axes[r][c].set_title(f"{ds}: {factor}", fontsize=9)
            axes[r][c].set_ylim(0, 1)
            print(f"\n{ds} marginal over {factor}:\n{a.to_string(index=False)}")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "architecture_factors.png", dpi=200)
    print(f"Saved plot: {out / 'architecture_factors.png'}")


def main(argv: list[str] | None = None) -> dict:
    args = sweep.parse_args(__doc__, argv, extra)
    if args.quick:
        args.d_hidden, args.n_layers, args.norms, args.dropouts, args.activations = [32], [1], ["layernorm", "none"], [0.0], ["relu"]
    cfgs = []
    for g in sweep.grid(dataset=args.datasets, variant=args.variants, d_hidden=args.d_hidden, n_layers=args.n_layers,
                        norm=args.norms, dropout=args.dropouts, activation=args.activations, seed=args.seeds):
        cfgs.append(sweep.base_config(args, **g, model={k: g[k] for k in FACTORS}, batch_size=args.batch_size))
    return sweep.run_experiment(args, cfgs, report, ["dataset", "variant", *FACTORS, "seed"])


if __name__ == "__main__":
    main()
