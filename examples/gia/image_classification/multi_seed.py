#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Does See-Through-Gradients' multi-seed search with group consistency beat a single seed at the same budget?"""
from __future__ import annotations

import dataclasses

from examples.gia.common import cli, runner
from leakpro.attacks.gia_attacks.modular.components.aggregators import BestLossAggregator
from leakpro.attacks.gia_attacks.modular.presets import yin


def add_args(p) -> None:
    p.add_argument("--num-seeds", type=int, default=4)


def main(argv: list[str] | None = None) -> dict:
    """One seed with the whole step budget against several seeds sharing it, averaged or best-loss."""
    a = cli.parse(__doc__, argv, extra=add_args)
    if a.replot:
        runner.replot_grid(a.out)
        return {}
    batch, model, device = runner.setup(a)
    total, k = runner.iters(a, 8000, quick=40), a.num_seeds
    common = dict(lr=1.0, tv_weight=0.052, bn_weight=1.6e-4)
    args = (batch.mean, batch.std, device)
    best = yin(*args, num_seeds=k, max_iter=total // k, **common)
    best = dataclasses.replace(best, stages=[dataclasses.replace(best.stages[0], aggregator=BestLossAggregator())])
    attacks = {
        "1 seed": yin(*args, num_seeds=1, max_iter=total, **common),
        f"{k} seeds, averaged": yin(*args, num_seeds=k, max_iter=total // k, **common),
        f"{k} seeds, best loss": best,
    }
    rows, recons = {}, {}
    for name, attack in attacks.items():
        evaluation, _, _ = runner.run(attack, model, batch, device, seed=a.seed)
        rows[name] = {m: evaluation.extras[f"{m}_per_image"] for m in ("psnr", "ssim")}
        recons[name] = evaluation.extras["matched_recon"]
        print(f"{name}: PSNR {sum(rows[name]['psnr']) / len(rows[name]['psnr']):.2f} dB")
    runner.save_grid(a.out, batch.images, recons, batch.labels, rows)
    return runner.finish(a, rows, num_seeds=k, total_iterations=total)


if __name__ == "__main__":
    main()
