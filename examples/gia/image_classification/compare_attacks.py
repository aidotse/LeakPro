#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Which classic attack recovers images best from one gradient? DLG, iDLG, Inverting Gradients, Huang, Yin, GIA Running and Estimate."""
from __future__ import annotations

from examples.gia.common import cli, runner
from leakpro.attacks.gia_attacks.modular.presets import (
    dlg, gia_estimate, gia_running, huang, idlg, inverting_gradients, yin,
)

# name -> (factory, full-run iterations, labels known to the attacker)
ATTACKS = {
    "DLG": (lambda s, n: dlg(s.mean, s.std, s.device, s.num_classes, max_iter=n), 300, False),
    "iDLG": (lambda s, n: idlg(s.mean, s.std, s.device, max_iter=n), 300, False),
    "Inverting Gradients": (lambda s, n: inverting_gradients(s.mean, s.std, s.device, max_iter=n), 4000, True),
    "Huang": (lambda s, n: huang(s.mean, s.std, s.device, max_iter=n), 4000, True),
    "Yin": (lambda s, n: yin(s.mean, s.std, s.device, max_iter=n), 4000, True),
    "GIA Running": (lambda s, n: gia_running(s.mean, s.std, s.device, max_iter=n), 4000, True),
    "GIA Estimate": (lambda s, n: gia_estimate(s.mean, s.std, s.device, max_iter=n), 4000, True),
}


class _Setup:
    """The statistics and device the preset factories need."""

    def __init__(self, batch, device):
        self.mean, self.std, self.num_classes, self.device = batch.mean, batch.std, batch.num_classes, device


def add_args(p) -> None:
    p.add_argument("--attacks", nargs="+", default=list(ATTACKS), choices=list(ATTACKS))
    p.add_argument("--seeds", type=int, default=None, help="Attack restarts per attack (default 3, quick 1).")


def main(argv: list[str] | None = None) -> dict:
    """Attack one client batch with every selected preset and report per-image PSNR/SSIM."""
    a = cli.parse(__doc__, argv, extra=add_args)
    if a.replot:
        runner.replot_grid(a.out)
        return {}
    batch, model, device = runner.setup(a)
    setup = _Setup(batch, device)
    proxy = runner.proxy_loader(a, batch, device)
    rows, recons = {}, {}
    for name in a.attacks:
        factory, full_iters, oracle = ATTACKS[name]
        per_image = {"psnr": [], "ssim": []}
        for s in range(a.seeds or (1 if a.quick else 3)):
            attack = factory(setup, runner.iters(a, full_iters))
            evaluation, _, _ = runner.run(attack, model, batch, device, oracle=oracle, seed=a.seed + s, proxy_dataloader=proxy)
            for m in per_image:
                per_image[m] += evaluation.extras[f"{m}_per_image"]
        rows[name] = per_image
        recons[name] = evaluation.extras["matched_recon"]
        print(f"{name}: PSNR {sum(per_image['psnr']) / len(per_image['psnr']):.2f} dB")
    runner.save_grid(a.out, batch.images, recons, batch.labels, rows)
    return runner.finish(a, rows, attacks=a.attacks)


if __name__ == "__main__":
    main()
