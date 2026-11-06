#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""What does knowing the BatchNorm statistics buy? Running vs proxy-estimated statistics, with the client in train or eval mode."""
from __future__ import annotations

from examples.gia.common import cli, runner
from leakpro.attacks.gia_attacks.modular.presets import gia_estimate, gia_running, huang, inverting_gradients


def add_args(p) -> None:
    p.add_argument("--model-mode", choices=("train", "eval"), default="train", help="Mode of the client's model while it trains.")
    p.add_argument("--seeds", type=int, default=None, help="Attack restarts per attack (default 3, quick 1).")


def main(argv: list[str] | None = None) -> dict:
    """Compare an attack without BN knowledge to the three that use BN statistics."""
    a = cli.parse(__doc__, argv, extra=add_args)
    if a.replot:
        runner.replot_grid(a.out)
        return {}
    batch, model, device = runner.setup(a, mode=a.model_mode)
    n, mm, args = runner.iters(a, 4000), a.model_mode, (batch.mean, batch.std, device)
    attacks = {
        "No BN knowledge (Inverting Gradients)": lambda: inverting_gradients(*args, max_iter=n, model_mode=mm),
        "Huang (client running statistics)": lambda: huang(*args, max_iter=n, model_mode=mm),
        "GIA Running": lambda: gia_running(*args, max_iter=n, model_mode=mm),
        "GIA Estimate (proxy data)": lambda: gia_estimate(*args, max_iter=n, model_mode=mm),
    }
    if mm == "eval":  # an eval-mode client leaves its running statistics unchanged: nothing for GIA Running to invert
        del attacks["GIA Running"]
    proxy = runner.proxy_loader(a, batch, device)
    rows, recons = {}, {}
    for name, make in attacks.items():
        per_image = {"psnr": [], "ssim": []}
        for s in range(a.seeds or (1 if a.quick else 3)):
            evaluation, _, _ = runner.run(make(), model, batch, device, seed=a.seed + s, proxy_dataloader=proxy)
            for m in per_image:
                per_image[m] += evaluation.extras[f"{m}_per_image"]
        rows[name], recons[name] = per_image, evaluation.extras["matched_recon"]
        print(f"{name}: PSNR {sum(per_image['psnr']) / len(per_image['psnr']):.2f} dB")
    runner.save_grid(a.out, batch.images, recons, batch.labels, rows)
    return runner.finish(a, rows, model_mode=mm)


if __name__ == "__main__":
    main()
