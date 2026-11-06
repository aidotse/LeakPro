#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""FedAvg: how do the number of local epochs and the client's batch size change what Dimitrov's attack recovers?"""
from __future__ import annotations

from examples.gia.common import cli, runner
from leakpro.attacks.gia_attacks.modular.presets import dimitrov


def add_args(p) -> None:
    p.add_argument("--epochs", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--client-batch-sizes", type=int, nargs="+", default=[1, 2])


def main(argv: list[str] | None = None) -> dict:
    """Invert the weight update of every (epochs, client batch size) setting of the grid."""
    a = cli.parse(__doc__, argv, extra=add_args)
    if a.replot:
        runner.replot_grid(a.out)
        return {}
    batch, model, device = runner.setup(a, num_images=2, mode="eval")
    rows, recons = {}, {}
    for epochs in a.epochs:
        for bs in a.client_batch_sizes:
            name = f"epochs={epochs}, client batch={bs}"
            attack = dimitrov(batch.mean, batch.std, device, num_epochs=epochs, client_batch_size=bs,
                              max_iter=runner.iters(a, 8000))
            evaluation, _, _ = runner.run(attack, model, batch, device, seed=a.seed)
            rows[name] = {m: evaluation.extras[f"{m}_per_image"] for m in ("psnr", "ssim")}
            recons[name] = evaluation.extras["matched_recon"]
            print(f"{name}: PSNR {sum(rows[name]['psnr']) / len(rows[name]['psnr']):.2f} dB")
    runner.save_grid(a.out, batch.images, recons, batch.labels, rows)
    return runner.finish(a, rows, epochs=a.epochs, client_batch_sizes=a.client_batch_sizes)


if __name__ == "__main__":
    main()
