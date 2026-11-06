#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""How does reconstruction quality fall with the batch size (FedSGD) and rise or fall with the number of local epochs (FedAvg)?"""
from __future__ import annotations

from examples.gia.common import cli, results, runner, viz
from leakpro.attacks.gia_attacks.modular.presets import dimitrov, inverting_gradients


def add_args(p) -> None:
    p.add_argument("--sweep", choices=("batch", "epochs"), default="batch")
    p.add_argument("--values", type=int, nargs="+", default=None, help="Sweep values (default 1..16 batch, 1..5 epochs).")


def _plot(out, sweep: str, rows: dict) -> None:
    xs = [int(k.split("=")[1]) for k in rows]
    series = [results.mean(rows[k]["psnr"]) for k in rows]
    viz.plot_curves(xs, {"PSNR (dB)": series}, sweep, "PSNR (dB)", out / f"{sweep}_psnr.png")


def main(argv: list[str] | None = None) -> dict:
    """One attack per sweep value; every row is the per-image metrics of that setting."""
    a = cli.parse(__doc__, argv, extra=add_args)
    if a.replot:
        _plot(a.out, a.sweep, results.load(a.out)["rows"])
        return {}
    default = {"batch": [1, 2, 4, 8, 16], "epochs": [1, 2, 3, 4, 5]}[a.sweep]
    values = a.values or ([1, 2, 3] if a.quick and a.sweep == "batch" else [1, 2] if a.quick else default)
    rows = {}
    for v in values:
        batch, model, device = runner.setup(a, num_images=v if a.sweep == "batch" else 2, mode="train" if a.sweep == "batch" else "eval",
                                          fixed_num_images=a.sweep == "batch")
        n = runner.iters(a, 2000)
        if a.sweep == "batch":
            attack = inverting_gradients(batch.mean, batch.std, device, max_iter=n)
        else:
            attack = dimitrov(batch.mean, batch.std, device, num_epochs=v, max_iter=n)
        evaluation, _, _ = runner.run(attack, model, batch, device, seed=a.seed)
        rows[f"{a.sweep}={v}"] = {m: evaluation.extras[f"{m}_per_image"] for m in ("psnr", "ssim")}
        print(f"{a.sweep}={v}: PSNR {evaluation.metrics['psnr']:.2f} dB")
    a.out.mkdir(parents=True, exist_ok=True)
    _plot(a.out, a.sweep, rows)
    return runner.finish(a, rows, sweep=a.sweep, values=values)


if __name__ == "__main__":
    main()
