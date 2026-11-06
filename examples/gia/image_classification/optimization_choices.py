#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""One-factor ablation of the optimisation choices behind Inverting Gradients: pixel space, gradient sign, schedule, initial image."""
from __future__ import annotations

import dataclasses

from examples.gia.common import cli, runner
from leakpro.attacks.gia_attacks.modular.components.drivers import LinearDecay, MultiStep, SignGradient
from leakpro.attacks.gia_attacks.modular.components.initializers import Gaussian, Uniform
from leakpro.attacks.gia_attacks.modular.components.representations import PixelRepresentation
from leakpro.attacks.gia_attacks.modular.presets import inverting_gradients

FACTORS = ("space", "grad_sign", "schedule", "initializer")


def _variants(factor: str, args: tuple, n: int) -> dict:
    ig = lambda **kw: inverting_gradients(*args, max_iter=n, **kw)  # noqa: E731
    if factor == "space":
        return {"optimise model input": ig(space="model_input"), "optimise data space": ig(space="data")}
    if factor == "grad_sign":
        return {"signed gradient": ig(grad_transforms=(SignGradient(),)), "raw gradient": ig(grad_transforms=())}
    if factor == "schedule":
        return {"no schedule": ig(schedule=None), "step decay": ig(schedule=MultiStep()), "linear decay": ig(schedule=LinearDecay())}
    # Both starts optimise in the preset's model-input space; only the first iterate differs (U(0, 1) pixels or N(0, 1) inputs).
    base = ig()
    swap = lambda init: dataclasses.replace(  # noqa: E731
        base, stages=[dataclasses.replace(base.stages[0], rep=PixelRepresentation(0.0, 1.0, space="model_input", init=init))])
    return {"uniform start": swap(Uniform()), "gaussian start": swap(Gaussian())}


def add_args(p) -> None:
    p.add_argument("--factor", choices=(*FACTORS, "all"), default="all")
    p.add_argument("--seeds", type=int, default=None, help="Attack restarts per variant (default 3, quick 1).")


def main(argv: list[str] | None = None) -> dict:
    """Change one factor at a time and compare per-image PSNR/SSIM."""
    a = cli.parse(__doc__, argv, extra=add_args)
    if a.replot:
        runner.replot_grid(a.out)
        return {}
    batch, model, device = runner.setup(a, mode="eval")
    rows, recons = {}, {}
    for factor in (FACTORS if a.factor == "all" else (a.factor,)):
        for name, attack in _variants(factor, (batch.mean, batch.std, device), runner.iters(a, 2000)).items():
            per_image = {"psnr": [], "ssim": []}
            for s in range(a.seeds or (1 if a.quick else 3)):
                evaluation, _, _ = runner.run(attack, model, batch, device, seed=a.seed + s)
                for m in per_image:
                    per_image[m] += evaluation.extras[f"{m}_per_image"]
            key = f"{factor}: {name}"
            rows[key], recons[key] = per_image, evaluation.extras["matched_recon"]
            print(f"{key}: PSNR {sum(per_image['psnr']) / len(per_image['psnr']):.2f} dB")
    runner.save_grid(a.out, batch.images, recons, batch.labels, rows)
    return runner.finish(a, rows, factor=a.factor)


if __name__ == "__main__":
    main()
