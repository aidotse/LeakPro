#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Does a diffusion prior help? Inverting Gradients against GGDM and GradInvDiff (a pretrained DDPM guided by the gradient).

Full runs use ``google/ddpm-cifar10-32`` (CIFAR-10) or ``google/ddpm-celebahq-256`` (CelebA, images upsampled to 256 and
attacked through a ResNet-18); ``--quick`` uses a random-weight tiny UNet on synthetic images, so it checks the
pipeline, not the quality.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

from examples.gia.common import cli, data, models, runner
from leakpro.attacks.gia_attacks.modular.presets import ggdm, gradinvdiff, inverting_gradients

ATTACKS = ("inverting_gradients", "ggdm", "gradinvdiff")
DDPM = {"cifar10": "google/ddpm-cifar10-32", "celeba": "google/ddpm-celebahq-256"}


def add_args(p) -> None:
    p.add_argument("--attacks", nargs="+", default=list(ATTACKS), choices=ATTACKS)
    p.add_argument("--dataset", choices=("cifar10", "celeba"), default="cifar10")
    p.add_argument("--steps", type=int, default=None, help="Reverse-diffusion steps (default 100, quick 5).")


def _setup(a):  # noqa: ANN001, ANN202
    """Batch, victim model, device and the (unet, scheduler) pair."""
    if a.quick:
        from diffusers import DDPMScheduler  # noqa: PLC0415

        from leakpro.fl_utils.diffusion_handler import DDPMSchedulerWrapper  # noqa: PLC0415

        batch, model, device = runner.setup(a)
        return batch, model, device, models.tiny_unet().to(device), DDPMSchedulerWrapper(DDPMScheduler(num_train_timesteps=20), device)
    from leakpro.fl_utils.diffusion_handler import load_pretrained_ddpm  # noqa: PLC0415

    if a.dataset == "cifar10":
        batch, model, device = runner.setup(a)
    else:  # CelebA: 64 px crops upsampled to the DDPM's 256 px, attacked through a ResNet-18
        import torchvision  # noqa: PLC0415

        device = cli.device_for(a)
        torch.manual_seed(a.seed)
        batch = data.image_batch(False, a.num_images or 1, a.data_dir, device, dataset="celeba")
        batch.images = F.interpolate(batch.images, size=256, mode="bilinear", align_corners=False).clamp(0, 1)
        model = torchvision.models.resnet18(num_classes=batch.num_classes).to(device).train()
    unet, scheduler = load_pretrained_ddpm(DDPM[a.dataset], device)
    return batch, model, device, unet, scheduler


def main(argv: list[str] | None = None) -> dict:
    """Attack one client batch with the selected attacks and report per-image PSNR/SSIM."""
    a = cli.parse(__doc__, argv, extra=add_args)
    if a.replot:
        runner.replot_grid(a.out)
        return {}
    batch, model, device, unet, scheduler = _setup(a)
    g = {"mean": batch.mean, "std": batch.std, "device": device, "num_classes": batch.num_classes}
    steps = a.steps or runner.iters(a, 100, 5)
    factories = {
        "inverting_gradients": lambda: inverting_gradients(g["mean"], g["std"], device, max_iter=runner.iters(a, 4000)),
        "ggdm": lambda: ggdm(unet, scheduler, **g, num_steps=steps),
        "gradinvdiff": lambda: gradinvdiff(unet, scheduler, **g, num_steps=steps, inner_steps=1 if a.quick else 5),
    }
    rows, recons = {}, {}
    for name in a.attacks:
        evaluation, _, _ = runner.run(factories[name](), model, batch, device, seed=a.seed)
        rows[name] = {m: list(evaluation.extras[f"{m}_per_image"]) for m in ("psnr", "ssim")}
        recons[name] = evaluation.extras["matched_recon"]
        print(f"{name}: PSNR {sum(rows[name]['psnr']) / len(rows[name]['psnr']):.2f} dB")
    runner.save_grid(a.out, batch.images, recons, batch.labels, rows)
    return runner.finish(a, rows, attacks=a.attacks, dataset="synthetic" if a.quick else a.dataset, steps=steps)


if __name__ == "__main__":
    main()
