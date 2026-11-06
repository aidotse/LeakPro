#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Does a generative image prior help? Inverting Gradients against GAN-constrained attacks (GIAS, GIFD, GGL).

The GAN attacks search the generator's latent space instead of pixel space. Full runs use the pretrained CIFAR-10
DCGAN ``csinva/cifar10_dcgan`` (one download) or a local checkpoint via ``--generator-path``; ``--quick`` uses a
random-weight tiny generator and synthetic images, so it checks the pipeline, not the quality.
"""
from __future__ import annotations

from examples.gia.common import cli, models, runner
from leakpro.attacks.gia_attacks.modular.presets import ggl, gias, gifd, inverting_gradients
from leakpro.fl_utils.gan_models import LayeredGANWrapper

ATTACKS = ("inverting_gradients", "gias", "gifd", "ggl")


def add_args(p) -> None:
    p.add_argument("--attacks", nargs="+", default=list(ATTACKS), choices=ATTACKS)
    p.add_argument("--generator-path", default=None, help="Local copy of the DCGAN weights instead of downloading them from GitHub.")
    p.add_argument("--github-model", default="csinva/cifar10_dcgan")
    p.add_argument("--latent-dim", type=int, default=100, help="Latent size of a local generator (the GitHub DCGANs need 100; quick: 16).")


def load_generator(a, device):  # noqa: ANN001, ANN201
    """Tiny random generator for ``--quick``; otherwise the pretrained DCGAN. Returns ``(generator, latent_dim)``."""
    if a.quick:
        return models.tiny_dcgan_generator(16).to(device), 16
    from leakpro.fl_utils.gan_handler import load_pretrained_gan  # noqa: PLC0415

    if a.generator_path:
        gen = load_pretrained_gan(checkpoint_path=a.generator_path, device=device, architecture="dcgan", latent_dim=a.latent_dim)
    else:
        gen = load_pretrained_gan(github_model=a.github_model, device=device, latent_dim=a.latent_dim)
    return gen, a.latent_dim


def main(argv: list[str] | None = None) -> dict:
    """Attack one client batch with the selected attacks and report per-image PSNR/SSIM."""
    a = cli.parse(__doc__, argv, extra=add_args)
    if a.replot:
        runner.replot_grid(a.out)
        return {}
    batch, model, device = runner.setup(a)
    generator, latent = load_generator(a, device)
    g = {"mean": batch.mean, "std": batch.std, "device": device}
    full = not a.quick
    factories = {
        "inverting_gradients": lambda: inverting_gradients(**g, max_iter=runner.iters(a, 4000)),
        "gias": lambda: gias(generator, latent, **g, num_classes=batch.num_classes,
                             stage1_max_iter=runner.iters(a, 1000, 5), stage2_max_iter=runner.iters(a, 500, 5)),
        "gifd": lambda: gifd(LayeredGANWrapper(generator), latent, **g, num_classes=batch.num_classes,
                             stage_max_iter=runner.iters(a, 500, 3)),
        "ggl": lambda: ggl(generator, latent, **g, num_classes=batch.num_classes,
                           population_size=32 if full else 6, max_generations=runner.iters(a, 200, 5)),
    }
    rows, recons = {}, {}
    for name in a.attacks:
        evaluation, _, _ = runner.run(factories[name](), model, batch, device, seed=a.seed)
        rows[name] = {m: list(evaluation.extras[f"{m}_per_image"]) for m in ("psnr", "ssim")}
        recons[name] = evaluation.extras["matched_recon"]
        print(f"{name}: PSNR {sum(rows[name]['psnr']) / len(rows[name]['psnr']):.2f} dB")
    runner.save_grid(a.out, batch.images, recons, batch.labels, rows)
    return runner.finish(a, rows, attacks=a.attacks, generator=a.generator_path or a.github_model)


if __name__ == "__main__":
    main()
