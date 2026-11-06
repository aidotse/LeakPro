# Generative priors

Does a generative image prior help gradient inversion? Run from the repository root; results go to `examples/gia/outputs/<script>/`.

`--quick` is an offline CPU smoke run with a random-weight tiny generator or UNet on synthetic images. It checks that the pipeline runs, not the reconstruction quality.

## gan_priors
Inverting Gradients against the GAN-constrained attacks GIAS, GIFD and GGL, which search the generator's latent space.

- Quick: `python -m examples.gia.generative_priors.gan_priors --quick` (~10 s).
- Full: drop `--quick`. Downloads the pretrained CIFAR-10 DCGAN `csinva/cifar10_dcgan` once, or pass a local copy of its weights file (`netG_epoch_199.pth`) with `--generator-path`. GGL needs `pip install cma`.

<!-- results:gan_priors -->
| Attack | psnr | ssim |
|---|---|---|
| inverting_gradients | 30.85 ± 2.62 (n=4) | 0.93 ± 0.02 (n=4) |
| gias | 6.17 ± 1.21 (n=4) | 0.02 ± 0.01 (n=4) |
| gifd | 13.00 ± 2.24 (n=4) | 0.14 ± 0.11 (n=4) |
| ggl | 13.17 ± 1.87 (n=4) | 0.20 ± 0.07 (n=4) |

Command: `python -m examples.gia.generative_priors.gan_priors --out examples/gia/outputs/gan_priors --num-images 4`; torch `2.5.1+cu121`, Quadro RTX 5000.
<!-- /results -->

Highest PSNR: inverting_gradients (30.85 dB); lowest: gias (6.17 dB).

## diffusion_priors
Inverting Gradients against GGDM and GradInvDiff, which guide a pretrained DDPM with the gradient.

- Quick: `python -m examples.gia.generative_priors.diffusion_priors --quick` (~20 s). Needs `diffusers`.
- Full: drop `--quick`. Downloads `google/ddpm-cifar10-32` (CIFAR-10) or `google/ddpm-celebahq-256` (CelebA, attacked through a ResNet-18; the CelebA dataset must be downloaded manually). A GPU is recommended.

<!-- results:diffusion_priors -->
| Attack | psnr | ssim |
|---|---|---|
| inverting_gradients | 30.62 ± 2.54 (n=4) | 0.93 ± 0.02 (n=4) |
| ggdm | 7.74 ± 0.63 (n=4) | 0.01 ± 0.01 (n=4) |
| gradinvdiff | 9.99 ± 1.28 (n=4) | 0.03 ± 0.01 (n=4) |

Command: `python -m examples.gia.generative_priors.diffusion_priors --out examples/gia/outputs/diffusion_priors --dataset cifar10 --num-images 4`; torch `2.5.1+cu121`, Quadro RTX 5000.
<!-- /results -->

Highest PSNR: inverting_gradients (30.62 dB); lowest: ggdm (7.74 dB).
