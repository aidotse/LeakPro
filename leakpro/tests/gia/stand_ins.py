#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Weight-free stand-ins for the pretrained networks the generative attacks need."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from leakpro.fl_utils.diffusion_handler import DDPMSchedulerWrapper
from leakpro.fl_utils.gan_models import LayeredGANWrapper


class TinyGenerator(nn.Module):
    """Latent [B, latent_dim] -> image [B, C, S, S] in [-1, 1] via one ConvTranspose2d (a DCGAN in miniature)."""

    def __init__(self, latent_dim: int, image_shape: tuple[int, int, int]) -> None:
        super().__init__()
        channels, height, width = image_shape
        if height != width:
            raise ValueError("TinyGenerator produces square images")
        self.latent_dim = latent_dim
        self.net = nn.Sequential(nn.ConvTranspose2d(latent_dim, channels, kernel_size=height, stride=1, padding=0), nn.Tanh())

    def forward(self, z: Tensor, labels: Tensor | None = None) -> Tensor:  # noqa: ARG002
        return self.net(z.view(z.shape[0], self.latent_dim, 1, 1))


def tiny_layered_generator(latent_dim: int = 8, channels: int = 3) -> LayeredGANWrapper:
    """A two-group DCGAN-style generator for GIFD: latent ``[B, latent_dim]`` to ``[B, channels, 8, 8]`` via Tanh."""
    base = nn.Sequential(
        nn.ConvTranspose2d(latent_dim, 8, kernel_size=4, stride=1, padding=0),
        nn.ReLU(True),
        nn.ConvTranspose2d(8, channels, kernel_size=4, stride=2, padding=1),
        nn.Tanh(),
    )
    return LayeredGANWrapper(base)


class TinyClassifier(nn.Module):
    """Flatten + linear head; the cheapest differentiable victim model."""

    def __init__(self, image_shape: tuple[int, int, int], num_classes: int = 2) -> None:
        super().__init__()
        channels, height, width = image_shape
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(channels * height * width, num_classes))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _UNetOutput:
    def __init__(self, sample: Tensor) -> None:
        self.sample = sample


class ZeroUNet(nn.Module):
    """Always predicts zero noise, mimicking the diffusers UNet call signature and output."""

    def forward(self, x: Tensor, t: Tensor) -> _UNetOutput:  # noqa: ARG002
        return _UNetOutput(torch.zeros_like(x))


def tiny_ddpm_scheduler(num_train_timesteps: int = 4) -> DDPMSchedulerWrapper:
    """A real diffusers DDPM scheduler with very few timesteps (requires diffusers)."""
    from diffusers import DDPMScheduler

    return DDPMSchedulerWrapper(DDPMScheduler(num_train_timesteps=num_train_timesteps), device=torch.device("cpu"))
