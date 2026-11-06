#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Utilities for loading and managing GAN models for gradient inversion attacks.

This module provides functions to load pre-trained GAN generators and
prepare them for use in latent-space gradient inversion attacks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


def load_pretrained_gan(
    checkpoint_path: str | Path | None = None,
    device: torch.device | str = "cpu",
    architecture: str = "stylegan2",
    github_model: str | None = None,
    **kwargs: Any,  # noqa: ANN401 -- genuinely heterogeneous value
) -> nn.Module:
    """Load a pre-trained GAN generator from a local checkpoint or from GitHub.

    Supports:
    - Local checkpoints (StyleGAN2, DCGAN, custom formats). ``architecture="dcgan"`` expects the DCGAN
      layout of ``gan-vae-pretrained-pytorch``, so a downloaded ``netG_epoch_*.pth`` loads from disk.
    - DCGAN weights published in the GitHub repository ``{user}/gan-vae-pretrained-pytorch``
      (e.g. csinva/gan-vae-pretrained-pytorch)

    DCGAN and custom weights are loaded with ``torch.load(weights_only=True)``: a file holding anything but
    tensors is rejected rather than unpickled.

    Args:
        checkpoint_path: Path to local checkpoint file (optional if using github_model)
        device: Device to load model on
        architecture: GAN architecture type ("stylegan2", "dcgan", "custom"; "github" with github_model)
        github_model: GitHub model ID ``user/model_name`` of a DCGAN in ``user/gan-vae-pretrained-pytorch``.
            Available models:
            - csinva/cifar10_dcgan (32x32, 3 channels)
            - csinva/mnist_dcgan (1 channel)
        **kwargs: Additional architecture-specific arguments
            - latent_dim: Latent dimension (default: 100; the GitHub DCGANs only exist with 100)
            - img_channels, feature_maps: DCGAN output channels (default 3) and base width (default 64)
            - epoch: Model epoch for GitHub models (default: model-specific)
            - trust_checkpoint: StyleGAN2 only. Its checkpoints hold pickled modules, which ``torch.load``
              can only restore by running arbitrary code from the file; pass ``True`` only for a file you trust.

    Returns:
        generator: Loaded generator in eval mode

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        ValueError: If architecture or GitHub model is unknown, both checkpoint_path and github_model are None,
            or a StyleGAN2 checkpoint is loaded without ``trust_checkpoint=True``
        RuntimeError: If the download fails

    """
    device = torch.device(device) if isinstance(device, str) else device

    if github_model is not None or architecture == "github":
        if github_model is None:
            raise ValueError("github_model must be specified when architecture='github'")
        return _load_github_dcgan(github_model, device, **kwargs)

    # Local checkpoint loading
    if checkpoint_path is None:
        raise ValueError("Either checkpoint_path or github_model must be provided")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"GAN checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading GAN generator from {checkpoint_path}")
    logger.info(f"  Architecture: {architecture}")
    logger.info(f"  Device: {device}")

    if architecture == "stylegan2":
        generator = _load_stylegan2(checkpoint_path, device, **kwargs)
    elif architecture == "dcgan":
        generator = _load_dcgan(checkpoint_path, device, **kwargs)
    elif architecture == "custom":
        generator = _load_custom(checkpoint_path, device, **kwargs)
    else:
        raise ValueError(f"Unknown GAN architecture: {architecture}")

    # Set to eval mode and move to device
    generator.eval()
    generator = generator.to(device)

    logger.info("GAN generator loaded")
    return generator


def _load_stylegan2(
    checkpoint_path: Path,
    device: torch.device,
    trust_checkpoint: bool = False,
    **kwargs: Any,  # noqa: ANN401, ARG001 -- genuinely heterogeneous value; kwargs accepted for a uniform loader signature
) -> nn.Module:
    """Load StyleGAN2 generator.

    Args:
        checkpoint_path: Path to checkpoint
        device: Target device
        trust_checkpoint: Confirms the file is trusted. StyleGAN2 checkpoints hold pickled modules, and
            unpickling them runs code from the file.
        **kwargs: Additional arguments

    Returns:
        StyleGAN2 generator

    Raises:
        ValueError: Without ``trust_checkpoint=True``.

    """
    if not trust_checkpoint:
        raise ValueError(
            "StyleGAN2 checkpoints hold pickled modules, and loading them runs code from the file. "
            "Pass trust_checkpoint=True to load_pretrained_gan only for a checkpoint you trust."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract generator (checkpoint format varies)
    if "generator" in checkpoint:
        generator = checkpoint["generator"]
    elif "g_ema" in checkpoint:
        generator = checkpoint["g_ema"]
    elif isinstance(checkpoint, nn.Module):
        generator = checkpoint
    else:
        raise ValueError(
            f"Could not find generator in checkpoint. "
            f"Available keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'N/A'}"
        )

    return generator


def _load_dcgan(
    checkpoint_path: Path,
    device: torch.device,
    latent_dim: int = 100,
    img_channels: int = 3,
    feature_maps: int = 64,
    **kwargs: Any,  # noqa: ANN401, ARG001 -- genuinely heterogeneous value; kwargs accepted for a uniform loader signature
) -> nn.Module:
    """Load a local DCGAN generator in the ``gan-vae-pretrained-pytorch`` layout (see ``_build_dcgan_generator``).

    Args:
        checkpoint_path: Path to a state dict (optionally under ``"generator"`` or ``"state_dict"``).
        device: Target device
        latent_dim: Latent dimension
        img_channels: Number of image channels
        feature_maps: Base number of feature maps
        **kwargs: Unused; accepted for a uniform loader signature

    Returns:
        DCGAN generator

    """
    generator = _build_dcgan_generator(nz=latent_dim, nc=img_channels, ngf=feature_maps)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    for key in ("generator", "state_dict"):
        if key in checkpoint:
            checkpoint = checkpoint[key]
            break
    generator.load_state_dict(_strip_main_prefix(checkpoint))
    return generator


def _strip_main_prefix(state: dict) -> dict:
    """Drop the ``main.`` prefix the original training script's ``Generator`` class puts on every key."""
    return {(k[len("main."):] if k.startswith("main.") else k): v for k, v in state.items()}


def _load_custom(
    checkpoint_path: Path,
    device: torch.device,
    model_class: type[nn.Module] | None = None,
    **kwargs: Any,  # noqa: ANN401 -- genuinely heterogeneous value
) -> nn.Module:
    """Load custom GAN generator.

    Args:
        checkpoint_path: Path to a state dict (optionally under ``"generator"`` or ``"state_dict"``).
        device: Target device
        model_class: Generator class to instantiate
        **kwargs: Arguments for model constructor

    Returns:
        Custom generator

    Raises:
        ValueError: If model_class not provided

    """
    if model_class is None:
        raise ValueError("model_class must be provided for custom architecture")
    generator = model_class(**kwargs)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    for key in ("generator", "state_dict"):
        if key in checkpoint:
            checkpoint = checkpoint[key]
            break
    generator.load_state_dict(checkpoint)
    return generator


#: DCGAN models of gan-vae-pretrained-pytorch: model name -> (output channels, default weights epoch).
_GITHUB_DCGANS: dict[str, tuple[int, int]] = {
    "cifar10_dcgan": (3, 199),
    "mnist_dcgan": (1, 99),
}


def _load_github_dcgan(
    model_id: str,
    device: torch.device,
    **kwargs: Any,  # noqa: ANN401 -- genuinely heterogeneous value
) -> nn.Module:
    """Load a DCGAN generator from GitHub (``{user}/gan-vae-pretrained-pytorch``).

    Model format: github_username/model_name (e.g., "csinva/cifar10_dcgan"); see ``_GITHUB_DCGANS``.

    Args:
        model_id: GitHub model ID (e.g., "csinva/cifar10_dcgan")
        device: Device to load model on
        **kwargs: Additional arguments (``epoch``; ``latent_dim`` must be 100 if given)

    Returns:
        generator: Loaded generator in eval mode

    Raises:
        ValueError: If the model ID is malformed, the model is not a known DCGAN, or ``latent_dim`` is not 100
        RuntimeError: If the download fails

    """
    import os
    import tempfile
    import urllib.request

    if kwargs.get("latent_dim", 100) != 100:
        raise ValueError(f"the gan-vae-pretrained-pytorch DCGANs have latent_dim=100, got latent_dim={kwargs['latent_dim']}")

    logger.info(f"Loading GAN from GitHub: {model_id}")

    # Parse model_id: expected format "csinva/cifar10_dcgan" or similar
    parts = model_id.split("/")
    if len(parts) != 2:
        raise ValueError(f"Model ID should be in format 'username/model_name', got: {model_id}")

    username, model_name = parts
    if model_name not in _GITHUB_DCGANS:
        raise ValueError(f"Unknown GitHub DCGAN {model_name!r}; known: {', '.join(sorted(_GITHUB_DCGANS))}")
    channels, default_epoch = _GITHUB_DCGANS[model_name]
    epoch = kwargs.get("epoch", default_epoch)

    # Construct GitHub raw URL for weights
    base_url = f"https://raw.githubusercontent.com/{username}/gan-vae-pretrained-pytorch/master/{model_name}/weights"
    netg_filename = f"netG_epoch_{epoch}.pth"
    netg_url = f"{base_url}/{netg_filename}"

    logger.info(f"  Downloading from: {netg_url}")

    # Download the weights to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp:
        try:
            urllib.request.urlretrieve(netg_url, tmp.name)  # noqa: S310 -- https URL built from a fixed host
            checkpoint_path = tmp.name
        except Exception as e:
            raise RuntimeError(f"Failed to download model from {netg_url}: {e}") from e

    try:
        # The DCGAN architecture of gan-vae-pretrained-pytorch: nz=100, ngf=64.
        generator = _build_dcgan_generator(nz=100, nc=channels, ngf=64)

        # The file is a state dict: load tensors only, never arbitrary pickled objects.
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

        generator.load_state_dict(_strip_main_prefix(checkpoint))

        generator = generator.to(device)
        generator.eval()

        logger.info(f"Loaded generator from {model_id}")
        return generator

    finally:
        # Clean up temporary file
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)


class _FlatLatentSequential(nn.Sequential):
    """DCGAN ``nn.Sequential`` that also accepts a flat ``[N, nz]`` latent (the csinva net wants ``[N, nz, 1, 1]``)."""

    def forward(self, z: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:  # noqa: ARG002 -- generator interface
        """Generate images from ``[N, nz]`` or ``[N, nz, 1, 1]`` latents."""
        if z.ndim == 2:
            z = z[:, :, None, None]
        return super().forward(z)


def _build_dcgan_generator(nz: int = 100, nc: int = 3, ngf: int = 64) -> nn.Module:
    """Build the 32x32 DCGAN generator of ``gan-vae-pretrained-pytorch`` (the csinva CIFAR-10/MNIST weights).

    Args:
        nz: Latent dimension
        nc: Number of output channels
        ngf: Base number of features

    Returns:
        Generator module

    """
    return _FlatLatentSequential(
        # input is Z, going into a convolution
        nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(ngf * 8),
        nn.ReLU(True),
        # state size. (ngf*8) x 4 x 4
        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 4),
        nn.ReLU(True),
        # state size. (ngf*4) x 8 x 8
        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(True),
        # state size. (ngf*2) x 16 x 16
        nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True),
        # state size. (ngf) x 32 x 32
        nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=0, bias=False),
        nn.Tanh()
        # state size. (nc) x 32 x 32
    )
