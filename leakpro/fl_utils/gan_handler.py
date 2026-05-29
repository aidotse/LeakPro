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
    huggingface_model: str | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Load a pre-trained GAN generator from checkpoint or GitHub.

    Supports:
    - Local checkpoints (StyleGAN2, DCGAN formats)
    - GitHub-hosted models (csinva/gan-vae-pretrained-pytorch)

    Args:
        checkpoint_path: Path to local checkpoint file (optional if using huggingface_model)
        device: Device to load model on
        architecture: GAN architecture type ("stylegan2", "dcgan", "huggingface", "custom")
        huggingface_model: GitHub model ID (e.g., "csinva/cifar10_dcgan")
            Available models:
            - csinva/cifar10_dcgan (32x32, 3 channels)
            - csinva/mnist_dcgan (28x28, 1 channel)
            - csinva/cifar100_dcgan (32x32, 1 channel)
        **kwargs: Additional architecture-specific arguments
            - latent_dim: Latent dimension (default: 100)
            - epoch: Model epoch for GitHub models (default: dataset-specific)

    Returns:
        generator: Loaded generator in eval mode

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        ValueError: If architecture is unknown or both checkpoint_path and huggingface_model are None
        RuntimeError: If model loading fails
    """
    device = torch.device(device) if isinstance(device, str) else device

    # HuggingFace loading
    if huggingface_model is not None or architecture == "huggingface":
        if huggingface_model is None:
            raise ValueError("huggingface_model must be specified when architecture='huggingface'")
        return _load_huggingface_gan(huggingface_model, device, **kwargs)

    # BigGAN: auto-downloads via pytorch_pretrained_biggan; no checkpoint needed
    if architecture == "biggan":
        return _load_biggan_wrapper(device, truncation=kwargs.pop("truncation", 0.4), **kwargs)

    # Local checkpoint loading
    if checkpoint_path is None:
        raise ValueError("Either checkpoint_path or huggingface_model must be provided")
    
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

    logger.info(f"✓ GAN generator loaded successfully")
    return generator


def _load_stylegan2(
    checkpoint_path: Path,
    device: torch.device,
    **kwargs: Any,
) -> nn.Module:
    """Load StyleGAN2 generator.

    Args:
        checkpoint_path: Path to checkpoint
        device: Target device
        **kwargs: Additional arguments

    Returns:
        StyleGAN2 generator
    """
    # Load checkpoint
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
    **kwargs: Any,
) -> nn.Module:
    """Load DCGAN generator.

    Args:
        checkpoint_path: Path to checkpoint
        device: Target device
        latent_dim: Latent dimension
        img_channels: Number of image channels
        feature_maps: Base number of feature maps
        **kwargs: Additional arguments

    Returns:
        DCGAN generator
    """
    from leakpro.fl_utils.gan_models import DCGANGenerator

    generator = DCGANGenerator(
        latent_dim=latent_dim,
        img_channels=img_channels,
        feature_maps=feature_maps,
    )

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "generator" in checkpoint:
        generator.load_state_dict(checkpoint["generator"])
    elif "state_dict" in checkpoint:
        generator.load_state_dict(checkpoint["state_dict"])
    else:
        generator.load_state_dict(checkpoint)

    return generator


def _load_custom(
    checkpoint_path: Path,
    device: torch.device,
    model_class: type[nn.Module] | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Load custom GAN generator.

    Args:
        checkpoint_path: Path to checkpoint
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

    # Instantiate model
    generator = model_class(**kwargs)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict):
        if "generator" in checkpoint:
            generator.load_state_dict(checkpoint["generator"])
        elif "state_dict" in checkpoint:
            generator.load_state_dict(checkpoint["state_dict"])
        else:
            generator.load_state_dict(checkpoint)
    else:
        # Checkpoint is the model itself
        generator = checkpoint

    return generator


def sample_latent(
    batch_size: int,
    latent_dim: int,
    device: torch.device | str = "cpu",
    distribution: str = "normal",
) -> torch.Tensor:
    """Sample latent codes for GAN generation.

    Args:
        batch_size: Number of samples
        latent_dim: Latent dimension
        device: Device to create tensor on
        distribution: Distribution to sample from ("normal", "uniform")

    Returns:
        Latent codes [batch_size, latent_dim]
    """
    device = torch.device(device) if isinstance(device, str) else device

    if distribution == "normal":
        z = torch.randn(batch_size, latent_dim, device=device)
    elif distribution == "uniform":
        z = torch.rand(batch_size, latent_dim, device=device) * 2 - 1  # [-1, 1]
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return z


def _load_huggingface_gan(
    model_id: str,
    device: torch.device,
    **kwargs: Any,
) -> nn.Module:
    """Load GAN generator from GitHub (gan-vae-pretrained-pytorch).

    Supports pre-trained DCGAN models from csinva/gan-vae-pretrained-pytorch.
    Model format: github_username/dataset_dcgan (e.g., "csinva/cifar10_dcgan")

    Args:
        model_id: GitHub model ID (e.g., "csinva/cifar10_dcgan")
        device: Device to load model on
        **kwargs: Additional arguments (latent_dim, img_size, etc.)

    Returns:
        generator: Loaded generator in eval mode

    Raises:
        ImportError: If torch_utils not available
        RuntimeError: If model loading fails
    """
    import urllib.request
    import tempfile
    import os
    
    logger.info(f"Loading GAN from GitHub: {model_id}")

    # Parse model_id: expected format "csinva/cifar10_dcgan" or similar
    parts = model_id.split('/')
    if len(parts) != 2:
        raise ValueError(f"Model ID should be in format 'username/model_name', got: {model_id}")
    
    username, model_name = parts
    
    # Map common model names to their epoch
    model_epoch_map = {
        "cifar10_dcgan": 199,
        "mnist_dcgan": 99,
        "cifar100_dcgan": 299,
    }
    
    epoch = kwargs.get("epoch", model_epoch_map.get(model_name, 199))
    
    # Construct GitHub raw URL for weights
    base_url = f"https://raw.githubusercontent.com/{username}/gan-vae-pretrained-pytorch/master/{model_name}/weights"
    netG_filename = f"netG_epoch_{epoch}.pth"
    netG_url = f"{base_url}/{netG_filename}"
    
    logger.info(f"  Downloading from: {netG_url}")
    
    # Download the weights to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
        try:
            urllib.request.urlretrieve(netG_url, tmp.name)
            checkpoint_path = tmp.name
        except Exception as e:
            raise RuntimeError(f"Failed to download model from {netG_url}: {e}")
    
    try:
        # Use the standard DCGAN architecture that matches csinva's model
        # csinva uses: nz=100, ngf=64, nc=3 (or 1 for grayscale)
        if "cifar10" in model_name:
            generator = _build_dcgan_generator(
                nz=100,
                nc=3,
                ngf=64,
            )
        elif "mnist" in model_name:
            generator = _build_dcgan_generator(
                nz=100,
                nc=1,
                ngf=64,
            )
        elif "cifar100" in model_name:
            generator = _build_dcgan_generator(
                nz=100,
                nc=1,
                ngf=64,
            )
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # The checkpoint may have "main" prefix from the Generator class
        # Adjust keys if necessary
        if isinstance(checkpoint, dict):
            if any(k.startswith('main.') for k in checkpoint.keys()):
                # Remove 'main.' prefix
                checkpoint = {k.replace('main.', ''): v for k, v in checkpoint.items()}
        
        generator.load_state_dict(checkpoint)
        
        generator = generator.to(device)
        generator.eval()
        
        logger.info(f"✓ Successfully loaded generator from {model_id}")
        return generator
        
    finally:
        # Clean up temporary file
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)


def _build_dcgan_generator(nz: int = 100, nc: int = 3, ngf: int = 64) -> nn.Module:
    """Build DCGAN generator matching csinva architecture.
    
    Args:
        nz: Latent dimension
        nc: Number of output channels
        ngf: Base number of features
    
    Returns:
        Generator module
    """
    return nn.Sequential(
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


def _load_biggan_wrapper(
    device: torch.device,
    model_name: str = "biggan-deep-256",
    img_size: int = 64,
    truncation: float = 0.4,
    **kwargs: Any,
) -> nn.Module:
    """Load a BigGAN model and wrap it in a :class:`BigGANLayeredWrapper`.

    Downloads the pre-trained ``pytorch_pretrained_biggan`` model on first call
    (cached in ``~/.cache/torch/pretrained_models``).

    Args:
        device:     Target device.
        model_name: BigGAN variant (e.g. ``"biggan-deep-256"``).
        img_size:   Spatial size to resize generated images to (default 64).
        truncation: BigGAN truncation factor (default 0.4).
        **kwargs:   Unused; accepted for interface compatibility.

    Returns:
        :class:`~leakpro.fl_utils.gan_models.BigGANLayeredWrapper` in eval mode.
    """
    try:
        from pytorch_pretrained_biggan import BigGAN  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "pytorch_pretrained_biggan is required for the BigGAN architecture.  "
            "Install it with: pip install pytorch_pretrained_biggan"
        ) from exc

    from leakpro.fl_utils.gan_models import BigGANLayeredWrapper

    logger.info(f"Loading BigGAN '{model_name}' (img_size={img_size}, truncation={truncation})")
    biggan = BigGAN.from_pretrained(model_name)
    biggan = biggan.to(device).eval()
    biggan.requires_grad_(False)
    wrapper = BigGANLayeredWrapper(biggan, truncation=truncation, img_size=img_size)
    wrapper.to(device)
    return wrapper