#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Initialization strategies for gradient inversion attacks.

Initialization strategies create the starting point for reconstruction optimization.
Good initialization can significantly improve convergence speed and final quality.

Strategies:
    - RandomNoiseInitialization: Random Gaussian noise
    - UniformNoiseInitialization: Uniform random values
    - DataPriorInitialization: Initialize using dataset statistics (mean/std)
"""

from __future__ import annotations

import torch

from leakpro.attacks.gia_attacks.modular.config.registry import register
from leakpro.attacks.gia_attacks.modular.core.component_base import (
    ComponentMetadata,
    InitializationResult,
    InitializationStrategy,
)


@register("init.random_noise")
class RandomNoiseInitialization(InitializationStrategy):
    """Initialize with random Gaussian noise.

    Simple baseline initialization - samples from N(mean, std).
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        """Initialize random noise strategy.

        Args:
            mean: Mean of Gaussian distribution
            std: Standard deviation of Gaussian distribution

        """
        self.mean = mean
        self.std = std

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Get metadata for this initialization strategy."""
        return ComponentMetadata(
            name="random_noise",
            required_capabilities={},
        )

    def initialize(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> InitializationResult:
        """Initialize with random Gaussian noise.

        Args:
            shape: Shape of reconstruction
            device: Device to create tensor on
            dtype: Data type for tensor

        Returns:
            InitializationResult with reconstruction tensor in [E, N, G, C, H, W] format.

        """

        # Generate different random noise for each epoch and seed
        reconstruction = torch.randn(shape, device=device, dtype=dtype) * self.std + self.mean

        metadata = {
            "mean": self.mean,
            "std": self.std,
            "method": "random_noise",
        }

        return InitializationResult(
            reconstruction=reconstruction,
            labels=None,
            metadata=metadata,
        )


@register("init.fixed")
class FixedInitialization(InitializationStrategy):
    """Initialize with a pre-generated fixed tensor.
    
    Useful for exact reproducibility when comparing implementations,
    as it guarantees identical starting points regardless of RNG state.
    """

    def __init__(self, fixed_tensor: torch.Tensor) -> None:
        """Initialize with fixed tensor strategy.

        Args:
            fixed_tensor: Pre-generated tensor to use as initialization.
                         Will be cloned and moved to the correct device/dtype.

        """
        self.fixed_tensor = fixed_tensor.detach().clone()

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Get metadata for this initialization strategy."""
        return ComponentMetadata(
            name="fixed",
            required_capabilities={},
        )

    def initialize(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> InitializationResult:
        """Initialize with fixed tensor.

        Args:
            shape: Expected shape of reconstruction (must match fixed_tensor shape)
            device: Device to create tensor on
            dtype: Data type for tensor

        Returns:
            InitializationResult with reconstruction tensor in [E, N, G, C, H, W] format.

        """
        # Clone and move to correct device/dtype
        reconstruction = self.fixed_tensor.clone().to(device=device, dtype=dtype)

        # Verify shape matches
        if reconstruction.shape != shape:
            raise ValueError(
                f"Fixed tensor shape {reconstruction.shape} does not match "
                f"expected shape {shape}. Ensure the pre-generated tensor has "
                f"the correct dimensions [E, N, G, C, H, W]."
            )

        metadata = {
            "method": "fixed",
            "original_device": str(self.fixed_tensor.device),
            "original_dtype": str(self.fixed_tensor.dtype),
        }

        return InitializationResult(
            reconstruction=reconstruction,
            labels=None,
            metadata=metadata,
        )


__all__ = [
    "RandomNoiseInitialization",
    "FixedInitialization",
]
