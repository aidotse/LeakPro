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

from leakpro.attacks.gia_attacks.modular.core.component_base import (
    ComponentMetadata,
    InitializationResult,
    InitializationStrategy,
)


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
            display_name="Random Noise Initialization",
            description="Initialize with Gaussian noise N(mean, std)",
            required_capabilities={},
        )

    def initialize(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        num_seeds: int = 1,
    ) -> InitializationResult:
        """Initialize with random Gaussian noise.

        Args:
            shape: Base shape of reconstruction (e.g., [B, C, H, W])
            device: Device to create tensor on
            dtype: Data type for tensor
            num_seeds: Number of random seeds per image (for multi-seed optimization)

        Returns:
            InitializationResult with reconstruction tensor.
            If num_seeds > 1, shape is [B, G, C, H, W]
            If num_seeds == 1, shape is [B, C, H, W] (backward compatible)

        """
        if num_seeds > 1:
            # Multi-seed: insert seed dimension
            # shape is [B, C, H, W] → create [B, G, C, H, W]
            batch_size = shape[0]
            extended_shape = (batch_size, num_seeds) + shape[1:]

            # Generate different random noise for each seed
            reconstruction = torch.randn(extended_shape, device=device, dtype=dtype) * self.std + self.mean

            metadata = {
                "mean": self.mean,
                "std": self.std,
                "method": "random_noise",
                "num_seeds": num_seeds,
            }
        else:
            # Single seed: keep original behavior
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



__all__ = [
    "RandomNoiseInitialization",
]
