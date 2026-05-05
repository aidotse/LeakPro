#
# Copyright 2023-2026 AI Sweden
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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



__all__ = [
    "RandomNoiseInitialization",
]
