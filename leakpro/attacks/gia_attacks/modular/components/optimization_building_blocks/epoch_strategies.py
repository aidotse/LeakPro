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
"""Epoch handling strategies for gradient inversion attacks.

This module defines different strategies for handling epochs during reconstruction.
Different attacks may want to handle multi-epoch training differently:
- Separate images per epoch (Dimitrov FedAvg style)
- Same batch repeated across epochs (Geiping style)
- Single collapsed batch

The strategy provides data on demand for each epoch during training simulation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class EpochHandlingStrategy(ABC):
    """Base class for epoch handling strategies.

    An epoch handling strategy controls how reconstruction data is provided
    to the training simulator for each epoch. It abstracts away the differences
    in how data is organized (separate per epoch vs repeated vs collapsed).
    """

    @abstractmethod
    def get_data_for_epoch(
        self,
        reconstruction: torch.Tensor,
        epoch_idx: int,
    ) -> torch.Tensor:
        """Get the data to use for a specific epoch during training.

        Args:
            reconstruction: Full reconstruction tensor [E, N, C, H, W] for single seed
            epoch_idx: Which epoch to get data for (0-indexed)

        Returns:
            Data for this epoch: [N, C, H, W]

        """
        pass

    @abstractmethod
    def get_expected_reconstruction_shape(
        self,
        num_images: int,
        num_epochs: int,
        num_seeds: int,
        input_shape: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Return the expected shape for reconstruction initialization.

        Always returns [E, N, G, C, H, W] format for consistency.
        For strategies that don't use separate data per epoch, E can be 1.

        Args:
            num_images: Total number of images to reconstruct
            num_epochs: Number of training epochs
            num_seeds: Number of random seeds for optimization
            input_shape: Shape of a single image (C, H, W)

        Returns:
            Expected reconstruction tensor shape [E, N, G, C, H, W]

        """
        pass

    @abstractmethod
    def get_label_for_epoch(
        self,
        labels: torch.Tensor,
        epoch_idx: int,
    ) -> torch.Tensor:
        """Get the labels to use for a specific epoch during training.

        Args:
            labels: Full label tensor [E, N] or [E, N, K]
            epoch_idx: Which epoch to get labels for (0-indexed)

        Returns:
            Labels for this epoch: [N] or [N, K]

        """
        pass

    @abstractmethod
    def get_expected_label_shape(
        self,
        num_images: int,
        num_epochs: int,
    ) -> tuple[int, ...]:
        """Return the expected shape for label tensor.

        Args:
            num_images: Total number of images
            num_epochs: Number of training epochs

        Returns:
            Expected label tensor shape, e.g., [E, N] or [N]

        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return name of this strategy."""
        pass


class MultiEpochSeparate(EpochHandlingStrategy):
    """Separate images for each epoch (Dimitrov FedAvg style).

    Stores different optimization variables for each epoch, allowing them to evolve
    independently during reconstruction. This handles the epoch order-invariance
    problem in FedAvg where we don't know which batch order was used in each epoch.

    Reconstruction shape: [E, N, G, C, H, W]
    - E: Number of epochs (each epoch has separate optimization variables)
    - N: Number of images (all client images, used in each epoch)
    - G: Number of random seeds
    - C, H, W: Image dimensions

    Example: 12 client images trained for 3 epochs → [3, 12, 1, 3, 32, 32]
             This creates 3 separate sets of 12 images to optimize.
    """

    def get_data_for_epoch(
        self,
        reconstruction: torch.Tensor,
        epoch_idx: int,
    ) -> torch.Tensor:
        """Return data for specific epoch - just index into epoch dimension.

        Args:
            reconstruction: [E, N, C, H, W] tensor for single seed
            epoch_idx: Which epoch (0-indexed)

        Returns:
            [N, C, H, W] - the images for this epoch

        """
        return reconstruction[epoch_idx]

    def get_expected_reconstruction_shape(
        self,
        num_images: int,
        num_epochs: int,
        num_seeds: int,
        input_shape: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Return [E, N, G, C, H, W]."""
        return (num_epochs, num_images, num_seeds, *input_shape)

    def get_label_for_epoch(
        self,
        labels: torch.Tensor,
        epoch_idx: int,
    ) -> torch.Tensor:
        """Return labels for specific epoch - just index into epoch dimension.

        Args:
            labels: [E, N] or [E, N, K] tensor
            epoch_idx: Which epoch (0-indexed)

        Returns:
            [N] or [N, K] - the labels for this epoch

        """
        return labels[epoch_idx]

    def get_expected_label_shape(
        self,
        num_images: int,
        num_epochs: int,
    ) -> tuple[int, ...]:
        """Return [E, N]."""
        return (num_epochs, num_images)

    def get_name(self) -> str:
        """Return name of this strategy."""
        return "MultiEpochSeparate"


class SingleStorageReused(EpochHandlingStrategy):
    """Single storage reused across epochs (Geiping style).

    Uses E=1 storage and returns the same data for all epochs.
    Whether this represents "repeated images" or "single batch" depends on
    the training simulator's epochs parameter, not the strategy itself.
    """

    def get_data_for_epoch(
        self,
        reconstruction: torch.Tensor,
        epoch_idx: int,
    ) -> torch.Tensor:
        """Return the same data for every epoch.

        Args:
            reconstruction: [E, N, C, H, W] tensor for single seed (E=1)
            epoch_idx: Which epoch (ignored)

        Returns:
            [N, C, H, W] - the images from the single stored epoch

        """
        _ = epoch_idx  # Unused since we return the same data for all epochs
        return reconstruction[0]

    def get_expected_reconstruction_shape(
        self,
        num_images: int,
        num_epochs: int,
        num_seeds: int,
        input_shape: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Return [E, N, G, C, H, W] where E=1 and N=num_images."""
        _ = num_epochs  # Unused since we only store one epoch's data
        return (1, num_images, num_seeds, *input_shape)

    def get_label_for_epoch(
        self,
        labels: torch.Tensor,
        epoch_idx: int,
    ) -> torch.Tensor:
        """Return the same labels for every epoch.

        Args:
            labels: [E, N] or [E, N, K] tensor (E=1)
            epoch_idx: Which epoch (ignored)

        Returns:
            [N] or [N, K] - the labels from the single stored epoch

        """
        _ = epoch_idx  # Unused since we return the same labels for all epochs
        return labels[0]

    def get_expected_label_shape(
        self,
        num_images: int,
        num_epochs: int,
    ) -> tuple[int, ...]:
        """Return [E, N] where E=1 and N=num_images."""
        _ = num_epochs  # Unused since we only store one epoch's labels
        return (1, num_images)

    def get_name(self) -> str:
        """Return name of this strategy."""
        return "SingleStorageReused"


__all__ = [
    "EpochHandlingStrategy",
    "MultiEpochSeparate",
    "SingleStorageReused",
]
