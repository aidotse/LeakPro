#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Aggregation strategies for combining multiple reconstruction candidates.

This module contains two types of aggregation strategies:

1. Seed Aggregation: Combines multiple random initializations (seeds) for the same image
   - Operates on G dimension: [E, N, G, C, H, W] → [E, N, 1, C, H, W]
   - All seeds at (epoch, position) represent the SAME true image
   - Safe to register/align spatially before aggregating

2. Epoch Aggregation: Matches and combines reconstructions across training epochs
   - Operates on E dimension: [E, N, 1, C, H, W] → [N, C, H, W]
   - Images may be in DIFFERENT order across epochs (requires matching)
   - Assumes seeds already aggregated (G=1)

Pipeline order:
    1. Seed aggregation (within each epoch-position)
    2. Epoch aggregation (match across epochs, then average)

Seed Aggregation Strategies:
    - NoSeedAggregation: Single seed only (G=1)
    - MeanSeedAggregation: Pixel-wise averaging across seeds
    - RegisteredSeedAggregation: Image registration + averaging (future work)

Epoch Aggregation Strategies:
    - EpochMatchingConsensus: Hungarian matching + averaging for FedAvg attacks
"""

from __future__ import annotations

import torch

from leakpro.attacks.gia_attacks.modular.config.registry import register
from leakpro.attacks.gia_attacks.modular.core.component_base import (
    AggregationStrategy,
    ComponentMetadata,
)
from leakpro.fl_utils.matching_utils import compute_cost_matrix, hungarian_matching


@register("aggregation.none")
class NoSeedAggregation(AggregationStrategy):
    """No seed aggregation - assumes single seed per image (G=1).

    This is the default strategy for attacks that don't use multi-seed optimization.
    """

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Get metadata for no seed aggregation."""
        return ComponentMetadata(
            name="no_seed_aggregation",
            required_capabilities={},
        )

    def compute_consensus(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """Return first seed only (assumes G=1).

        Args:
            reconstruction: Standardized format [E, N, G, C, H, W] where G should be 1

        Returns:
            First seed with shape [E, N, 1, C, H, W]

        Raises:
            ValueError: If num_seeds > 1

        """
        return reconstruction


@register("aggregation.mean")
class MeanSeedAggregation(AggregationStrategy):
    """Mean aggregation across seeds.

    Computes pixel-wise average across all G seeds for each image.
    This is the primary aggregation method from the See Through Gradients paper.

    Reference:
        Yin et al., "See through Gradients: Image Batch Recovery via
        GradInversion", CVPR 2021, Equation 12
    """

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Get metadata for mean seed aggregation."""
        return ComponentMetadata(
            name="mean_seed_aggregation",
            required_capabilities={},
        )

    def compute_consensus(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """Compute mean across seeds.

        Args:
            reconstruction: Standardized format [E, N, G, C, H, W]

        Returns:
            Mean across seeds with shape [E, N, 1, C, H, W] (keepdim=True)

        """
        if reconstruction.ndim != 6:
            raise ValueError(
                f"Expected 6D reconstruction [E, N, G, C, H, W], got {reconstruction.ndim}D"
            )

        # Average across seed dimension (dim=2)
        return reconstruction.mean(dim=2, keepdim=True)


@register("aggregation.best")
class BestSeedAggregation(AggregationStrategy):
    """Best seed selection based on gradient matching loss.

    Selects the seed with minimum gradient matching loss for each image.
    This is the approach used in GIFD and GIAS papers with restarts.
    
    Note: This strategy REQUIRES storing per-seed losses during optimization.
    The orchestrator must track losses for each seed and pass them via aux_data.

    Reference:
        Jeon et al., "Gradient Inversion with Generative Image Prior", NeurIPS 2021
        Fang et al., "GIFD: A Generative Gradient Inversion Method", ICCV 2023
    """

    def __init__(self) -> None:
        """Initialize best seed aggregation."""
        self._seed_losses = None  # Will be set by orchestrator

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Get metadata for best seed aggregation."""
        return ComponentMetadata(
            name="best_seed_aggregation",
            required_capabilities={},
        )

    def set_seed_losses(self, losses: torch.Tensor) -> None:
        """Set per-seed losses for selection.
        
        Args:
            losses: Tensor of shape [E, N, G] with gradient matching loss for each seed
        """
        self._seed_losses = losses

    def compute_consensus(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """Select best seed based on minimum loss.

        Args:
            reconstruction: Standardized format [E, N, G, C, H, W]

        Returns:
            Best seeds with shape [E, N, 1, C, H, W]

        """
        if reconstruction.ndim != 6:
            raise ValueError(
                f"Expected 6D reconstruction [E, N, G, C, H, W], got {reconstruction.ndim}D"
            )

        E, N, G, C, H, W = reconstruction.shape

        if self._seed_losses is None:
            # Fallback: if no losses provided, just take first seed
            # This shouldn't happen in normal operation
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("BestSeedAggregation: No seed losses provided, returning first seed only")
            return reconstruction[:, :, 0:1]

        # Validate seed losses shape
        if self._seed_losses.shape != (E, N, G):
            import logging
            logger = logging.getLogger(__name__)
            logger.error(
                f"BestSeedAggregation: seed_losses shape mismatch. "
                f"Expected [{E}, {N}, {G}], got {list(self._seed_losses.shape)}"
            )
            return reconstruction[:, :, 0:1]

        # Check for invalid values in losses
        if torch.isnan(self._seed_losses).any() or torch.isinf(self._seed_losses).any():
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("BestSeedAggregation: NaN or Inf in seed losses, returning first seed only")
            return reconstruction[:, :, 0:1]

        # Select best seed (minimum loss) for each (epoch, image) pair
        # _seed_losses shape: [E, N, G]
        best_indices = torch.argmin(self._seed_losses, dim=2)  # [E, N]

        # Gather best seeds
        # Expand indices to match reconstruction shape
        # best_indices: [E, N] → [E, N, 1, 1, 1, 1] → [E, N, 1, C, H, W]
        best_indices_expanded = best_indices[:, :, None, None, None, None].expand(E, N, 1, C, H, W)

        # Gather: select the seed at best_indices for each (e, n)
        best_recon = torch.gather(reconstruction, 2, best_indices_expanded)  # [E, N, 1, C, H, W]

        return best_recon


@register("aggregation.registered")
class RegisteredSeedAggregation(AggregationStrategy):
    """Registered mean aggregation with image alignment.

    This strategy first computes a coarse mean, then registers (aligns) each
    seed to the mean using optical flow or image registration, and finally
    computes the mean of the registered images.

    From See Through Gradients paper:
        E(x̂) = (1/|G|) Σ_g F_{x̂_g → target}(x̂_g)

    Where F is an image registration function (e.g., RANSAC-flow).

    Not yet implemented — instantiation raises ``NotImplementedError``.
    The See Through Gradients preset uses :class:`MeanSeedAggregation` for now;
    swap in this strategy once the registration step is wired up.

    Reference:
        Yin et al., "See through Gradients: Image Batch Recovery via
        GradInversion", CVPR 2021, Equation 12
    """

    def __init__(self, registration_method: str = "ransac_flow") -> None:
        """Initialize registered seed aggregation.

        Args:
            registration_method: Method for image registration
                Options: "ransac_flow" (requires external library)

        """
        self.registration_method = registration_method
        self._registration_available = False

        raise NotImplementedError(
            "RegisteredSeedAggregation is not yet implemented. "
        )

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Get metadata for registered seed aggregation."""
        return ComponentMetadata(
            name="registered_seed_aggregation",
            required_capabilities={},
        )


@register("aggregation.epoch_matching")
class EpochMatchingConsensus(AggregationStrategy):
    """Match and average reconstructions across epochs for FedAvg attacks.

    This strategy assumes seeds have already been aggregated (G=1).
    It only handles matching reconstructions across epochs (E dimension).

    The matching problem: In FedAvg with E local epochs, client trains on
    shuffled data each epoch. We reconstruct E separate variables, but
    don't know which position in epoch 0 corresponds to which in epoch 1.
    Uses Hungarian algorithm to find optimal matching based on similarity.

    Reference:
        Dimitrov et al., "Data Leakage in Federated Averaging", 2022
    """

    def __init__(
        self,
        epochs: int = 3,
        metric: str = "l2",
    ) -> None:
        """Initialize epoch matching consensus.

        Args:
            epochs: Number of local epochs (E)
            metric: Similarity metric for matching
                - "l2": L2 distance
                - "mse": Mean squared error
                - "psnr": Peak signal-to-noise ratio (lower is better)

        """
        self.epochs = epochs
        self.metric = metric

        if metric not in ["l2", "mse", "psnr"]:
            raise ValueError(f"Unknown metric: {metric}")

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Get metadata for epoch matching consensus."""
        return ComponentMetadata(
            name="epoch_matching_consensus",
            required_capabilities={},
        )

    def compute_consensus(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """Match reconstructions across epochs and average.

        Assumes seeds have already been aggregated (G=1).

        Args:
            reconstruction: Standardized format [E, N, 1, C, H, W] with G=1

        Returns:
            Consensus reconstruction [1, N, 1, C, H, W] (maintains 6D contract)

        Raises:
            ValueError: If seeds not aggregated (G != 1) or wrong number of dimensions

        """
        if reconstruction.ndim != 6:
            raise ValueError(
                f"Expected 6D reconstruction [E, N, G, C, H, W], got {reconstruction.ndim}D. "
                f"Shape: {reconstruction.shape}"
            )

        num_epochs, num_images, num_seeds = reconstruction.shape[:3]

        if num_seeds != 1:
            raise ValueError(
                f"EpochMatchingConsensus requires seeds to be aggregated first (G=1). "
                f"Got G={num_seeds}. Apply seed aggregation before epoch aggregation."
            )

        if self.epochs != num_epochs:
            raise ValueError(
                f"Expected E={self.epochs} epochs, got {num_epochs}"
            )

        # Remove seed dimension for matching: [E, N, 1, C, H, W] → [E, N, C, H, W]
        reconstruction = reconstruction[:, :, 0]

        # Match and average across epochs → [N, C, H, W]
        if num_epochs == 1:
            matched = reconstruction[0]
        else:
            matched = self._match_and_average(reconstruction)

        # Restore E=1 and G=1 to maintain 6D output contract: [N, C, H, W] → [1, N, 1, C, H, W]
        return matched.unsqueeze(0).unsqueeze(2)

    def _match_and_average(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """Match and average reconstructions for single seed.

        Args:
            reconstruction: Shape [E, N, C, H, W]

        Returns:
            Matched and averaged with shape [N, C, H, W]

        """
        num_epochs, num_images = reconstruction.shape[:2]

        # If only one epoch, no matching needed
        if num_epochs == 1:
            return reconstruction[0]

        # Use first epoch as reference and match others to it
        epoch_0 = reconstruction[0]  # [N, C, H, W]

        # Store matched images for each position in epoch 0
        matched_images_list = [epoch_0[i].unsqueeze(0) for i in range(num_images)]

        # Match each subsequent epoch to epoch 0
        for e in range(1, num_epochs):
            epoch_e = reconstruction[e]  # [N, C, H, W]

            # Compute cost matrix: [N, N]
            cost_matrix = compute_cost_matrix(epoch_0, epoch_e, metric=self.metric)

            # Use Hungarian algorithm for optimal matching
            matches = hungarian_matching(cost_matrix)

            # Store matched images
            for i in range(num_images):
                matched_idx = matches[i]
                matched_images_list[i] = torch.cat([
                    matched_images_list[i],
                    epoch_e[matched_idx].unsqueeze(0)
                ], dim=0)

        # Average matched images across epochs
        averaged = [imgs.mean(dim=0) for imgs in matched_images_list]  # Each is [C, H, W]

        # Stack to [N, C, H, W]
        return torch.stack(averaged, dim=0)




__all__ = [
    "NoSeedAggregation",
    "MeanSeedAggregation",
    "BestSeedAggregation",
    "RegisteredSeedAggregation",
    "EpochMatchingConsensus",
]
