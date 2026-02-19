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

from leakpro.attacks.gia_attacks.modular.core.component_base import (
    AggregationStrategy,
    ComponentMetadata,
)
from leakpro.fl_utils.matching_utils import compute_cost_matrix, hungarian_matching


class NoSeedAggregation(AggregationStrategy):
    """No seed aggregation - assumes single seed per image (G=1).

    This is the default strategy for attacks that don't use multi-seed optimization.
    """

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Get metadata for no seed aggregation."""
        return ComponentMetadata(
            name="no_seed_aggregation",
            display_name="No Seed Aggregation (Single Seed)",
            description="Assumes single seed per image, takes first seed only",
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
            display_name="Mean Seed Aggregation",
            description="Pixel-wise averaging across all seeds per image",
            required_capabilities={},
            paper_reference="Yin et al., See Through Gradients, CVPR 2021",
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


class RegisteredSeedAggregation(AggregationStrategy):
    """Registered mean aggregation with image alignment.

    This strategy first computes a coarse mean, then registers (aligns) each
    seed to the mean using optical flow or image registration, and finally
    computes the mean of the registered images.

    From See Through Gradients paper:
        E(x̂) = (1/|G|) Σ_g F_{x̂_g → target}(x̂_g)

    Where F is an image registration function (e.g., RANSAC-flow).

    Current implementation: Falls back to simple mean (registration TODO).

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
            display_name="Registered Seed Aggregation (with Image Alignment)",
            description="Image registration + averaging for better consensus",
            required_capabilities={},
            paper_reference="Yin et al., See Through Gradients, CVPR 2021",
        )


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
            display_name="Epoch Matching Consensus (FedAvg)",
            description="Match and average reconstructions across epochs using Hungarian algorithm",
            required_capabilities={},
            paper_reference="Dimitrov et al., Data Leakage in Federated Averaging, 2022",
        )

    def compute_consensus(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """Match reconstructions across epochs and average.

        Assumes seeds have already been aggregated (G=1).

        Args:
            reconstruction: Standardized format [E, N, 1, C, H, W] with G=1

        Returns:
            Matched and averaged reconstruction with shape [N, C, H, W]

        Raises:
            ValueError: If seeds not aggregated (G != 1) or wrong number of dimensions

        """
        # Expect standardized 6D format [E, N, G, C, H, W]
        if reconstruction.ndim != 6:
            raise ValueError(
                f"Expected 6D reconstruction [E, N, G, C, H, W], got {reconstruction.ndim}D. "
                f"Shape: {reconstruction.shape}"
            )

        num_epochs, num_images, num_seeds = reconstruction.shape[:3]

        # Validate that seeds have been aggregated
        if num_seeds != 1:
            raise ValueError(
                f"EpochMatchingConsensus requires seeds to be aggregated first (G=1). "
                f"Got G={num_seeds}. Apply seed aggregation before epoch aggregation."
            )

        # Validate dimensions
        if self.epochs != num_epochs:
            raise ValueError(
                f"Expected E={self.epochs} epochs, got {num_epochs}"
            )
        # Note: N is the total number of images being reconstructed, which may differ
        # from the mini-batch size used during training. The matching works with any N.

        # Squeeze out seed dimension: [E, N, 1, C, H, W] -> [E, N, C, H, W]
        reconstruction = reconstruction.squeeze(2)

        # If single epoch, no matching needed
        if num_epochs == 1:
            return reconstruction[0]  # [N, C, H, W]

        # Multi-epoch: match and average
        return self._match_and_average(reconstruction)

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
    "RegisteredSeedAggregation",
    "EpochMatchingConsensus",
]
