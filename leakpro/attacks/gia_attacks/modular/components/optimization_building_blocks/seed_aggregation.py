"""Seed aggregation strategies for multi-seed gradient inversion attacks.

Seed aggregation handles multiple random initializations (seeds) per image,
enabling joint optimization of G parallel reconstructions per image. This is
used in attacks like "See Through Gradients" (Yin et al., CVPR 2021).

When using multi-seed optimization, the reconstruction tensor has shape:
    [batch_size, num_seeds, channels, height, width]

Strategies:
    - NoSeedAggregation: Single seed only (G=1), default for backward compatibility
    - MeanSeedAggregation: Simple pixel-wise averaging across seeds
    - RegisteredSeedAggregation: Image registration + averaging (future work)
"""

from __future__ import annotations

import torch

from leakpro.attacks.gia_attacks.modular.core.component_base import (
    ComponentMetadata,
    SeedAggregationStrategy,
)


class NoSeedAggregation(SeedAggregationStrategy):
    """No seed aggregation - assumes single seed per image (G=1).

    This is the default strategy for backward compatibility with attacks
    that don't use multi-seed optimization.
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
            reconstruction: Shape [B, G, C, H, W] where G should be 1

        Returns:
            First seed with shape [B, C, H, W]

        Raises:
            ValueError: If num_seeds > 1

        """
        if reconstruction.ndim != 5:
            raise ValueError(
                f"Expected 5D reconstruction [B, G, C, H, W], got {reconstruction.ndim}D"
            )

        if reconstruction.shape[1] > 1:
            raise ValueError(
                f"NoSeedAggregation requires num_seeds=1, got {reconstruction.shape[1]} seeds. "
                "Use MeanSeedAggregation or RegisteredSeedAggregation for multi-seed optimization."
            )

        return reconstruction[:, 0, ...]


class MeanSeedAggregation(SeedAggregationStrategy):
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
            reconstruction: Shape [B, G, C, H, W]

        Returns:
            Mean across seeds with shape [B, C, H, W]

        """
        if reconstruction.ndim != 5:
            raise ValueError(
                f"Expected 5D reconstruction [B, G, C, H, W], got {reconstruction.ndim}D"
            )

        # Average across seed dimension (dim=1)
        return reconstruction.mean(dim=1)


class RegisteredSeedAggregation(SeedAggregationStrategy):
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


__all__ = [
    "NoSeedAggregation",
    "MeanSeedAggregation",
    "RegisteredSeedAggregation",
]
