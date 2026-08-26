#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Validated attack configuration models."""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SimilarityBand(BaseModel):
    """Inclusive similarity or distance band used for AMS and UMS."""

    lower: float
    upper: float
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_bounds(self) -> "SimilarityBand":
        """Reject inverted distance bands."""
        if self.lower > self.upper:
            raise ValueError("SimilarityBand.lower must not exceed upper.")
        return self


class CommonExtractionConfig(BaseModel):
    """Shared runtime and authorization settings."""

    random_seed: int = 42
    authorized_audit: bool = Field(
        default=False,
        description="Must be true to confirm authorization to audit the model and referenced data.",
    )
    image_range: Literal["zero_one", "minus_one_one"] = "zero_one"
    generation_batch_size: int = Field(default=64, ge=1)
    distance_block_size: int = Field(default=64, ge=1)
    distance_device: str = "cpu"
    compute_device: str = "auto"
    overwrite_results: bool = Field(
        default=False,
        description="Allow an existing result with the same audit identity to be replaced.",
    )
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class CarliniConfig(CommonExtractionConfig):
    """Configuration for the Carlini et al. generate-and-filter attack."""

    mode: Literal["conditional_black_box", "unconditional_reference_audit"] = "conditional_black_box"
    num_generations_per_condition: int = Field(default=500, ge=2)
    num_unconditional_generations: int = Field(default=1_000_000, ge=1)
    tile_grid: Tuple[int, int] = (4, 4)
    tiled_l2_threshold: float = Field(
        default=0.1,
        gt=0,
        description="Engineering default: the paper does not publish the graph-edge threshold; calibrate per model.",
    )
    min_clique_size: int = Field(default=10, ge=2)
    verification_l2_threshold: float = Field(default=0.15, gt=0)
    reference_alpha: float = Field(default=0.5, gt=0)
    reference_neighbors: int = Field(default=50, ge=2)
    ratio_threshold: float = Field(default=1.0, gt=0)
    max_extractions: Optional[int] = Field(default=None, ge=1)

    @field_validator("tile_grid")
    @classmethod
    def validate_tile_grid(cls, value: Tuple[int, int]) -> Tuple[int, int]:
        """Require a positive two-dimensional tile grid."""
        if value[0] < 1 or value[1] < 1:
            raise ValueError("tile_grid values must be positive.")
        return value

    @model_validator(mode="after")
    def validate_clique_budget(self) -> "CarliniConfig":
        """Require enough generations to form the configured clique."""
        if self.mode == "conditional_black_box" and self.min_clique_size > self.num_generations_per_condition:
            raise ValueError("min_clique_size cannot exceed num_generations_per_condition.")
        return self


class SIDEConfig(CommonExtractionConfig):
    """Configuration for SIDE's white-box time-dependent classifier branch."""

    synthetic_samples: int = Field(default=10_000, ge=2)
    synthetic_batch_size: int = Field(default=64, ge=1)
    clusters: int = Field(default=100, ge=2)
    cohesion_threshold: float = Field(default=0.5, ge=-1, le=1)
    min_cluster_size: int = Field(default=2, ge=1)
    kmeans_n_init: int = Field(default=10, ge=1)
    classifier_epochs: int = Field(
        default=20,
        ge=1,
        description="Engineering default: the paper does not report classifier epoch count.",
    )
    classifier_batch_size: int = Field(default=64, ge=1)
    classifier_learning_rate: float = Field(default=1e-4, gt=0)
    classifier_weight_decay: float = Field(default=1e-2, ge=0)
    classifier_base_width: int = Field(default=64, ge=4)
    classifier_blocks: Tuple[int, int, int, int] = (3, 4, 6, 3)
    timestep_embedding_dim: int = Field(default=128, ge=8)
    guidance_scale: float = Field(
        default=10.0,
        ge=0,
        le=50,
        description="Engineering default within the paper's evaluated [0, 50] range; tune per model.",
    )
    num_generations: int = Field(default=10_000, ge=1)
    l2_bands: Dict[str, SimilarityBand] = Field(
        default_factory=dict,
        description="Inclusive normalized-L2 bands interpreted in the configured image_range coordinates.",
    )
    similarity_bands: Dict[str, SimilarityBand] = Field(
        default_factory=dict,
        description="Inclusive bands for a higher-is-more-similar pairwise score, such as SSCD.",
    )

    @model_validator(mode="after")
    def validate_sample_and_cluster_counts(self) -> "SIDEConfig":
        """Reject impossible cluster and classifier layouts."""
        if self.clusters > self.synthetic_samples:
            raise ValueError("clusters cannot exceed synthetic_samples.")
        if 2 * self.min_cluster_size > self.synthetic_samples:
            raise ValueError("synthetic_samples must fit at least two clusters of min_cluster_size samples.")
        if any(block_count < 1 for block_count in self.classifier_blocks):
            raise ValueError("classifier_blocks entries must be positive.")
        return self
