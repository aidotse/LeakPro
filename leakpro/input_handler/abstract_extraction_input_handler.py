#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""User-supplied boundaries for diffusion extraction attacks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from torch import Tensor, nn

from leakpro.attacks.extraction_attacks.protocols import (
    ExtractionAdapter,
    FeatureTransform,
    PairwiseScore,
)


class AbstractExtractionInputHandler(ABC):
    """Provide model-specific objects without assuming a classifier target."""

    @abstractmethod
    def get_diffusion_adapter(self) -> ExtractionAdapter[Tensor]:
        """Return the audited generator through a supported adapter."""

    def get_extraction_metadata(self) -> dict[str, Any] | None:
        """Declare target/sampler configs for custom adapters, or use the adapter's getter.

        Return ``{"target": {...}, "sampler": {...}}`` describing effective
        architecture, sampling, conditioning and non-state behavior. Accurate
        declarations are required for custom components. Carlini supplies the
        actual mode via the conditional keyword if this method accepts it.
        """
        return None

    def get_extraction_conditions(self) -> Sequence[Any] | None:
        """Return prompts or labels for conditional Carlini extraction."""
        return None

    def get_extraction_reference_images(self) -> Tensor | None:
        """Return authorized training references when verification is enabled."""
        return None

    def get_side_feature_extractor(self) -> nn.Module | None:
        """Return SIDE's frozen feature extractor."""
        return None

    def get_side_feature_transform(self) -> FeatureTransform | None:
        """Return preprocessing compatible with the feature extractor."""
        return None

    def get_side_feature_metadata(self) -> dict[str, Any] | None:
        """Declare extractor/preprocessing architecture, nested settings and revisions.

        Return ``{"extractor": {...}, "transform": {...}}``. The transform
        descriptor may be omitted only for the default identity transform.
        Module weights and buffers are hashed separately by SIDE.
        """
        return None

    def get_extraction_reference_score(self) -> PairwiseScore | None:
        """Return an optional pairwise similarity scorer, such as SSCD.

        When SIDE uses similarity bands, set its reference_score_id to identify
        this scorer and its settings in the saved report.
        """
        return None
