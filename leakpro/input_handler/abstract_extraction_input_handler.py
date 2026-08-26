#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""User-supplied boundaries for diffusion extraction attacks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

from torch import Tensor, nn

from leakpro.attacks.extraction_attacks.protocols import (
    DiffusionAdapter,
    FeatureTransform,
    PairwiseScore,
    SamplingAdapter,
)


class AbstractExtractionInputHandler(ABC):
    """Provide model-specific objects without assuming a classifier target."""

    @abstractmethod
    def get_diffusion_adapter(self) -> SamplingAdapter | DiffusionAdapter:
        """Return the audited generator through a supported adapter."""

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

    def get_side_classifier(self) -> nn.Module | None:
        """Return an optional preconstructed time-dependent classifier."""
        return None

    def get_side_classifier_factory(self) -> Callable[[int, int], nn.Module] | None:
        """Return an optional time-dependent classifier factory."""
        return None

    def get_extraction_reference_score(self) -> PairwiseScore | None:
        """Return an optional pairwise similarity scorer, such as SSCD."""
        return None
