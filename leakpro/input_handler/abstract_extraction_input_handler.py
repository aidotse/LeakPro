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
    """Provide the model and other inputs for an extraction audit.

    Report identity assumes a fixed reference dataset. Modules contribute the
    tensor state used by LeakPro's model hash. Plain callbacks contribute their
    qualified name. Custom adapters can supply a sampler dictionary or the
    legacy ``{"target": ..., "sampler": ...}`` envelope through the optional
    ``get_extraction_metadata(*, guided)`` method. SIDE asks for both guided and
    unguided settings. Return ``None`` or raise ``NotImplementedError`` if a mode
    has no metadata. Configure any conditioning-specific settings on the
    adapter before constructing the audit.

    Use a separate output directory when changing the reference dataset or
    when changing behavior outside the saved state and declared settings.
    """

    @abstractmethod
    def get_diffusion_adapter(self) -> ExtractionAdapter[Tensor]:
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

    def get_extraction_reference_score(self) -> PairwiseScore | None:
        """Return an optional pairwise similarity scorer, such as SSCD."""
        return None
