#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""LeakPro input handler for the CIFAR-10 extraction example."""

from __future__ import annotations

from typing import ClassVar

from torch import Tensor, nn

from leakpro import AbstractExtractionInputHandler
from leakpro.attacks.extraction_attacks.protocols import ExtractionAdapter, FeatureTransform


class CIFAR10ExtractionHandler(AbstractExtractionInputHandler):
    """Provide the notebook's target and authorized references to LeakPro."""

    adapter: ClassVar[ExtractionAdapter[Tensor] | None] = None
    references: ClassVar[Tensor | None] = None
    feature_extractor: ClassVar[nn.Module | None] = None
    feature_transform: ClassVar[nn.Module | None] = None

    @classmethod
    def configure(
        cls,
        *,
        adapter: ExtractionAdapter[Tensor],
        references: Tensor,
        feature_extractor: nn.Module,
        feature_transform: nn.Module,
    ) -> None:
        """Set the objects required by the extraction scheduler."""
        cls.adapter = adapter
        cls.references = references
        cls.feature_extractor = feature_extractor
        cls.feature_transform = feature_transform

    def get_diffusion_adapter(self) -> ExtractionAdapter[Tensor]:
        """Return the trained unconditional DDPM adapter."""
        if self.adapter is None:
            raise RuntimeError("Configure CIFAR10ExtractionHandler before creating LeakPro.")
        return self.adapter

    def get_extraction_reference_images(self) -> Tensor:
        """Return the authorized target-training references."""
        if self.references is None:
            raise RuntimeError("Configure CIFAR10ExtractionHandler before creating LeakPro.")
        return self.references

    def get_side_feature_extractor(self) -> nn.Module:
        """Return the frozen feature model used to build SIDE labels."""
        if self.feature_extractor is None:
            raise RuntimeError("Configure CIFAR10ExtractionHandler before creating LeakPro.")
        return self.feature_extractor

    def get_side_feature_transform(self) -> FeatureTransform:
        """Return preprocessing paired with the SIDE feature model."""
        if self.feature_transform is None:
            raise RuntimeError("Configure CIFAR10ExtractionHandler before creating LeakPro.")
        return self.feature_transform
