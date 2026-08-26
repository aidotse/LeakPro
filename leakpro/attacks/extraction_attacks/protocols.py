#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Narrow model-stack boundaries used by both attacks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, Protocol, runtime_checkable

from torch import Tensor

ConditionGradient = Callable[[Tensor, Tensor, Tensor], Tensor]


@runtime_checkable
class SamplingAdapter(Protocol):
    """Minimum black-box generation surface required by Carlini extraction."""

    image_shape: tuple[int, int, int]

    def sample(
        self,
        batch_size: int,
        *,
        conditions: Sequence[Any] | None,
        seed: int,
    ) -> Tensor:
        """Generate a BCHW image batch using the requested conditions and seed."""


@runtime_checkable
class DiffusionAdapter(SamplingAdapter, Protocol):
    """White-box diffusion operations required by the classifier-guided SIDE branch."""

    num_timesteps: int

    def validate_side_capabilities(self) -> None:
        """Fail before attack work if required white-box operations are unavailable."""

    def classifier_timesteps(self, timesteps: Tensor) -> Tensor:
        """Map raw diffusion indices to the time representation used by guidance."""

    def q_sample(self, clean_images: Tensor, timesteps: Tensor, noise: Tensor) -> Tensor:
        """Apply the target diffusion model's forward noising process."""

    def sample_with_classifier_guidance(
        self,
        batch_size: int,
        *,
        labels: Tensor,
        gradient_fn: ConditionGradient,
        seed: int,
    ) -> Tensor:
        """Run reverse diffusion while adding gradient_fn to the target score."""


FeatureTransform = Callable[[Tensor], Tensor]
PairwiseScore = Callable[[Tensor, Tensor], Tensor]


def identity_feature_transform(images: Tensor) -> Tensor:
    """Return images unchanged."""
    return images
