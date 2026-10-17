#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Narrow model-stack boundaries used by both attacks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, Protocol, TypeVar, runtime_checkable

from torch import Tensor

ConditionGradient = Callable[[Tensor, Tensor, Tensor], Tensor]


SampleBatch_co = TypeVar("SampleBatch_co", covariant=True)


@runtime_checkable
class ExtractionAdapter(Protocol[SampleBatch_co]):
    """Sampling contract independent of data modality."""

    def sample(
        self,
        batch_size: int,
        *,
        conditions: Sequence[Any] | None,
        seed: int,
    ) -> SampleBatch_co:
        """Generate a batch using the requested conditions and seed."""


FeatureTransform = Callable[[Tensor], Tensor]
PairwiseScore = Callable[[Tensor, Tensor], Tensor]


def identity_feature_transform(images: Tensor) -> Tensor:
    """Return images unchanged."""
    return images
