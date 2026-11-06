#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Named parallel axes (seed, epoch) that prefix the shape of the optimized tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Axis:
    """Represents a single generative/optimization dimension."""

    name: str
    size: int


class AxisManager:
    """Manages dynamic tensor dimensions for generation and aggregation.

    Axes form a prefix prepended to (N, C, H, W) when allocating or slicing
    reconstruction tensors.  An empty AxisManager yields an empty prefix shape
    () so single-seed/single-epoch code is unaffected.
    """

    def __init__(self, *axes: Axis) -> None:
        self.axes: Tuple[Axis, ...] = tuple(axes)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Prefix shape contributed by all registered axes."""
        return tuple(axis.size for axis in self.axes)

    def get_dim(self, name: str) -> Optional[int]:
        """Tensor dimension index for the named axis, or None if absent."""
        for i, axis in enumerate(self.axes):
            if axis.name == name:
                return i
        return None

    def get_size(self, name: str) -> int:
        """Size of the named axis, defaulting to 1 when absent."""
        for axis in self.axes:
            if axis.name == name:
                return axis.size
        return 1

    def __repr__(self) -> str:
        """Lists each axis as ``name=size``."""
        inner = ", ".join(f"{a.name}={a.size}" for a in self.axes)
        return f"AxisManager({inner})"

    def __eq__(self, other: object) -> bool:
        """Equal when the axes match."""
        if not isinstance(other, AxisManager):
            return NotImplemented
        return self.axes == other.axes
