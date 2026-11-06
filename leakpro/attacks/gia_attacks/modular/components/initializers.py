#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Starting iterates for representations.

A ``seed`` of ``None`` draws from the global RNG, which ``run_attack`` seeds, so the draw follows the
run's seed. An integer seed draws on CPU from a fresh ``torch.Generator`` on every call and then moves
the result to the device: the draw is reproducible, independent of the device, and leaves the global
RNG untouched. The initializer keeps no generator, so running the same attack twice starts from the
same iterate.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from leakpro.attacks.gia_attacks.modular.core.interfaces import SPACES, Initializer, Space


def _check_space(space: str) -> None:
    if space not in SPACES:
        raise ValueError(f"space must be 'data' or 'model_input', got {space!r}")


def _generator(seed: int | None) -> torch.Generator | None:
    return None if seed is None else torch.Generator().manual_seed(seed)


@dataclass(frozen=True)
class Uniform(Initializer):
    """Uniform samples on ``[low, high)``."""

    low: float = 0.0
    high: float = 1.0
    space: Space = "data"
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate the space."""
        _check_space(self.space)

    def sample(self, shape: tuple[int, ...], device: torch.device) -> Tensor:
        """Uniform draw of ``shape`` on ``device``."""
        if self.seed is None:
            return torch.empty(shape, device=device).uniform_(self.low, self.high)
        return torch.empty(shape).uniform_(self.low, self.high, generator=_generator(self.seed)).to(device)


@dataclass(frozen=True)
class Gaussian(Initializer):
    """Gaussian samples with the given mean and standard deviation."""

    mean: float = 0.0
    std: float = 1.0
    space: Space = "model_input"
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate the space."""
        _check_space(self.space)

    def sample(self, shape: tuple[int, ...], device: torch.device) -> Tensor:
        """Gaussian draw of ``shape`` on ``device``."""
        if self.seed is None:
            return torch.randn(shape, device=device) * self.std + self.mean
        return (torch.randn(shape, generator=_generator(self.seed)) * self.std + self.mean).to(device)


@dataclass(frozen=True)
class FromTensor(Initializer):
    """A fixed starting tensor, e.g. a recorded initial iterate to replay a trajectory from."""

    tensor: Tensor
    space: Space = "model_input"

    def __post_init__(self) -> None:
        """Validate the space."""
        _check_space(self.space)

    def sample(self, shape: tuple[int, ...], device: torch.device) -> Tensor:
        """An independent copy of the tensor; its shape must equal ``shape``."""
        if tuple(self.tensor.shape) != tuple(shape):
            raise ValueError(f"FromTensor holds shape {tuple(self.tensor.shape)}, but {tuple(shape)} was requested")
        return self.tensor.detach().clone().to(device)


@dataclass(frozen=True)
class FromState(Initializer):
    """Start from the reconstruction already in the state (e.g. the previous stage's output)."""

    space: Space = "data"

    def __post_init__(self) -> None:
        """Validate the space."""
        _check_space(self.space)

    def sample(self, shape: tuple[int, ...], device: torch.device) -> Tensor:  # noqa: ARG002 -- Initializer interface; the shape comes from the stored tensor
        """Always raises: the representation must take its start from the state, not from a draw."""
        raise ValueError("FromState draws nothing: the representation must take its start from the state")
