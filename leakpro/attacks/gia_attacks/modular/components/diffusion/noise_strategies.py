#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Noise injection strategies for the stochastic (DDPM) reverse-diffusion step."""
from __future__ import annotations

import abc

import torch
from torch import Tensor


class NoiseInjectionStrategy(abc.ABC):
    """Samples the noise term added at each DDPM reverse step."""

    @abc.abstractmethod
    def sample(
        self, shape: tuple[int, ...], device: torch.device, dtype: torch.dtype, delta_mu: Tensor
    ) -> Tensor:
        """Sample the noise term added at the current reverse-diffusion step."""


class StandardNoiseInjection(NoiseInjectionStrategy):
    """Plain N(0, I) noise (GGDM)."""

    def sample(
        self, shape: tuple[int, ...], device: torch.device, dtype: torch.dtype, delta_mu: Tensor  # noqa: ARG002 -- NoiseStrategy interface; delta_mu is unused here
    ) -> Tensor:
        """Sample standard Gaussian noise; ``delta_mu`` is unused."""
        return torch.randn(shape, device=device, dtype=dtype)


class GradientAlignedNoiseInjection(NoiseInjectionStrategy):
    """GANI (GradInvDiff).

    Projects standard noise onto the delta_mu direction so stochastic
    perturbations push the trajectory toward the mean-adjustment strategy's
    optimum instead of a random direction.

    The projection is over the whole ``delta_mu`` tensor (every image and seed together), so one scalar
    scales the noise of the whole batch along that single direction.

    Falls back to standard noise when delta_mu's squared norm is below
    ``fallback_threshold`` (no meaningful direction to align to).
    """

    def __init__(self, fallback_threshold: float = 1e-6) -> None:
        self.fallback_threshold = fallback_threshold

    def sample(self, shape: tuple[int, ...], device: torch.device, dtype: torch.dtype, delta_mu: Tensor) -> Tensor:
        """Sample noise, projected onto ``delta_mu``'s direction when it is non-negligible."""
        eps = torch.randn(shape, device=device, dtype=dtype)
        norm_sq = delta_mu.pow(2).sum()
        if norm_sq.item() < self.fallback_threshold:
            return eps
        direction = delta_mu / norm_sq.sqrt()
        scalar_projection = (eps * direction).sum()
        return scalar_projection * direction
