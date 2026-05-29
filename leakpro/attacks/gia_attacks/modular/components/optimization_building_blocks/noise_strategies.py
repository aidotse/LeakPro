#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Noise injection strategies for diffusion-based gradient inversion.

These strategies determine what noise is added at each step of the DDPM
reverse process.  The noise is added *after* the mean has been adjusted by a
:class:`~leakpro.attacks.gia_attacks.modular.components.
optimization_building_blocks.mean_strategies.MeanAdjustmentStrategy`.

Classes
-------
* :class:`NoiseInjectionStrategy` — abstract base
* :class:`StandardNoiseInjection` — eps ~ N(0, I)  (classic DDPM / GGDM)
* :class:`GradientAlignedNoiseInjection` — project eps onto the
  gradient-residual direction ``delta_mu = mu* - mu_theta``  (GANI from
  GradInvDiff, Wang et al. 2024)

References
----------
* Wang et al., "GradInvDiff: Stealing Medical Privacy in Federated
  Learning via Diffusion-Based Gradient Inversion", 2024.  Eq. 7.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from leakpro.attacks.gia_attacks.modular.config.registry import register


class NoiseInjectionStrategy(ABC):
    """How noise is sampled / transformed at each DDPM reverse step."""

    @abstractmethod
    def sample(
        self,
        shape: tuple[int, ...],
        delta_mu: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample (possibly modified) noise for the reverse step.

        Args:
            shape: Shape of noise to sample [N, C, H, W].
            delta_mu: Direction ``mu_adjusted - mu_theta`` from the mean
                strategy.  Ignored by strategies that don't use it.
            device: Target device.

        Returns:
            Noise tensor [N, C, H, W].
        """
        ...


@register("noise.standard")
class StandardNoiseInjection(NoiseInjectionStrategy):
    """Standard Gaussian noise eps ~ N(0, I).

    Used by the classic DDPM ancestral sampler and GGDM (Gu et al. 2024).
    """

    def sample(
        self,
        shape: tuple[int, ...],
        delta_mu: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.randn(shape, device=device)

    def __repr__(self) -> str:
        return "StandardNoiseInjection()"


@register("noise.gani")
class GradientAlignedNoiseInjection(NoiseInjectionStrategy):
    """GANI: project noise onto the gradient-residual direction ``delta_mu``.

    Implements Eq. 7 from Wang et al. (2024).  Instead of sampling
    isotropic Gaussian noise, GANI aligns the noise with the direction
    ``delta_mu = mu* - mu_theta`` so that the stochastic step continues to
    push the sample toward the gradient-inversion optimum.

    When ``delta_mu`` is near-zero (no meaningful gradient direction),
    falls back to standard Gaussian noise.

    Args:
        fallback_threshold: If ``||delta_mu||^2 < threshold``, return
            standard noise instead of the projection.
    """

    def __init__(self, fallback_threshold: float = 1e-12) -> None:
        self.fallback_threshold = fallback_threshold

    def sample(
        self,
        shape: tuple[int, ...],
        delta_mu: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        eps = torch.randn(shape, device=device)
        delta_norm_sq = delta_mu.pow(2).sum()

        if delta_norm_sq.item() < self.fallback_threshold:
            return eps  # No meaningful direction → standard noise

        # Project eps onto the delta_mu direction: eps_proj = (eps · delta_mu / ||delta_mu||^2) * delta_mu
        dot = (eps * delta_mu).sum()
        return (dot / delta_norm_sq) * delta_mu

    def __repr__(self) -> str:
        return f"GradientAlignedNoiseInjection(threshold={self.fallback_threshold})"


__all__ = [
    "NoiseInjectionStrategy",
    "StandardNoiseInjection",
    "GradientAlignedNoiseInjection",
]
