#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Time-dependent blending schedules for diffusion-based gradient inversion.

These schedules control how strongly the gradient-information-adjusted mean
(mu*) is blended with the UNet's original posterior mean (mu_theta) at each
reverse-process timestep.

Convention: ``t`` counts *down* from ``T-1`` to ``0`` (reverse-process order).
``gamma`` should be **high** at large ``t`` (early, coarse structure) and
**low** at small ``t`` (late, fine detail).

Classes
-------
* :class:`BlendingSchedule` — abstract base
* :class:`ConstantSchedule` — fixed gamma for all timesteps (GGDM default)
* :class:`LinearDecaySchedule` — linear decay from gamma_max to gamma_min
* :class:`CosineDecaySchedule` — cosine annealing from gamma_max to gamma_min
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


class BlendingSchedule(ABC):
    """Time-dependent weight gamma_t in [0, 1] for mean blending / scaling.

    Called once per reverse-process timestep ``t`` (counting down from T-1
    to 0).  The return value is used differently depending on the mean
    strategy:

    * :class:`~leakpro.attacks.gia_attacks.modular.components.
      optimization_building_blocks.mean_strategies.SimilarityGuidance` —
      multiplies the guidance scale ``gamma`` (via :class:`ConstantSchedule`).
    * :class:`~leakpro.attacks.gia_attacks.modular.components.
      optimization_building_blocks.mean_strategies.AdaptiveMeanOptimization` —
      blends mu* and mu_theta:
      ``mu_bar = gamma_t * mu*  +  (1 - gamma_t) * mu_theta``.
    """

    @abstractmethod
    def __call__(self, t: int, T: int) -> float:
        """Return gamma_t for timestep *t* out of *T* total steps.

        Args:
            t: Current timestep, counting down from ``T-1`` to ``0``.
            T: Total number of diffusion steps.

        Returns:
            Scalar blending weight ``gamma_t`` in [0, 1].
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ConstantSchedule(BlendingSchedule):
    """Fixed gamma for all timesteps (GGDM default).

    Args:
        gamma: Constant blending weight returned at every timestep.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        self.gamma = gamma

    def __call__(self, t: int, T: int) -> float:
        return self.gamma

    def __repr__(self) -> str:
        return f"ConstantSchedule(gamma={self.gamma})"


class LinearDecaySchedule(BlendingSchedule):
    """Linear decay from ``gamma_max`` at ``t=T-1`` to ``gamma_min`` at ``t=0``.

    Default (1 → 0) matches GradInvDiff Eq. 6.

    Args:
        gamma_max: Upper bound (returned when ``t = T-1``).
        gamma_min: Lower bound (returned when ``t = 0``).
    """

    def __init__(self, gamma_max: float = 1.0, gamma_min: float = 0.0) -> None:
        self.gamma_max = gamma_max
        self.gamma_min = gamma_min

    def __call__(self, t: int, T: int) -> float:
        if T <= 1:
            return self.gamma_min
        frac = t / (T - 1)
        return self.gamma_min + frac * (self.gamma_max - self.gamma_min)

    def __repr__(self) -> str:
        return f"LinearDecaySchedule(max={self.gamma_max}, min={self.gamma_min})"


class CosineDecaySchedule(BlendingSchedule):
    """Cosine annealing from ``gamma_max`` to ``gamma_min``.

    Uses a half-cosine curve so the decay starts slowly at large ``t``,
    accelerates in the middle, and slows again near ``t=0``.

    Args:
        gamma_max: Upper bound (returned when ``t = T-1``).
        gamma_min: Lower bound (returned when ``t = 0``).
    """

    def __init__(self, gamma_max: float = 1.0, gamma_min: float = 0.0) -> None:
        self.gamma_max = gamma_max
        self.gamma_min = gamma_min

    def __call__(self, t: int, T: int) -> float:
        if T <= 1:
            return self.gamma_min
        frac = t / (T - 1)
        cos_val = 0.5 * (1.0 + math.cos(math.pi * (1.0 - frac)))
        return self.gamma_min + cos_val * (self.gamma_max - self.gamma_min)

    def __repr__(self) -> str:
        return f"CosineDecaySchedule(max={self.gamma_max}, min={self.gamma_min})"


__all__ = [
    "BlendingSchedule",
    "ConstantSchedule",
    "LinearDecaySchedule",
    "CosineDecaySchedule",
]
