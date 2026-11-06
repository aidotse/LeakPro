#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Timestep-dependent blending weight schedules for AdaptiveMeanOptimization."""
from __future__ import annotations

import abc
import math


class BlendingSchedule(abc.ABC):
    """Returns the blend weight gamma_t in [0, 1] at reverse-diffusion timestep t."""

    @abc.abstractmethod
    def gamma(self, t: int, num_timesteps: int) -> float:
        """Blend weight at timestep ``t`` out of ``num_timesteps`` total steps."""


class ConstantSchedule(BlendingSchedule):
    """Fixed blend weight, independent of the timestep."""

    def __init__(self, gamma: float = 0.5) -> None:
        self._gamma = gamma

    def gamma(self, t: int, num_timesteps: int) -> float:  # noqa: ARG002 -- Schedule interface; a constant gamma ignores t and num_timesteps
        """Return the fixed blend weight; ``t``/``num_timesteps`` are ignored."""
        return self._gamma


class LinearDecaySchedule(BlendingSchedule):
    """gamma_t = t / (T - 1) — full weight on mu* at t=T-1, none at t=0."""

    def gamma(self, t: int, num_timesteps: int) -> float:
        """Linearly interpolate the blend weight from 0 at t=0 to 1 at t=T-1."""
        return t / max(num_timesteps - 1, 1)


class CosineDecaySchedule(BlendingSchedule):
    """Smooth cosine ramp with the same endpoints as LinearDecaySchedule."""

    def gamma(self, t: int, num_timesteps: int) -> float:
        """Cosine-interpolate the blend weight from 0 at t=0 to 1 at t=T-1."""
        frac = t / max(num_timesteps - 1, 1)
        return 0.5 * (1 - math.cos(math.pi * frac))
