#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Learning-rate schedules for ``GradientDriver``, each built from the run's optimizer and length."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, LinearLR, LRScheduler, MultiStepLR

from leakpro.attacks.gia_attacks.modular.core.interfaces import Schedule


@dataclass(frozen=True)
class Cosine(Schedule):
    """Cosine annealing from the base learning rate to ``eta_min`` over the whole run."""

    eta_min: float = 0.0

    def build(self, optimizer: torch.optim.Optimizer, max_iter: int) -> LRScheduler:
        """``CosineAnnealingLR`` with ``T_max = max_iter``."""
        return CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=self.eta_min)


@dataclass(frozen=True)
class MultiStep(Schedule):
    """Multiply the learning rate by ``gamma`` at each milestone step.

    ``milestones`` are explicit step indices. Without them, ``divisors`` give the milestones
    ``max_iter // d`` (floor division, so ``4800 // 1.6`` is 2999); without those, the milestones are
    the given ``fractions`` of the run length, rounded down.
    """

    milestones: tuple[int, ...] | None = None
    fractions: tuple[float, ...] = (3 / 8, 5 / 8, 7 / 8)
    gamma: float = 0.1
    divisors: tuple[float, ...] | None = None

    def build(self, optimizer: torch.optim.Optimizer, max_iter: int) -> LRScheduler:
        """``MultiStepLR`` at the explicit or fractional milestones."""
        if self.milestones is not None:
            milestones = self.milestones
        elif self.divisors is not None:
            milestones = [int(max_iter // d) for d in self.divisors]
        else:
            milestones = [int(f * max_iter) for f in self.fractions]
        return MultiStepLR(optimizer, milestones=list(milestones), gamma=self.gamma)


@dataclass(frozen=True)
class Exponential(Schedule):
    """Multiply the learning rate by ``gamma`` after every step."""

    gamma: float = 0.99

    def build(self, optimizer: torch.optim.Optimizer, max_iter: int) -> LRScheduler:  # noqa: ARG002 -- Scheduler interface; a constant schedule ignores max_iter
        """``ExponentialLR`` with ``gamma``."""
        return ExponentialLR(optimizer, gamma=self.gamma)


@dataclass(frozen=True)
class LinearDecay(Schedule):
    """Linear interpolation of the learning-rate factor from ``start_factor`` to ``end_factor`` over the run."""

    start_factor: float = 1.0
    end_factor: float = 0.1

    def build(self, optimizer: torch.optim.Optimizer, max_iter: int) -> LRScheduler:
        """``LinearLR`` with ``total_iters = max(1, max_iter)``."""
        return LinearLR(optimizer, start_factor=self.start_factor, end_factor=self.end_factor,
                        total_iters=max(1, int(max_iter)))
