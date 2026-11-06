#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Mean-adjustment strategies used by DiffusionSamplingDriver.

How the UNet's raw step mean is shifted toward the gradient-matching
objective at each reverse-diffusion step.
"""
from __future__ import annotations

import abc
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.components.diffusion.schedules import BlendingSchedule


class MeanAdjustmentStrategy(abc.ABC):
    """Shifts the UNet's raw step mean toward the gradient-matching objective."""

    @abc.abstractmethod
    def adjust(
        self,
        mu_theta: Tensor,
        xt: Tensor,
        sigma_t: float,
        t: int,
        num_timesteps: int,
        objective: Callable[[Tensor], Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Returns (adjusted_mean, delta_mu) where delta_mu = adjusted_mean - mu_theta.

        ``objective(x)`` is the stage's weighted matching loss at sample ``x`` (a scalar tensor
        differentiable w.r.t. ``x``), built by the driver from the shared stage-loss evaluator.
        """


class SimilarityGuidance(MeanAdjustmentStrategy):
    """GGDM (Gu et al., WWW 2024): mu_adj = mu_theta - gamma * sigma_t^2 * grad_xt(matching_loss).

    Computes the gradient of the matching loss with respect to x_t via a
    single autograd call — the only place in DiffusionSamplingDriver's loop
    where a gradient is computed; the returned tensors are detached before
    this method returns, so no autograd graph survives into the next step.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        self.gamma = gamma

    def adjust(
        self,
        mu_theta: Tensor,
        xt: Tensor,
        sigma_t: float,
        t: int,  # noqa: ARG002 -- MeanStrategy interface; unused by this strategy
        num_timesteps: int,  # noqa: ARG002 -- MeanStrategy interface; unused by this strategy
        objective: Callable[[Tensor], Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Shift the mean by the negative gradient of the matching loss w.r.t. x_t."""
        xt_leaf = xt.detach().clone().requires_grad_(True)
        loss = objective(xt_leaf)
        grad = torch.autograd.grad(loss, xt_leaf)[0].detach()
        delta_mu = -self.gamma * (sigma_t ** 2) * grad
        return (mu_theta + delta_mu).detach(), delta_mu


class AdaptiveMeanOptimization(MeanAdjustmentStrategy):
    """GradInvDiff AMO (Wang et al., MICCAI 2025).

    Runs K inner Adam steps from mu_theta toward a matching-loss-minimizing mu*, then blends
    ``gamma_t * mu* + (1 - gamma_t) * mu_theta`` via a time-dependent schedule.
    """

    def __init__(self, schedule: BlendingSchedule, inner_steps: int = 5, inner_lr: float = 0.01) -> None:
        self.schedule = schedule
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr

    def adjust(
        self,
        mu_theta: Tensor,
        xt: Tensor,  # noqa: ARG002 -- MeanStrategy interface; unused by this strategy
        sigma_t: float,  # noqa: ARG002 -- MeanStrategy interface; unused by this strategy
        t: int,
        num_timesteps: int,
        objective: Callable[[Tensor], Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Run inner Adam steps toward mu*, then blend with mu_theta via the schedule."""
        mu_star = mu_theta.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([mu_star], lr=self.inner_lr)
        for _ in range(self.inner_steps):
            optimizer.zero_grad()
            loss = objective(mu_star)
            loss.backward()
            optimizer.step()
        mu_star = mu_star.detach()

        gamma_t = self.schedule.gamma(t, num_timesteps)
        adjusted_mean = gamma_t * mu_star + (1.0 - gamma_t) * mu_theta
        delta_mu = adjusted_mean - mu_theta
        return adjusted_mean, delta_mu
