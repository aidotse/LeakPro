#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Registry factories for PyTorch optimizers and LR schedulers.

Each factory returns a **builder callable** — a function that accepts
``(param_groups, lr, max_iterations)`` and returns the configured
optimizer or scheduler.  This two-step pattern is necessary because
PyTorch optimizers need ``param_groups`` at construction time, which
is not available when the :class:`~...config.spec.ComponentSpec` is
parsed.

Usage::

    # Schema
    optimizer_type=ComponentSpec(type="optimizer.adam")
    # Builder
    optimizer_builder = build_component(stage.optimizer_type)
    optimizer = optimizer_builder(param_groups, lr=learning_rate, max_iterations=max_iterations)

Registered keys
---------------
Optimizers: ``optimizer.adam``, ``optimizer.sgd``, ``optimizer.lbfgs``
Schedulers: ``scheduler.cosine``, ``scheduler.step``,
            ``scheduler.exponential``, ``scheduler.cosine_warmup``
"""

from __future__ import annotations

import math
from typing import Any, Callable

import torch

from leakpro.attacks.gia_attacks.modular.config.registry import register

# ---------------------------------------------------------------------------
# Optimizer factories
# ---------------------------------------------------------------------------

@register("optimizer.adam")
def _adam_factory(**kwargs: Any) -> Callable:
    """Factory for :class:`torch.optim.Adam`."""
    def build(param_groups: list, lr: float, **_: Any) -> torch.optim.Adam:
        return torch.optim.Adam(param_groups, lr=lr, **kwargs)
    return build


@register("optimizer.sgd")
def _sgd_factory(momentum: float = 0.0, **kwargs: Any) -> Callable:
    """Factory for :class:`torch.optim.SGD`."""
    def build(param_groups: list, lr: float, **_: Any) -> torch.optim.SGD:
        return torch.optim.SGD(param_groups, lr=lr, momentum=momentum, **kwargs)
    return build


@register("optimizer.lbfgs")
def _lbfgs_factory(max_iter: int = 20, history_size: int = 100, **kwargs: Any) -> Callable:
    """Factory for :class:`torch.optim.LBFGS`.

    LBFGS does not support per-parameter options, so ``param_groups`` is
    flattened into a single parameter list.
    """
    def build(param_groups: list, lr: float, **_: Any) -> torch.optim.LBFGS:
        all_params = [p for group in param_groups for p in group["params"]]
        return torch.optim.LBFGS(
            all_params, lr=lr, max_iter=max_iter, history_size=history_size, **kwargs
        )
    return build


# ---------------------------------------------------------------------------
# Scheduler factories
# ---------------------------------------------------------------------------

@register("scheduler.cosine")
def _cosine_scheduler_factory(**kwargs: Any) -> Callable:
    """Factory for :class:`torch.optim.lr_scheduler.CosineAnnealingLR`."""
    def build(optimizer: torch.optim.Optimizer, max_iterations: int) -> Any:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_iterations, **kwargs
        )
    return build


@register("scheduler.step")
def _step_scheduler_factory(**kwargs: Any) -> Callable:
    """Factory for :class:`torch.optim.lr_scheduler.MultiStepLR`.

    Milestones default to the GIFD paper's schedule fractions.
    """
    def build(optimizer: torch.optim.Optimizer, max_iterations: int) -> Any:
        milestones = kwargs.pop(
            "milestones",
            [
                max_iterations // 2 + max_iterations // 2 - max_iterations // 2,
                max_iterations // 2,
                round(max_iterations * 0.875),
            ],
        )
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                max_iterations // 2 + max_iterations // 2 - max_iterations // 2,
                int(max_iterations // 1.6),
                int(max_iterations // 1.142),
            ],
            gamma=kwargs.pop("gamma", 0.1),
            **kwargs,
        )
    return build


@register("scheduler.exponential")
def _exponential_scheduler_factory(gamma: float = 0.99, **kwargs: Any) -> Callable:
    """Factory for :class:`torch.optim.lr_scheduler.ExponentialLR`."""
    def build(optimizer: torch.optim.Optimizer, max_iterations: int) -> Any:
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma, **kwargs)
    return build


@register("scheduler.cosine_warmup")
def _cosine_warmup_scheduler_factory(
    rampup: float = 0.05,
    rampdown: float = 0.75,
    **kwargs: Any,
) -> Callable:
    """Factory for the GIFD/GIAS cosine-warmup schedule (LambdaLR)."""
    def build(optimizer: torch.optim.Optimizer, max_iterations: int) -> Any:
        def lr_lambda(step: int) -> float:
            t = step / max_iterations
            lr_ramp = min(1.0, (1.0 - t) / rampdown)
            lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
            return lr_ramp * min(1.0, t / rampup)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return build


__all__ = [
    "_adam_factory",
    "_sgd_factory",
    "_lbfgs_factory",
    "_cosine_scheduler_factory",
    "_step_scheduler_factory",
    "_exponential_scheduler_factory",
    "_cosine_warmup_scheduler_factory",
]
