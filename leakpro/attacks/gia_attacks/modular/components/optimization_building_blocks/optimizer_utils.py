#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Shared utilities for optimizer logging and loss computation.

These helpers are used by both :class:`~leakpro.attacks.gia_attacks.modular.\
components.composable_optimizer.ComposableOptimizer` and
:class:`~leakpro.attacks.gia_attacks.modular.components.diffusion_optimizer.\
DiffusionSamplingOptimizer` so that loss breakdowns and progress lines are
formatted identically regardless of the optimizer type.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, List, Tuple

import torch

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.loss_components import (
        LossComponent,
    )
    from leakpro.attacks.gia_attacks.modular.core.state import RunContext

logger = logging.getLogger(__name__)


def compute_loss_components(
    loss_components: List[LossComponent],
    reconstruction: torch.Tensor,
    labels: torch.Tensor,
    target_gradients: List[torch.Tensor],
    ctx: "RunContext",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute all loss components and return the total plus a per-component dict.

    Args:
        loss_components: List of :class:`LossComponent` instances.
        reconstruction: Current reconstruction tensor in data space.
        labels: Labels for the reconstruction.
        target_gradients: True gradients to match.
        ctx: Run context providing target_model and loss_fn.

    Returns:
        ``(total_loss, {component_name: loss_tensor})`` where
        ``total_loss = sum(component losses)``.

    """
    losses: dict[str, torch.Tensor] = {}
    for component in loss_components:
        losses[component.name] = component.compute(
            reconstruction=reconstruction,
            labels=labels,
            target_gradients=target_gradients,
            ctx=ctx,
        )
    total_loss = sum(losses.values())
    return total_loss, losses


def setup_loss_components(
    loss_components: List[LossComponent],
    reconstruction: torch.Tensor,
    ctx: "RunContext",
) -> None:
    """Call ``setup()`` on every loss component before the optimisation loop.

    Components whose ``setup()`` is a no-op (the default) pay no cost.
    Components that need per-stage initialisation (e.g.
    :class:`~leakpro.attacks.gia_attacks.modular.components.
    optimization_building_blocks.loss_components.BNStatisticsRegularization`)
    override ``setup()`` to perform their work.

    Shared by :class:`~leakpro.attacks.gia_attacks.modular.components.
    composable_optimizer.ComposableOptimizer` and
    :class:`~leakpro.attacks.gia_attacks.modular.components.
    diffusion_optimizer.DiffusionSamplingOptimizer`.

    Args:
        loss_components: All loss components for the optimizer.
        reconstruction: Initial reconstruction tensor (forwarded to components
            that need it, e.g. for batch-size inference).
        ctx: Run context providing model, observations, simulator, and dataloader.

    """
    for component in loss_components:
        component.setup(ctx=ctx, reconstruction=reconstruction)


# Backward-compatible alias — prefer setup_loss_components in new code.
setup_bn_components = setup_loss_components


def log_progress(
    logger_obj: logging.Logger,
    prefix: str,
    total_loss: float,
    losses: dict[str, float],
    suffix: str = "",
    latent_code: torch.Tensor | None = None,
) -> None:
    """Log a standardised loss-breakdown line.

    Format::

        {prefix}: total_loss={:.4f}  {name1}={:.4f}  {name2}={:.4f} ...

    Args:
        logger_obj: Logger instance to write to.
        prefix: Text before the colon, e.g. ``"Iteration 42/1000"`` or
            ``"t= 999/1000"``.
        total_loss: Scalar total loss value (tensor or float both accepted).
        losses: Per-component loss values (tensors or floats both accepted).
        suffix: Optional extra text appended at the end of the log line.
        latent_code: Optional latent code tensor to log statistics for.

    """
    # Use scientific notation for all losses to match paper precision
    msg = f"  {prefix}: total_loss={float(total_loss):.12e}"
    for name, val in losses.items():
        msg += f"  {name}={float(val):.12e}"
    if suffix:
        msg += f"  {suffix}"
    if latent_code is not None:
        z_mean = latent_code.mean().item()
        z_std = latent_code.std().item()
        z_norm = torch.norm(latent_code).item()
        msg += f"  [z_mean={z_mean:.6f}, z_std={z_std:.6f}, z_norm={z_norm:.6f}]"
    logger_obj.info(msg)


def run_inner_adam_loop(
    init_tensor: torch.Tensor,
    loss_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, dict]],
    steps: int,
    lr: float,
    clamp_range: tuple[float, float] | None = (-1.0, 1.0),
) -> tuple[torch.Tensor, float, dict[str, float]]:
    """Run a short inner Adam optimisation loop on a detached copy of a tensor.

    Used by :class:`~leakpro.attacks.gia_attacks.modular.components.
    optimization_building_blocks.mean_strategies.AdaptiveMeanOptimization`
    to clone a tensor, run K Adam steps, track the best iterate, and
    optionally clamp to a valid range.

    Args:
        init_tensor: Tensor to use as starting point.  Cloned + detached
            internally so the original is never modified.
        loss_fn: Callable that accepts the current tensor and returns
            ``(total_loss, component_losses_dict)``.  Must produce a
            differentiable scalar loss.
        steps: Number of inner Adam iterations.
        lr: Adam learning rate.
        clamp_range: If not ``None``, clamp the tensor to ``[min, max]``
            after each step.  Keeps the optimisation on the image manifold
            when working in DDPM ``[-1, 1]`` space.

    Returns:
        ``(best_tensor, best_loss, best_component_losses)`` where
        ``best_tensor`` is detached and ``best_component_losses`` maps each
        component name to its scalar float value at the best iterate.

    """
    x_var = init_tensor.detach().clone().requires_grad_(True)
    inner_opt = torch.optim.Adam([x_var], lr=lr)

    best_x = x_var.detach().clone()
    best_loss: float = float("inf")
    best_component_losses: dict[str, float] = {}

    for _ in range(steps):
        inner_opt.zero_grad()
        loss, component_losses = loss_fn(x_var)
        loss.backward()
        inner_opt.step()

        if clamp_range is not None:
            with torch.no_grad():
                x_var.clamp_(clamp_range[0], clamp_range[1])

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_component_losses = {k: v.item() for k, v in component_losses.items()}
            best_x = x_var.detach().clone()

    return best_x, best_loss, best_component_losses


__all__ = [
    "compute_loss_components",
    "setup_loss_components",
    "setup_bn_components",
    "log_progress",
    "run_inner_adam_loop",
]
