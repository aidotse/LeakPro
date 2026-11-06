#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""The full observation a server intercepts from one simulated client."""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from torch import Tensor, nn

from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.interfaces import Modality
    from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo
    from leakpro.attacks.gia_attacks.modular.core.task import Task


def bn_running_stats(
    model: nn.Module, buffers: Mapping[str, Tensor] | None = None,
) -> list[tuple[Tensor, Tensor]] | None:
    """Detached copies of every BatchNorm layer's ``(running_mean, running_var)``, in module order; None without BN.

    ``buffers`` (by name, as ``model.named_buffers()``) replaces the model's own values where given.
    """
    buffers = buffers or {}
    stats = [
        (
            buffers.get(f"{name}.running_mean" if name else "running_mean", m.running_mean).detach().clone(),
            buffers.get(f"{name}.running_var" if name else "running_var", m.running_var).detach().clone(),
        )
        for name, m in model.named_modules()
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.running_mean is not None
    ]
    return stats or None


def simulate_client(
    model: nn.Module,
    data: Tensor,
    target: TargetInfo,
    *,
    task: Task,
    modality: Modality,
    input_shape: tuple[int, ...] | None = None,
) -> ClientObservation:
    """What the server intercepts when a client trains on (data, target); the caller's model is never modified.

    Fills gradients, pre_bn_stats (running statistics before training), bn_stats (after),
    batch_size (data.shape[0] without an epoch prefix, else input_shape[0]) and loss_value (first local step).
    Models without BatchNorm get None for both BN fields.

    ``data`` is the client's real data; ``modality.client_to_model_input`` maps it to the model input.
    Pass ``input_shape`` (the canonical per-epoch shape) when ``data`` is epoch-prefixed ``[E, N, ...]``. Every field is
    filled; the attack's threat model strips what the attacker may not see.
    """
    has_epoch_prefix = input_shape is not None and data.ndim > len(input_shape)
    pre_bn_stats = bn_running_stats(model)
    out = task.victim_run(model, modality.client_to_model_input(data), target, input_shape=input_shape)
    return ClientObservation(
        gradients=[g.detach().clone() for g in out.gradients],
        bn_stats=bn_running_stats(model, out.buffers),
        pre_bn_stats=pre_bn_stats,
        batch_size=input_shape[0] if has_epoch_prefix else data.shape[0],
        loss_value=float(out.first_loss),
    )
