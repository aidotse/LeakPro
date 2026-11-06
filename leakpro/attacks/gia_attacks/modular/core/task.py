#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""The client task: a training objective run under a client protocol, in a given module mode."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from leakpro.attacks.gia_attacks.modular.core.interfaces import MODEL_MODES, ClientProtocol, ModelMode, Objective
from leakpro.attacks.gia_attacks.modular.core.protocol import FedSGD, ProtocolOutput
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.state import Context, TargetInfo


@dataclass(frozen=True)
class Task:
    """What the client computes: ``objective`` trained under ``protocol`` with the model in ``model_mode``.

    The same task produces the victim's observed signal (``victim_signal``) and the attacker's
    simulated signal on a reconstruction (``simulate_dummy_signal``), so both sides always run the
    same objective, protocol and mode.

    Attributes:
        objective: The client's training loss.
        protocol: How the loss becomes the observed payload (gradient or weight update).
        model_mode: ``"train"`` or ``"eval"``, the module mode of the simulated client model.

    Raises:
        ValueError: If ``protocol`` is not one of ``objective.supported_protocols``, or
            ``model_mode`` is not a known mode.

    """

    objective: Objective
    protocol: ClientProtocol = field(default_factory=FedSGD)
    model_mode: ModelMode = "train"

    def __post_init__(self) -> None:
        """Checks that the objective supports the protocol and the model mode is valid."""
        supported = self.objective.supported_protocols
        if type(self.protocol) not in supported:
            names = ", ".join(p.__name__ for p in supported)
            raise ValueError(
                f"{type(self.objective).__name__} does not support the {type(self.protocol).__name__} protocol; "
                f"supported: {names}."
            )
        if self.model_mode not in MODEL_MODES:
            raise ValueError(f"model_mode must be one of {MODEL_MODES}, got {self.model_mode!r}.")

    def victim_signal(
        self,
        model: torch.nn.Module,
        model_input: Tensor,
        target: TargetInfo,
        *,
        input_shape: tuple[int, ...] | None = None,
    ) -> ClientObservation:
        """The payload the victim client sends after training on ``(model_input, target)``.

        The client trains a private copy of ``model`` put in ``model_mode`` by the objective, so
        the caller's model is never modified. Pass ``input_shape`` (the canonical per-epoch shape)
        when ``model_input`` is epoch-prefixed ``[E, N, ...]``; a ``model_input`` of higher rank
        than ``input_shape`` carries an epoch prefix, and omitting it declares none.
        """
        out = self.victim_run(model, model_input, target, input_shape=input_shape)
        return ClientObservation(gradients=out.gradients)

    def victim_run(
        self,
        model: torch.nn.Module,
        model_input: Tensor,
        target: TargetInfo,
        *,
        input_shape: tuple[int, ...] | None = None,
    ) -> ProtocolOutput:
        """Train a private copy of ``model`` as the victim client and return the protocol's output.

        The copy exists because the objective puts the model in ``model_mode``; the protocol itself
        never modifies the module. The client's post-training BN statistics are in ``output.buffers``.
        """
        victim = copy.deepcopy(model)
        self.objective.prepare_model(victim, self.model_mode)
        has_epoch_prefix = input_shape is not None and model_input.ndim > len(input_shape)
        return self.protocol.observe(
            victim, self.objective, model_input, target, create_graph=False, has_epoch_prefix=has_epoch_prefix,
        )

    def simulate_dummy_signal(
        self,
        model_input: Tensor,
        target: TargetInfo,
        ctx: Context,
        *,
        create_graph: bool = True,
    ) -> ClientObservation:
        """Simulate the client's training on the attacker's reconstruction.

        The driver has already sliced off its parallel axis, so the epoch prefix that remains is
        exactly what ``ctx.axes`` declares. ``create_graph=True`` keeps the graph so the attack can
        differentiate through the simulation; False returns a detached signal for gradient-free scoring.

        The simulation runs on ``ctx.session.execution_model``, starting from the session's
        pristine server weights and buffers.

        Raises:
            RuntimeError: If ``ctx.session.begin_stage`` has not been called.

        """
        session = ctx.session
        if session.execution_model is None:
            raise RuntimeError("ctx.session.begin_stage(ctx) must be called before simulate_dummy_signal.")
        out = self.protocol.observe(
            session.execution_model, self.objective, model_input, target,
            create_graph=create_graph,
            has_epoch_prefix=ctx.axes.get_dim("epoch") is not None,
            base_params=session.pristine_params,
            base_buffers=session.pristine_buffers,
        )
        return ClientObservation(gradients=out.gradients)
