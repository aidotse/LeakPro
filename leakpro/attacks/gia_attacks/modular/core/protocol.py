#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Client protocols: how a client's local training turns its batch into the payload the server sees.

``FedSGD`` sends the gradient of one training loss at the server weights. ``FedAvg`` runs local
epochs of differentiable SGD-style steps and sends the weight change. Both run the forward through
``Objective.loss`` so the same protocol serves classifiers, tabular heads and detectors.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torch.func import functional_call

from leakpro.attacks.gia_attacks.modular.core.interfaces import BatchSampler, ClientProtocol, ModelFn, Objective
from leakpro.attacks.gia_attacks.modular.core.meta import MetaOptimizer, MetaSGD

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo


def _sequential_sampler() -> BatchSampler:
    # Imported here: the components package imports core.protocol (target inference reads the protocol).
    from leakpro.attacks.gia_attacks.modular.components.samplers import SequentialSampler  # noqa: PLC0415 -- circular import

    return SequentialSampler()


def _functional(model: torch.nn.Module, params: Mapping[str, Tensor], buffers: Mapping[str, Tensor]) -> ModelFn:
    """``model`` called at ``params``/``buffers`` instead of its own state, with any call signature."""

    def model_fn(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 -- any model call signature
        state: dict[str, Tensor] = {}
        state.update(params)
        state.update(buffers)
        return functional_call(model, state, args, kwargs)

    return model_fn


@dataclass(frozen=True)
class ProtocolOutput:
    """The payload a protocol produces.

    Attributes:
        gradients: The payload, one tensor per trainable parameter in ``model.parameters()`` order.
            FedSGD: ``dL/dθ``; FedAvg: the weight change ``θ_E - θ_0``. A parameter the model froze is
            not part of it.
        first_loss: The training loss of the first local step (at the starting weights).
        buffers: The model's buffers (e.g. BN running statistics) after the client's training, detached,
            by name; ``None`` when the protocol leaves them at their starting values. The model the
            protocol was handed is never modified: this is the only place the trained buffers appear.

    """

    gradients: list[Tensor]
    first_loss: Tensor
    buffers: Mapping[str, Tensor] | None = None


@dataclass(frozen=True)
class FedSGD(ClientProtocol):
    """One gradient of the training loss at the server weights.

    Only trainable parameters are differentiated (``torch.autograd.grad`` requires every input to
    require grad, and some architectures freeze a subset by construction); a parameter the loss
    does not reach gets a zero gradient. The forward runs on a private clone of the buffers, so the
    given module is never modified; the buffers after the step are returned in ``ProtocolOutput.buffers``.

    Raises:
        ValueError: From ``observe`` for an epoch-prefixed input: one step has no epoch axis. This is
            the earliest point the axis is known, since a ``Task`` is built before any input shape.

    """

    def local_steps(self, num_samples: int) -> int:  # noqa: ARG002 -- one step regardless of the batch
        """One step: a single forward and gradient at the server weights."""
        return 1

    def observe(
        self,
        model: torch.nn.Module,
        objective: Objective,
        model_input: Tensor,
        target: TargetInfo,
        *,
        create_graph: bool,
        has_epoch_prefix: bool,
        base_params: Mapping[str, Tensor] | None = None,  # noqa: ARG002 -- FedSGD differentiates at the model's own weights, identical to the pristine ones
        base_buffers: Mapping[str, Tensor] | None = None,  # noqa: ARG002 -- see base_params
    ) -> ProtocolOutput:
        """Return ``dL/dθ`` of ``objective`` on ``model`` at its current weights."""
        if has_epoch_prefix:
            raise ValueError("FedSGD takes one step on one batch; it cannot be combined with an epoch axis (use FedAvg).")
        buffers = OrderedDict((k, v.detach().clone()) for k, v in model.named_buffers())
        model_fn = _functional(model, dict(model.named_parameters()), buffers)
        loss = objective.loss(model_fn, model_input, target)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, trainable_params, create_graph=create_graph, allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, trainable_params)]
        return ProtocolOutput(
            gradients=list(grads),
            first_loss=loss if create_graph else loss.detach(),
            buffers=OrderedDict((k, v.detach()) for k, v in buffers.items()),
        )


@dataclass(frozen=True)
class FedAvg(ClientProtocol):
    """Local epochs of differentiable updates; the payload is the weight change ``θ_E - θ_0``.

    The buffers (e.g. BN running statistics) start from the starting snapshot and accumulate over every
    local step, as on a real client; the result is returned in ``ProtocolOutput.buffers``. Epochs iterate
    minibatches drawn by ``sampler``. With an epoch-prefixed input ``[E, N, ...]`` epoch
    ``e`` trains on slice ``e``; otherwise every epoch trains on the same batch.

    Attributes:
        local_epochs: Number of local epochs.
        batch_size: Minibatch size; ``None`` trains on the whole batch in one step per epoch.
        lr: Learning rate of the local optimizer.
        optimizer: Differentiable optimizer class, built as ``optimizer(lr=lr)`` per observation.
        sampler: Minibatch order within an epoch.

    """

    local_epochs: int = 1
    batch_size: int | None = None
    lr: float = 0.01
    optimizer: type[MetaOptimizer] = MetaSGD
    sampler: BatchSampler = field(default_factory=_sequential_sampler)

    def local_steps(self, num_samples: int) -> int:
        """``local_epochs * ceil(num_samples / batch_size)``: one step per minibatch per epoch."""
        bs = self.batch_size if self.batch_size is not None else num_samples
        return self.local_epochs * math.ceil(num_samples / bs)

    def observe(
        self,
        model: torch.nn.Module,
        objective: Objective,
        model_input: Tensor,
        target: TargetInfo,
        *,
        create_graph: bool,
        has_epoch_prefix: bool,
        base_params: Mapping[str, Tensor] | None = None,
        base_buffers: Mapping[str, Tensor] | None = None,
    ) -> ProtocolOutput:
        """Simulate the local epochs functionally on ``model`` and return the weight change."""
        base_params = dict(model.named_parameters()) if base_params is None else base_params
        base_buffers = dict(model.named_buffers()) if base_buffers is None else base_buffers

        # Fresh leaves so each call is independent and the starting weights are never mutated.
        # A parameter the model froze stays frozen: it is not trained and not part of the payload.
        trainable = [k for k in base_params if model.get_parameter(k).requires_grad]
        current_params: dict[str, Tensor] = {
            k: v.detach().clone().requires_grad_(k in trainable) for k, v in base_params.items()
        }
        meta_opt = self.optimizer(lr=self.lr)
        first_loss: Tensor | None = None
        # Running statistics from a previous call never leak into this one; within the call they accumulate.
        buffers = OrderedDict({k: v.clone() for k, v in base_buffers.items()})

        for e in range(self.local_epochs):
            recon_e = model_input[e] if has_epoch_prefix else model_input
            # Only a target allocated with the axis prefix carries an epoch dimension.
            target_e = target.index_axis(0, e) if target.axis_ndim > 0 else target

            num_samples = recon_e.shape[0]
            bs = self.batch_size if self.batch_size is not None else num_samples

            for b_idx in self.sampler.yield_batches(num_samples, bs, epoch=e, device=recon_e.device):
                model_fn = _functional(model, current_params, buffers)
                loss_b = objective.loss(model_fn, recon_e[b_idx], target_e.index_axis(0, b_idx))
                if first_loss is None:
                    first_loss = loss_b
                current_params = meta_opt.step(loss_b, current_params, create_graph=create_graph)

        if first_loss is None:
            raise ValueError("FedAvg ran no batches: the sampler yielded none (is the input empty?).")
        delta_w = [current_params[name] - base_params[name].detach() for name in trainable]
        if not create_graph:
            delta_w = [dw.detach() for dw in delta_w]
            first_loss = first_loss.detach()
        return ProtocolOutput(gradients=delta_w, first_loss=first_loss,
                              buffers=OrderedDict((k, v.detach()) for k, v in buffers.items()))
