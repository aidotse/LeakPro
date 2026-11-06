#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Meta-learning utilities for multi-epoch client simulation.

Provides a stateless functional model wrapper (MetaModule) and differentiable
meta-optimizers (MetaSGD, MetaAdam) that build a computation graph through
multiple simulated client training steps.  This enables the attacker to model
FedAvg parameter updates (ΔW = W_new − W_old) instead of single-step gradients.
"""
from __future__ import annotations

import abc
from collections import OrderedDict
from typing import Optional

import torch
from torch import Tensor, nn
from torch.autograd import grad
from torch.func import functional_call


class MetaModule(nn.Module):
    """Run a network with a caller-supplied parameter dict (stateless).

    Uses torch.func.functional_call under the hood, which avoids monkey-patching
    and is compatible with modern architectures including ConvNeXt and ViT.

    The ``parameters`` instance attribute (an OrderedDict) shadows nn.Module's
    ``parameters()`` method intentionally so that the initial parameter snapshot
    can be read as a plain attribute at the call site.
    """

    def __init__(self, net: nn.Module, device: Optional[torch.device] = None) -> None:
        super().__init__()
        if device is None:
            try:
                device = next(iter(net.parameters())).device
            except StopIteration:
                device = torch.device("cpu")
        self.net = net.to(device)
        # Intentionally shadows nn.Module.parameters() — see class docstring.
        self.parameters: OrderedDict[str, Tensor] = OrderedDict(self.net.named_parameters())
        self.buffers: OrderedDict[str, Tensor] = OrderedDict(self.net.named_buffers())

    def forward(self, x: Tensor, parameters: dict[str, Tensor]) -> Tensor:
        """Forward pass using caller-supplied parameters and existing buffers."""
        state: dict = {}
        state.update(parameters)
        state.update(self.buffers)
        return functional_call(self.net, state, (x,))


class MetaOptimizer(abc.ABC):
    """Abstract base for differentiable meta-optimizers."""

    @abc.abstractmethod
    def step(
        self,
        loss: Tensor,
        params: dict[str, Tensor],
        create_graph: bool = True,
    ) -> OrderedDict[str, Tensor]:
        """Return a new parameter dict updated by one gradient step on loss.

        Args:
            loss:         Scalar loss to differentiate.
            params:       Current parameter dict (trainable and non-trainable).
            create_graph: If True, build the second-order computation graph so
                          the returned parameters are differentiable w.r.t. the
                          inputs that produced loss.  Set to False for the
                          victim's initial payload to avoid locking graphs in VRAM.

        """

    def reset(self) -> None:  # noqa: B027 -- optional hook, no-op by default
        """Reset any internal optimizer state (e.g. Adam moment estimates)."""


class MetaSGD(MetaOptimizer):
    """Differentiable SGD — computes gradients through the parameter update."""

    def __init__(self, lr: float = 1e-2, foreach: bool = True) -> None:
        self.lr = lr
        self.foreach = foreach

    def reset(self) -> None:
        """Does nothing because SGD keeps no state."""
        pass  # SGD is stateless

    def step(
        self,
        loss: Tensor,
        params: dict[str, Tensor],
        create_graph: bool = True,
    ) -> OrderedDict[str, Tensor]:
        """Takes one differentiable SGD step on the trainable parameters."""
        grad_params = [(name, p) for name, p in params.items() if p.requires_grad]
        p_list = [p for _, p in grad_params]

        grads = grad(
            loss, p_list,
            retain_graph=create_graph,
            create_graph=create_graph,
            only_inputs=True,
            allow_unused=True,
        )

        g_list = [torch.zeros_like(p) if g is None else g
                  for p, g in zip(p_list, grads)]

        if self.foreach and hasattr(torch, "_foreach_add"):
            new_p_list = torch._foreach_add(p_list, g_list, alpha=-self.lr)
        else:
            new_p_list = [p - self.lr * g for p, g in zip(p_list, g_list)]

        new_param_iter = iter(new_p_list)
        updated: OrderedDict[str, Tensor] = OrderedDict()
        for name, param in params.items():
            updated[name] = next(new_param_iter) if param.requires_grad else param
        return updated


class MetaAdam(MetaOptimizer):
    """Differentiable Adam — builds a graph through moment-corrected updates."""

    def __init__(
        self,
        lr: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m: dict[str, Tensor] = {}
        self.v: dict[str, Tensor] = {}
        self.t: int = 0

    def reset(self) -> None:
        """Clears the moment estimates and step count."""
        self.m = {}
        self.v = {}
        self.t = 0

    def step(
        self,
        loss: Tensor,
        params: dict[str, Tensor],
        create_graph: bool = True,
    ) -> OrderedDict[str, Tensor]:
        """Takes one differentiable Adam step on the trainable parameters."""
        trainable = {n: p for n, p in params.items() if p.requires_grad}
        grads = grad(
            loss, list(trainable.values()),
            retain_graph=create_graph,
            create_graph=create_graph,
            only_inputs=True,
        )

        if self.weight_decay:
            grads = tuple(g + self.weight_decay * p
                          for g, p in zip(grads, trainable.values()))

        if not self.m:
            self.m = {n: torch.zeros_like(p) for n, p in trainable.items()}
            self.v = {n: torch.zeros_like(p) for n, p in trainable.items()}

        self.t += 1
        updated: OrderedDict[str, Tensor] = OrderedDict()
        grad_iter = iter(grads)
        for name, param in params.items():
            if not param.requires_grad:
                updated[name] = param
                continue
            g = next(grad_iter)
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * g ** 2
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            updated[name] = param - self.lr * m_hat / (v_hat.sqrt() + self.eps)
        return updated
