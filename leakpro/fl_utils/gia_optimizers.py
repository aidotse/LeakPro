#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Optimizer objects used for GIA training to allow graph utilization through multiple epochs."""

from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
from torch import Tensor, zeros_like
from torch.autograd import grad

from leakpro.utils.import_helper import Dict, Self, Tuple


class MetaOptimizer(ABC):
    """Abstract Meta Optimizer."""

    def __init__(self: "MetaOptimizer") -> None:
        """Initialize the MetaOptimizer."""
        self.foreach = False
        self.lr = None
        raise NotImplementedError("This is an abstract class and should not be instantiated directly.")

    def reset(self: "MetaOptimizer") -> None:
        """Reset the optimizer state."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def step(self: "MetaOptimizer", loss: Tensor, params: Dict[str, Tensor]) -> OrderedDict[str, Tensor]:
        """Perform a single optimization step.

        Args:
        ----
            loss (torch.Tensor): The loss value calculated from the model's output.
            params (Dict[str, torch.Tensor]): A dictionary of model parameters to be updated.

        Returns:
        -------
            OrderedDict[str, torch.Tensor]: A new set of parameters which have been updated.

        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def update_params(
        self: "MetaOptimizer", p_list: list[Tensor], g_list: list[Tensor], params: OrderedDict
    ) -> OrderedDict[str, Tensor]:
        """Need text."""

        # Update in a way similar to torch.optim.SGD
        if self.foreach and hasattr(torch, "_foreach_add"):
            # Out-of-place foreach op, returns new tensors and keeps graph
            new_p_list = torch._foreach_add(p_list, g_list, alpha=-self.lr)
        else:
            new_p_list = [p - self.lr * g for p, g in zip(p_list, g_list)]

        new_param_iter = iter(new_p_list)
        updated_params = OrderedDict()

        for name, param in params.items():
            if param.requires_grad:
                updated_params[name] = next(new_param_iter)
            else:
                # Leave params that do not require grad as they are
                updated_params[name] = param

        return updated_params

    @staticmethod
    def _validate_init_shapes(init_state: Dict[str, Tensor], name: str, params: Dict[str, Tensor]) -> None:
        """Validate that a warm-start state dict matches the parameters by key and shape.

        Catches the silent-broadcast failure mode: a right-key/wrong-shape tensor would otherwise
        be broadcast during the moment update and produce quietly incorrect optimization.
        """
        for param_name, param in params.items():
            if param_name not in init_state:
                raise ValueError(f"{name} is missing key '{param_name}'")
            if init_state[param_name].shape != param.shape:
                raise ValueError(
                    f"{name}['{param_name}'] shape {tuple(init_state[param_name].shape)} "
                    f"!= param shape {tuple(param.shape)}"
                )


class MetaSGD(MetaOptimizer):
    """Implementation of SGD which perform step to a new set of parameters."""

    def __init__(self: Self, lr: float = 1e-2, foreach: bool = True) -> None:
        """Init."""
        self.lr = lr
        self.foreach = foreach

    def reset(self: Self) -> None:
        """Reset the optimizer state."""
        pass

    def step(self: Self, loss: Tensor, params: Dict[str, Tensor]) -> OrderedDict[str, Tensor]:
        """Perform a single optimization step.

        Args:
        ----
            loss (torch.Tensor): The loss value calculated from the model's output.
            params (Dict[str, torch.Tensor]): A dictionary of model parameters to be updated.

        Returns:
        -------
            OrderedDict[str, torch.Tensor]: A new set of parameters which have been updated.

        """
        grad_params = [(name, param) for name, param in params.items() if param.requires_grad]

        p_list = [param for _, param in grad_params]

        # Compute gradients only for grad params
        grads = grad(loss, p_list, retain_graph=True, create_graph=True, only_inputs=True, allow_unused=True)

        # Match PyTorch behavior better: None grads should behave like zero update
        g_list = []
        for p, g in zip(p_list, grads):
            if g is None:
                g_list.append(torch.zeros_like(p))
            else:
                g_list.append(g)

        # update

        # Update in a way similar to torch.optim.SGD
        if self.foreach and hasattr(torch, "_foreach_add"):
            # Out-of-place foreach op, returns new tensors and keeps graph
            new_p_list = torch._foreach_add(p_list, g_list, alpha=-self.lr)
        else:
            new_p_list = [p - self.lr * g for p, g in zip(p_list, g_list)]

        new_param_iter = iter(new_p_list)
        updated_params = OrderedDict()

        for name, param in params.items():
            if param.requires_grad:
                updated_params[name] = next(new_param_iter)
            else:
                # Leave params that do not require grad as they are
                updated_params[name] = param

        return updated_params


class MetaAdam(MetaOptimizer):
    """Implementation of Adam which perform step to a new set of parameters."""

    def __init__(
        self: Self,
        lr: float = 1e-2,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0,
        foreach: bool = False,
        m_init: Dict[str, Tensor] = None,
        v_init: Dict[str, Tensor] = None,
        t_init: int = 0,
    ) -> None:
        """Initializes the MetaAdam optimizer.

        Args:
        ----
            lr (float, optional): Learning rate. Default is 1e-2.
            betas (Tuple[float, float], optional): Coefficients used for computing running averages of gradient and its square.
            Default is (0.9, 0.999).
            eps (float, optional): Term added to the denominator to improve numerical stability. Default is 1e-08.
            weight_decay (float, optional): Weight decay (L2 penalty). Default is 0.
            foreach (bool): operator
            m_init (Dict[str, Tensor], optional): Initial first-moment state, keyed by parameter name. If None, m starts
                at zero (default). Must be provided together with v_init.
            v_init (Dict[str, Tensor], optional): Initial second-moment state, keyed by parameter name. If None, v starts
                at zero (default). Must be provided together with m_init and be non-negative.
            t_init (int, optional): Initial step counter, used for Adam's bias correction when warm-starting. Default is 0.

        """
        self.lr = lr
        self.weight_decay = weight_decay
        self.foreach = foreach
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps

        # Optional warm-start state (e.g. a client's initial optimizer state). Detach once so the
        # initial state is treated as a constant: it carries no graph back to the reconstruction
        # data and is safe to reuse across all simulations in a GIA run. Shapes are validated lazily
        # on the first step(), once the parameter shapes are known.
        if (m_init is None) != (v_init is None):
            raise ValueError("Provide both m_init and v_init, or neither.")
        if v_init is not None and any(bool((t < 0).any()) for t in v_init.values()):
            raise ValueError("v_init must be non-negative (it is a second moment, used under sqrt).")
        self.m_init = {k: v.detach() for k, v in m_init.items()} if m_init is not None else None
        self.v_init = {k: v.detach() for k, v in v_init.items()} if v_init is not None else None
        self.t_init = t_init

        self.m = {}
        self.v = {}
        self.t = t_init

    def reset(self: Self) -> None:
        """Reset the optimizer state (m, v, t) for a new training simulation.

        State is cleared here and re-seeded lazily on the next step() (from m_init/v_init if
        provided, otherwise zeros), so reset() does not need the parameter shapes.
        """
        self.m = {}
        self.v = {}
        self.t = self.t_init

    def step(self: Self, loss: Tensor, params: Dict[str, Tensor]) -> OrderedDict[str, Tensor]:
        """Perform a single optimization step.

        Args:
        ----
            loss (torch.Tensor): The loss value calculated from the model's output.
            params (Dict[str, torch.Tensor]): A dictionary of model parameters to be updated.

        Returns:
        -------
            OrderedDict[str, torch.Tensor]: A new set of parameters which have been updated.

        """
        gradients = grad(
            loss, [p for p in params.values() if p.requires_grad], retain_graph=True, create_graph=True, only_inputs=True
        )

        if self.weight_decay != 0:
            gradients = [grad + self.weight_decay * param for grad, param in zip(gradients, params.values())]

        # Initialize m and v (from warm-start state if provided, otherwise zeros)
        if not self.m:
            if self.m_init is not None:
                self._validate_init_shapes(self.m_init, "m_init", params)
                self._validate_init_shapes(self.v_init, "v_init", params)
                # New dicts so this simulation's in-loop reassignments never touch m_init/v_init.
                self.m = dict(self.m_init)
                self.v = dict(self.v_init)
            else:
                self.m = {name: zeros_like(param) for name, param in params.items()}
                self.v = {name: zeros_like(param) for name, param in params.items()}
        self.t += 1
        new_params = OrderedDict()
        for (name, param), gradient in zip(params.items(), gradients):
            # Mirror torch.optim.Adam's _single_tensor_adam fused op order exactly so the
            # result is bit-identical. Out-of-place ops (mul/add/addcmul/addcdiv) keep the
            # autograd graph intact for GIA differentiability.
            self.m[name] = self.m[name].lerp(gradient, 1 - self.beta1)
            self.v[name] = self.v[name].mul(self.beta2).addcmul(gradient, gradient, value=1 - self.beta2)

            bias_correction1 = 1 - self.beta1**self.t
            bias_correction2 = 1 - self.beta2**self.t
            step_size = self.lr / bias_correction1
            bias_correction2_sqrt = bias_correction2**0.5

            denom = (self.v[name].sqrt() / bias_correction2_sqrt).add(self.eps)
            new_params[name] = param.addcdiv(self.m[name], denom, value=-step_size)

        return new_params


class MetaMomentum(MetaOptimizer):
    """Implementation of SGD with momentum which perform step to a new set of parameters."""

    def __init__(
        self: Self,
        foreach: bool = False,
        lr: float = 1e-2,
        beta: float = 0.9,
        weight_decay: float = 0,
        m_init: Dict[str, Tensor] = None,
        t_init: int = 0,
    ) -> None:
        """Initializes the MetaMomentum optimizer.

        Args:
        ----
            foreach (bool): params
            lr (float, optional): Learning rate. Default is 1e-2.
            beta (float, optional): Coefficients used for computing running averages of gradient .
            Default is 0.9.
            weight_decay (float, optional): decay rate of gradient updates.
            m_init (Dict[str, Tensor], optional): Initial momentum buffer, keyed by parameter name (only parameters that
                require grad). If None, momentum starts at zero (default).
            t_init (int, optional): Initial step counter. Default is 0. (Momentum has no bias correction, so this only
                affects the diagnostic step count.)

        """
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta = beta
        self.foreach = foreach

        # Optional warm-start momentum. Detach once so it is a constant reusable across all
        # simulations in a GIA run; shapes are validated lazily on the first step().
        self.m_init = {k: v.detach() for k, v in m_init.items()} if m_init is not None else None
        self.t_init = t_init

        self.m = {}
        self.t = t_init

    def reset(self: Self) -> None:
        """Reset the optimizer state (m, t) for a new training simulation.

        State is cleared here and re-seeded lazily on the next step() (from m_init if provided,
        otherwise zeros), so reset() does not need the parameter shapes.
        """
        self.m = {}
        self.t = self.t_init

    def step(self: Self, loss: Tensor, params: Dict[str, Tensor]) -> OrderedDict[str, Tensor]:
        """Perform a single optimization step.

        Args:
        ----
            loss (torch.Tensor): The loss value calculated from the model's output.
            params (Dict[str, torch.Tensor]): A dictionary of model parameters to be updated.

        Returns:
        -------
            OrderedDict[str, torch.Tensor]: A new set of parameters which have been updated.

        """
        grad_params = [(name, param) for name, param in params.items() if param.requires_grad]

        p_list = [param for _, param in grad_params]

        # Compute gradients only for grad params
        grads = grad(loss, p_list, retain_graph=True, create_graph=True, only_inputs=True, allow_unused=True)

        # Match PyTorch behavior better: None grads should behave like zero update
        g_list = []
        for p, g in zip(p_list, grads):
            if g is None:
                g_list.append(torch.zeros_like(p))
            else:
                g_list.append(g)
        # Initialize m (from warm-start state if provided, otherwise zeros)
        if not self.m:
            if self.m_init is not None:
                grad_param_dict = OrderedDict(grad_params)
                self._validate_init_shapes(self.m_init, "m_init", grad_param_dict)
                # New dict so this simulation's in-loop reassignments never touch m_init.
                self.m = dict(self.m_init)
            else:
                self.m = {name: zeros_like(param) for name, param in grad_params}
        self.t += 1
        for (name, param), gradient in zip(grad_params, g_list):
            self.m[name] = self.beta * self.m[name] + gradient
        g_list = [self.m[n] for n in self.m]
        # new_params[name] = param - self.lr * self.m[name]
        return self.update_params(p_list, g_list, params)
