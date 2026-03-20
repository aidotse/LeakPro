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
        self: Self, lr: float = 1e-2, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0
    ) -> None:
        """Initializes the MetaAdam optimizer.

        Args:
        ----
            lr (float, optional): Learning rate. Default is 1e-2.
            betas (Tuple[float, float], optional): Coefficients used for computing running averages of gradient and its square.
            Default is (0.9, 0.999).
            eps (float, optional): Term added to the denominator to improve numerical stability. Default is 1e-08.
            weight_decay (float, optional): Weight decay (L2 penalty). Default is 0.

        """
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

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

        # Initialize m and v
        if not self.m:
            self.m = {name: zeros_like(param) for name, param in params.items()}
            self.v = {name: zeros_like(param) for name, param in params.items()}
        self.t += 1
        new_params = OrderedDict()
        for (name, param), gradient in zip(params.items(), gradients):
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * gradient
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (gradient**2)

            m_hat = self.m[name] / (1 - self.beta1**self.t)
            v_hat = self.v[name] / (1 - self.beta2**self.t)

            adam_grad = m_hat / (v_hat.sqrt() + self.eps)

            new_params[name] = param - self.lr * adam_grad
        return new_params
