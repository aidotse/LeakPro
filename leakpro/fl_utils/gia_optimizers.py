"""Optimizer objects used for GIA training to allow graph utilization through multiple epochs."""
import time
from abc import ABC, abstractmethod
from collections import OrderedDict

from torch import Tensor, zeros_like
from torch.autograd import grad

from leakpro.utils.import_helper import Dict, Self, Tuple


class MetaOptimizer(ABC):
    """Abstract Meta Optimizer."""

    def __init__(self: "MetaOptimizer") -> None:
        """Initialize the MetaOptimizer."""
        raise NotImplementedError("This is an abstract class and should not be instantiated directly.")

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

    def __init__(self: Self, lr: float=1e-2) -> None:
        """Init."""
        self.lr = lr

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

        # Compute gradients only for grad params
        grads = grad(loss, params.values(), retain_graph=True, create_graph=True, only_inputs=True, allow_unused=True)
        
           
        return OrderedDict(
    (name, param - self.lr * (grad_part if grad_part is not None else param))
    for ((name, param), grad_part) in zip(params.items(), grads))
        


class MetaAdam(MetaOptimizer):
    """Implementation of Adam which perform step to a new set of parameters."""

    def __init__(self: Self, lr: float = 1e-2, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-08,
                 weight_decay: float = 0) -> None:
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
        gradients = grad(loss, [p for p in params.values() if p.requires_grad],
                         retain_graph=True, create_graph=True, only_inputs=True)

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
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (gradient ** 2)

            m_hat = self.m[name] / (1 - self.beta1**self.t)
            v_hat = self.v[name] / (1 - self.beta2**self.t)

            adam_grad = m_hat / (v_hat.sqrt() + self.eps)

            new_params[name] = param - self.lr * adam_grad
        return new_params
