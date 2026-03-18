"""Optimizer objects used for GIA training to allow graph utilization through multiple epochs."""

from abc import ABC, abstractmethod
from collections import OrderedDict

from torch import Tensor, zeros_like
import torch
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

    def __init__(self: Self, lr: float = 1e-2, foreach: bool = True) -> None:
        """Init."""
        self.lr = lr
        self.foreach = foreach

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
    """Differentiable Adam that aims to mirror torch.optim.Adam as closely as possible."""

    def __init__(
        self: Self,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        foreach: bool | None = None,
        fused: bool | None = None,
        maximize: bool = False,
        capturable: bool = False,
        decoupled_weight_decay: bool = False,
        differentiable: bool = True,
    ) -> None:
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.foreach = foreach
        self.fused = fused
        self.maximize = maximize
        self.capturable = capturable
        self.decoupled_weight_decay = decoupled_weight_decay
        self.differentiable = differentiable

        self.m: Dict[str, Tensor] = {}
        self.v: Dict[str, Tensor] = {}
        self.vmax: Dict[str, Tensor] = {}
        self.steps: Dict[str, Tensor] = {}

    def step(self: Self, loss: Tensor, params: Dict[str, Tensor]) -> OrderedDict[str, Tensor]:
        """Perform one differentiable Adam step and return updated params."""
        grad_params = [(name, p) for name, p in params.items() if p.requires_grad]

        all_names = [name for name, _ in grad_params]
        all_p_list = [p for _, p in grad_params]

        grads = grad(
            loss,
            all_p_list,
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
            allow_unused=True,
        )

        active = [(name, p, g) for (name, p), g in zip(grad_params, grads) if g is not None]

        names = [name for name, _, _ in active]
        p_list = [p for _, p, _ in active]
        g_list = [g for _, _, g in active]

        # Lazy init optimizer state
        for name, p in grad_params:
            if name not in self.m:
                self.m[name] = torch.zeros_like(p)
                self.v[name] = torch.zeros_like(p)
                if self.amsgrad:
                    self.vmax[name] = torch.zeros_like(p)
                # Match torch.optim.Adam initialization more closely
                # If capturable or fused: keep step tensor on param device
                # Otherwise: CPU scalar tensor
                if self.capturable or self.fused:
                    self.steps[name] = torch.zeros((), dtype=torch.float32, device=p.device)
                else:
                    self.steps[name] = torch.tensor(0.0, dtype=torch.float32)

        # Clone so we produce updated tensors out-of-place
        new_p_list = [p.clone() for p in p_list]
        exp_avgs = [self.m[name].clone() for name in names]
        exp_avg_sqs = [self.v[name].clone() for name in names]
        max_exp_avg_sqs = [self.vmax[name].clone() for name in names] if self.amsgrad else []
        # state_steps = [self.steps[name].clone() for name in names]
        state_steps = [self.steps[name] for name in names]

        # Functional Adam call, closest path to torch.optim.Adam behavior.
        adam_kwargs = dict(
            params=new_p_list,
            grads=g_list,
            exp_avgs=exp_avgs,
            exp_avg_sqs=exp_avg_sqs,
            max_exp_avg_sqs=max_exp_avg_sqs,
            state_steps=state_steps,
            foreach=self.foreach,
            capturable=self.capturable,
            differentiable=self.differentiable,
            fused=self.fused,
            grad_scale=None,
            found_inf=None,
            has_complex=any(torch.is_complex(p) for p in new_p_list),
            amsgrad=self.amsgrad,
            beta1=self.beta1,
            beta2=self.beta2,
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=self.eps,
            maximize=self.maximize,
        )

        # Newer PyTorch versions support this kwarg.
        # Keep it guarded so the code is easier to adapt across versions.
        try:
            torch.optim._functional.adam(
                decoupled_weight_decay=self.decoupled_weight_decay,
                **adam_kwargs,
            )
        except TypeError:
            # Older PyTorch versions do not support decoupled_weight_decay here.
            # In that case this will behave like classic Adam, not AdamW-style decay.
            if self.decoupled_weight_decay:
                raise RuntimeError(
                    "This PyTorch version does not support decoupled_weight_decay "
                    "in torch.optim._functional.adam. Upgrade PyTorch to match AdamW-style behavior."
                )
            torch.optim._functional.adam(**adam_kwargs)

        # Persist state
        for i, name in enumerate(names):
            self.m[name] = exp_avgs[i]
            self.v[name] = exp_avg_sqs[i]
            self.steps[name] = state_steps[i]
            if self.amsgrad:
                self.vmax[name] = max_exp_avg_sqs[i]

        updated_params = OrderedDict(params)
        for name, new_p in zip(names, new_p_list):
            updated_params[name] = new_p

        return updated_params
