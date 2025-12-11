"""Helpers to train with 16-bit precision."""

import numpy as np
import torch as th
from torch import nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from leakpro.utils.logger import logger


def convert_module_to_f16(layer: nn.Module) -> None:
    """Convert primitive modules to float16."""

    if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        layer.weight.data = layer.weight.data.half()
        if layer.bias is not None:
            layer.bias.data = layer.bias.data.half()

def convert_module_to_f32(layer: nn.Module) -> None:
    """Convert primitive modules to float32, undoing convert_module_to_f16()."""
    if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        layer.weight.data = layer.weight.data.float()
        if layer.bias is not None:
            layer.bias.data = layer.bias.data.float()

def make_master_params(param_groups_and_shapes: list) -> list[nn.Parameter]:
    """Copy model parameters into a (differently-shaped) list of full-precision parameters."""
    master_params = []
    for param_group, shape in param_groups_and_shapes:
        master_param = nn.Parameter(
            _flatten_dense_tensors(
                [param.detach().float() for (_, param) in param_group]
            ).view(shape)
        )
        master_param.requires_grad = True
        master_params.append(master_param)
    return master_params


def model_grads_to_master_grads(param_groups_and_shapes: list, master_params: list[nn.Parameter]) -> None:
    """Copy the gradients from the model parameters into the master parameters from make_master_params()."""

    for master_param, (param_group, shape) in zip(
        master_params, param_groups_and_shapes
    ):
        master_param.grad = _flatten_dense_tensors(
            [param_grad_or_zeros(param) for (_, param) in param_group]
        ).view(shape)


def master_params_to_model_params(param_groups_and_shapes: list, master_params: list[nn.Parameter]) -> None:
    """Copy the master parameter data back into the model parameters."""
    # Without copying to a list, if a generator is passed, this will
    # silently not copy any parameters.
    for master_param, (param_group, _) in zip(master_params, param_groups_and_shapes):
        for (_, param), unflat_master_param in zip(
            param_group, unflatten_master_params(param_group, master_param.view(-1))
        ):
            param.detach().copy_(unflat_master_param)


def unflatten_master_params(param_group: list, master_param: th.Tensor) -> list[nn.Parameter]:
    """Unflatten a master parameter tensor into the shapes of the given param_group."""
    return _unflatten_dense_tensors(master_param, [param for (_, param) in param_group])


def get_param_groups_and_shapes(named_model_params: list, params: list[int] | None = None) -> list:
    """From named model parameters, get parameter groups and their shapes."""
    model_params = list(named_model_params)
    if params:
        named_model_params = [model_params[i] for i in params]
    scalar_vector_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim <= 1],
        (-1),
    )
    matrix_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim > 1],
        (1, -1),
    )
    return [scalar_vector_named_params, matrix_named_params]


def master_params_to_state_dict(
    model: nn.Module, param_groups_and_shapes: list, master_params: list[nn.Parameter], use_fp16: bool
) -> dict:
    """Convert master parameters back into a model state dict."""

    if use_fp16:
        state_dict = model.state_dict()
        for master_param, (param_group, _) in zip(
            master_params, param_groups_and_shapes
        ):
            for (name, _), unflat_master_param in zip(
                param_group, unflatten_master_params(param_group, master_param.view(-1))
            ):
                assert name in state_dict
                state_dict[name] = unflat_master_param
    else:
        state_dict = model.state_dict()
        for i, (name, _value) in enumerate(model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
    return state_dict


def state_dict_to_master_params(model: nn.Module, state_dict: dict, use_fp16: bool) -> list[nn.Parameter]:
    """Convert a model state dict into master parameters."""

    if use_fp16:
        named_model_params = [
            (name, state_dict[name]) for name, _ in model.named_parameters()
        ]
        param_groups_and_shapes = get_param_groups_and_shapes(named_model_params)
        master_params = make_master_params(param_groups_and_shapes)
    else:
        master_params = [state_dict[name] for name, _ in model.named_parameters()]
    return master_params


def zero_master_grads(master_params: list[nn.Parameter]) -> None:
    """Zero out the gradients of the master parameters."""
    for param in master_params:
        param.grad = None


def zero_grad(model_params: list[nn.Parameter]) -> None:
    """Zero out the gradients of the model parameters."""
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()

def param_grad_or_zeros(param: nn.Parameter) -> th.Tensor:
    """Return the gradient of the parameter, or a tensor of zeros if it is None."""
    if param.grad is not None:
        return param.grad.data.detach()
    return th.zeros_like(param)


class MixedPrecisionTrainer:
    """Mixed-precision trainer for training models with fp16 precision."""

    def __init__(
        self,
        *,
        model: nn.Module,
        layers: list | None = None,
        use_fp16: bool = False,
        fp16_scale_growth: float = 1e-3,
        initial_lg_loss_scale: float = 20.0,
    ) -> None:
        """Create a mixed-precision trainer."""

        self.model = model
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.loggs = {}

        self.model_params = list(self.model.parameters())
        if layers is None:
            self.master_params = self.model_params
            logger.info("Start training on All Layers")
            params = None
        else:
            params = []
            for i, x in enumerate(self.model.named_parameters()):
                permission = any(layer in x[0] for layer in layers[0]) and all(layer not in x[0] for layer in layers[1])
                if permission:
                    params.append(i)
            self.master_params = [self.model_params[i] for i in params]
            logger.info(f"Start Fine-tuning on {len(params)} Layers")

        self.param_groups_and_shapes = None
        self.lg_loss_scale = initial_lg_loss_scale

        if self.use_fp16:
            self.param_groups_and_shapes = get_param_groups_and_shapes(
                self.model.named_parameters(), params
            )
            self.master_params = make_master_params(self.param_groups_and_shapes)
            self.model.convert_to_fp16()

    def zero_grad(self) -> None:
        """Zero out the gradients of the model parameters."""
        zero_grad(self.model_params)

    def backward(self, loss: th.Tensor) -> None:
        """Backpropagate the given loss."""
        if self.use_fp16:
            loss_scale = 2 ** self.lg_loss_scale
            (loss * loss_scale).backward()
        else:
            loss.backward()

    def optimize(self, opt: th.optim.Optimizer) -> bool:
        """Take an optimization step with the given optimizer."""
        if self.use_fp16:
            return self._optimize_fp16(opt)
        return self._optimize_normal(opt)

    def _optimize_fp16(self, opt: th.optim.Optimizer) -> bool:
        """Take an optimization step with fp16 precision."""
        self.loggs["lg_loss_scale"] = self.lg_loss_scale

        model_grads_to_master_grads(self.param_groups_and_shapes, self.master_params)
        grad_norm, param_norm = self._compute_norms(grad_scale=2 ** self.lg_loss_scale)
        if check_overflow(grad_norm):
            self.lg_loss_scale -= 1
            logger.info(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            zero_master_grads(self.master_params)
            return False

        self.loggs["grad_norm"] = grad_norm
        self.loggs["param_norm"] = param_norm

        for p in self.master_params:
            p.grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        opt.step()
        zero_master_grads(self.master_params)
        master_params_to_model_params(self.param_groups_and_shapes, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth
        return True

    def _optimize_normal(self, opt: th.optim.Optimizer) -> bool:
        """Take an optimization step with normal precision."""
        grad_norm, param_norm = self._compute_norms()
        self.loggs["grad_norm"] = grad_norm
        self.loggs["param_norm"] = param_norm
        opt.step()
        return True

    def _compute_norms(self, grad_scale: float = 1.0) -> tuple[float, float]:
        """Compute the gradient and parameter norms."""

        grad_norm = 0.0
        param_norm = 0.0
        for p in self.master_params:
            with th.no_grad():
                param_norm += th.norm(p, p=2, dtype=th.float32).item() ** 2
                if p.grad is not None:
                    grad_norm += th.norm(p.grad, p=2, dtype=th.float32).item() ** 2
        return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)

    def master_params_to_state_dict(self, master_params: list[nn.Parameter]) -> dict:
        """Convert master parameters to a state dictionary."""

        return master_params_to_state_dict(
            self.model, self.param_groups_and_shapes, master_params, self.use_fp16
        )

    def state_dict_to_master_params(self, state_dict: dict) -> list[nn.Parameter]:
        """Convert a state dictionary to master parameters."""
        return state_dict_to_master_params(self.model, state_dict, self.use_fp16)


def check_overflow(value: float) -> bool:
    """Check if a value is infinite or NaN."""
    return (value == float("inf")) or (value == -float("inf")) or np.isnan(value)
