"""Various utilities for neural networks."""

import math
from collections.abc import Callable, Iterable, Sequence
from typing import Union

import torch as th
from torch import nn
from torch.autograd.function import FunctionCtx


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    """SiLU activation for older PyTorch versions."""

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Apply the SiLU activation."""
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    """Group normalization that preserves input dtype."""

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Normalize with float32 and cast back to input dtype."""
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims: int, *args: object, **kwargs: object) -> nn.Module:
    """Create a 1D, 2D, or 3D convolution module."""
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    if dims == 2:
        return nn.Conv2d(*args, **kwargs)
    if dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args: object, **kwargs: object) -> nn.Module:
    """Create a linear module."""
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims: int, *args: object, **kwargs: object) -> nn.Module:
    """Create a 1D, 2D, or 3D average pooling module."""
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    if dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    if dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(
    target_params: Iterable[th.Tensor],
    source_params: Iterable[th.Tensor],
    rate: float = 0.99,
) -> None:
    """Update target parameters using an exponential moving average.

    Args:
    ----
        target_params: Target parameter sequence.
        source_params: Source parameter sequence.
        rate: EMA rate (closer to 1 means slower).

    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module: nn.Module) -> nn.Module:
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module: nn.Module, scale: float) -> nn.Module:
    """Scale the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor: th.Tensor) -> th.Tensor:
    """Take the mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels: int) -> nn.Module:
    """Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps: th.Tensor, dim: int, max_period: int = 10000) -> th.Tensor:
    """Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(
    func: Callable[..., Union[th.Tensor, tuple[th.Tensor, ...]]],
    inputs: Sequence[th.Tensor],
    params: Sequence[th.Tensor],
    flag: bool,
) -> Union[th.Tensor, tuple[th.Tensor, ...]]:
    """Evaluate a function without caching intermediate activations.

    Args:
    ----
        func: Function to evaluate.
        inputs: Argument sequence to pass to `func`.
        params: Parameters `func` depends on but does not take as arguments.
        flag: If False, disable gradient checkpointing.

    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    """Custom autograd function for gradient checkpointing."""

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        run_function: Callable[..., Union[th.Tensor, tuple[th.Tensor, ...]]],
        length: int,
        *args: th.Tensor,
    ) -> Union[th.Tensor, tuple[th.Tensor, ...]]:
        """Run the function with inputs, storing context for backward."""
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            return ctx.run_function(*ctx.input_tensors)

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        *output_grads: th.Tensor,
    ) -> tuple[th.Tensor | None, ...]:
        """Recompute forward pass and return gradients."""
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
