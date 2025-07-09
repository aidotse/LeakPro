"""Utils used for model functionality in GIAs."""


import torch
import torch.nn.functional as f
from torch.nn.modules.utils import _pair, _quadruple

from leakpro.utils.import_helper import Self


class BNFeatureHook:
    """Implementation of the forward hook to track feature statistics and compute a loss on them.

    Will compute mean and variance, and will use l2 as a loss.
    """

    def __init__(self: Self, module: torch.nn.modules.BatchNorm2d) -> None:
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self: Self, module: torch.nn.modules.BatchNorm2d, input: torch.Tensor, _: torch.Tensor) -> None:
        """Hook to compute deepinversion's feature distribution regularization."""
        nch = input[0].shape[1]
        # Compute the mean of the feature maps across batch, height, and width dimensions
        mean = input[0].mean([0, 2, 3])
        # Compute the variance for each channel
        var = (input[0].permute(1, 0, 2,
                                3).contiguous().view([nch,
                                                      -1]).var(1,
                                                               unbiased=False))

        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)
        self.mean = mean
        self.var = var
        self.r_feature = r_feature

    def close(self: Self) -> None:
        """Remove the hook."""
        self.hook.remove()

class InferredBNFeatureHook:
    """Implementation of the forward hook to track feature statistics and compute a loss on them.

    Will compute mean and variance, and will use l2 as a loss.
    """

    def __init__(self: Self, module: torch.nn.modules.BatchNorm2d, client_batch_mean: torch.tensor, client_batch_var: torch.tensor
                 ) -> None:
        self.hook = module.register_forward_hook(self.hook_fn)
        self.client_batch_mean = client_batch_mean
        self.client_batch_var = client_batch_var

    def hook_fn(self: Self, _module: torch.nn.modules.BatchNorm2d, input: torch.Tensor, _: torch.Tensor) -> None:
        """Hook to compute feature distribution regularization toward client statistics."""
        nch = input[0].shape[1]
        # Compute the mean of the feature maps across batch, height, and width dimensions
        mean = input[0].mean([0, 2, 3])
        # Compute the variance for each channel
        var = (input[0].permute(1, 0, 2,
                                3).contiguous().view([nch,
                                                      -1]).var(1,
                                                               unbiased=False))

        r_feature = torch.norm(self.client_batch_var.data - var, 2) + torch.norm(
            self.client_batch_mean.data - mean, 2)
        self.mean = mean
        self.var = var
        self.r_feature = r_feature

    def close(self: Self) -> None:
        """Remove the hook."""
        self.hook.remove()

class MedianPool2d(torch.nn.Module):
    """Median pool (usable as median filter when stride=1) module.

    Args:
    ----
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean

    """

    def __init__(self: Self, kernel_size: int=3, stride: int=1, padding: int=0, same:bool=True) -> None:
        """Initialize with kernel_size, stride, padding."""
        super().__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self: Self, x: torch.Tensor) -> tuple[int,int,int]:
        """Calculate the padding needed for a tensor based on the specified mode and kernel size.

        Args:
        ----
                x (Tensor): Input tensor with shape (batch_size, channels, height, width).

        Returns:
        -------
                Tuple[int, int, int, int]: The calculated padding as (left, right, top, bottom).

        """
        if self.same:
            ih, iw = x.size()[2:]
            ph = max(self.k[0] - self.stride[0], 0) if ih % self.stride[0] == 0 else max(self.k[0] - ih % self.stride[0], 0)
            pw = max(self.k[1] - self.stride[1], 0) if iw % self.stride[1] == 0 else max(self.k[1] - iw % self.stride[1], 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        """Apply a padded unfolding operation to the input and compute the median value across each kernel's unfolded region.

        Args:
        ----
            x (Tensor): Input tensor with shape (batch_size, channels, height, width).

        Returns:
        -------
            Tensor: A tensor containing the median values for each
              patch, with shape (batch_size, channels, new_height, new_width).

        """
        x = f.pad(x, self._padding(x), mode="reflect")
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        return x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
