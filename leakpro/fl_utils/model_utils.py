"""Utils used for model functionality in GIAs."""


import torch

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
