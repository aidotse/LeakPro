"""Diverse util functions."""
import torch.nn.functional as f
from ignite.metrics import SSIM
from torch import Tensor, abs, cuda, mean, nn, no_grad, norm
from torch.nn.modules.utils import _pair, _quadruple
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio

from leakpro.utils.import_helper import Self


def total_variation(x: Tensor) -> Tensor:
        """Anisotropic TV."""
        dx = mean(abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        dy = mean(abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        return dx + dy

def l2_norm(x: Tensor) -> Tensor:
    """L2 norm."""
    batch_size = len(x)
    return norm(x.view(batch_size, -1), dim=1).mean()


def dataloaders_psnr(original_dataloader: DataLoader, recreated_dataloader: DataLoader) -> float:
    """Calculate the total PSNR between images from two dataloaders (original and recreated).

    Args:
    ----
        original_dataloader (torch.utils.data.DataLoader): Dataloader containing original images and labels.
        recreated_dataloader (torch.utils.data.DataLoader): Dataloader containing recreated images and labels.
        device (str): Device to perform the computation ('cuda' or 'cpu').

    Returns:
    -------
        avg_psnr (float): Average PSNR value over the dataset.

    """
    device = "cuda" if cuda.is_available() else "cpu"
    total_psnr = 0.0

    with no_grad():
        # Zip through both dataloaders
        for (orig_batch, rec_batch) in zip(original_dataloader, recreated_dataloader):
            orig_images = orig_batch[0].to(device)

            rec_images = rec_batch[0].to(device)    # Recreated images

            # Iterate over the batch to compute PSNR for each image pair
            for i in range(orig_images.size(0)):
                psnr = peak_signal_noise_ratio(orig_images[i], rec_images[i], data_range=1.0)
                total_psnr += psnr

    return total_psnr


def dataloaders_ssim_ignite(original_dataloader: DataLoader, recreated_dataloader: DataLoader) -> float:
    """Calculate the average SSIM between images from two dataloaders (original and recreated).

    Args:
    ----
        original_dataloader (torch.utils.data.DataLoader): Dataloader containing original images.
        recreated_dataloader (torch.utils.data.DataLoader): Dataloader containing recreated images.

    Returns:
    -------
        avg_ssim (float): Average SSIM value over the dataset.

    """
    device = "cuda" if cuda.is_available() else "cpu"
    ssim_metric = SSIM(data_range=1.0, device=device)

    ssim_metric.reset()
    total_images = 0

    with no_grad():
        # Zip through both dataloaders
        for (orig_batch, rec_batch) in zip(original_dataloader, recreated_dataloader):
            orig_images = orig_batch[0].to(device)  # Original images
            rec_images = rec_batch[0].to(device)    # Recreated images

            # Update SSIM metric
            ssim_metric.update((rec_images, orig_images))
            total_images += orig_images.size(0)

    # Compute average SSIM
    return ssim_metric.compute()

class MedianPool2d(nn.Module):
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

    def _padding(self: Self, x: Tensor) -> tuple[int,int,int]:
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

    def forward(self: Self, x: Tensor) -> Tensor:
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
