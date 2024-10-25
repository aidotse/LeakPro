"""Diverse util functions."""
from torch import Tensor, abs, cuda, mean, no_grad
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio


def total_variation(x: Tensor) -> Tensor:
        """Anisotropic TV."""
        dx = mean(abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        dy = mean(abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        return dx + dy


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
