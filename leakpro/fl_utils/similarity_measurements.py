"""Diverse util functions."""
import torch
from ignite.metrics import SSIM
from torch import Tensor, abs, cuda, mean, no_grad, norm
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio


def cosine_similarity_weights(client_gradient: torch.Tensor, reconstruction_gradient: torch.Tensor,
                                top10norms: bool) -> torch.Tensor:
    """Computes the reconstruction costs between client gradients and the reconstruction gradient.

    This function calculates the pairwise costs between each client gradient and the reconstruction gradient
    using the cosine similarity measure. The costs are accumulated and averaged over all client gradients.

    Returns
    -------
        torch.Tensor: The average reconstruction cost.

    """
    with torch.no_grad():
        if top10norms:
            _, indices = torch.topk(torch.stack([p.norm() for p in reconstruction_gradient], dim=0), 10)
        else:
            indices = torch.arange(len(reconstruction_gradient))
        filtered_trial_gradients = [reconstruction_gradient[i] for i in indices]
        filtered_input_gradients = [client_gradient[i] for i in indices]
    costs = sum((x * y).sum() for x, y in zip(filtered_input_gradients,
                                                filtered_trial_gradients))

    trial_norm = sum(x.pow(2).sum()
                        for x in filtered_trial_gradients).sqrt()
    input_norm = sum(y.pow(2).sum()
                        for y in filtered_input_gradients).sqrt()
    return 1 - (costs / trial_norm / input_norm)


def l2_distance(client_gradient: torch.Tensor, reconstruction_gradient: torch.Tensor) -> torch.Tensor:
    """Computes the reconstruction costs between client gradients and the reconstruction gradient.

    This function calculates the pairwise costs between each client gradient and the reconstruction gradient
    using the l2 norm measure. The costs are accumulated and averaged over all client gradients.

    Returns
    -------
        torch.Tensor: The average reconstruction cost.

    """
    with torch.no_grad():

        costs = sum(torch.norm(p1 - p2, p=2) for p1, p2 in zip(client_gradient, reconstruction_gradient))

    return costs

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

    with no_grad():
        # Zip through both dataloaders
        for (orig_batch, rec_batch) in zip(original_dataloader, recreated_dataloader):
            orig_images = orig_batch[0].to(device)  # Original images
            rec_images = rec_batch[0].to(device)    # Recreated images

            # Update SSIM metric
            ssim_metric.update((rec_images, orig_images))

    # Compute average SSIM
    return ssim_metric.compute()
