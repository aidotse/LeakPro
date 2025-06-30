"""Diverse util functions."""
import torch
from ignite.metrics import SSIM
from torch import Tensor, abs, cuda, mean, no_grad, norm
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio


def l2_distance_linear_only(client_grads: torch.Tensor,
                            recon_grads: torch.Tensor) -> torch.Tensor:
    """Computes the average L2 distance between client and reconstruction gradients,
    but only over the parameters of the overcomplete linear layer.
    Assumes those are the first `linear_param_count` entries in the gradient lists.
    """  # noqa: D205
    # number of params in your OvercompleteLinear (weight + bias = 2)
    linear_param_count = 2

    # pick out just the linear‐layer gradients
    lin_client = client_grads[:linear_param_count]
    lin_recon  = recon_grads[:linear_param_count]

    # compute L2 norm differences
    costs = [torch.norm(c - r, p=2) for c, r in zip(lin_client, lin_recon)]
    return torch.stack(costs).mean()


def l2_distance_weights(client_gradient: torch.Tensor, reconstruction_gradient: torch.Tensor,
                        top10norms: bool) -> torch.Tensor:
    """Computes the reconstruction costs between client gradients and the reconstruction gradient using L2 distance.

    This function calculates the pairwise costs between each client gradient and the reconstruction gradient
    using the L2 distance measure. The costs are accumulated and averaged over all client gradients.

    Returns
    -------
        torch.Tensor: The average reconstruction cost.

    """
    with torch.no_grad():
        if top10norms:
            # Calculate norms for each gradient
            norms = torch.stack([p.norm() for p in reconstruction_gradient], dim=0)
            _, indices = torch.topk(norms, 10)
        else:
            indices = torch.arange(len(reconstruction_gradient))

        # Filtered gradients
        filtered_trial_gradients = [reconstruction_gradient[i] for i in indices]
        filtered_input_gradients = [client_gradient[i] for i in indices]

    # Calculate L2 distance (as tensors, not detached)
    costs = sum(torch.norm(x - y, p=2) for x, y in zip(filtered_input_gradients, filtered_trial_gradients))
    return costs / len(filtered_input_gradients)

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
    cg = torch.cat([g.flatten() for g in filtered_input_gradients])
    rg = torch.cat([g.flatten() for g in filtered_trial_gradients])

    # compute the cosine similarity in double precision
    cos = torch.nn.functional.cosine_similarity(cg.unsqueeze(0).double(),
                              rg.unsqueeze(0).double()).clamp(-1.0, 1.0)
    return (1.0 - cos).float()


def l2_distance(client_gradient: torch.Tensor, reconstruction_gradient: torch.Tensor) -> torch.Tensor:
    """Computes the reconstruction costs between client gradients and the reconstruction gradient.

    This function calculates the pairwise costs between each client gradient and the reconstruction gradient
    using the l2 norm measure. The costs are accumulated and averaged over all client gradients.

    Returns
    -------
        torch.Tensor: The average reconstruction cost.

    """
    #with torch.no_grad():

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
