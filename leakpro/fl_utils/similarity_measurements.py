#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Diverse util functions."""
import numpy as np
import torch
from ignite.metrics import SSIM
from torch import Tensor, abs, cuda, mean, no_grad, norm
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


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

    return sum(torch.norm(p1 - p2, p=2) for p1, p2 in zip(client_gradient, reconstruction_gradient))

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
    """Calculate the average max SSIM for each recreated image over all original images.

    Each recreated image is compared against all original images in the current batch,
    taking the maximum SSIM as its score.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_range = 6.0  # adjust based on your normalization
    ssim_metric = SSIM(data_range=data_range, device=device)

    max_ssim_scores = []

    with torch.no_grad():
        for orig_batch, rec_batch in zip(original_dataloader, recreated_dataloader):
            orig_images = orig_batch[0].to(device)
            rec_images = rec_batch[0].to(device)

            for rec_image in rec_images:
                rec_image_expanded = rec_image.unsqueeze(0).repeat(orig_images.size(0), 1, 1, 1)
                ssim_metric.reset()
                ssim_metric.update((rec_image_expanded, orig_images))
                ssim_value = ssim_metric.compute()
                max_ssim_scores.append(ssim_value)
    return sum(max_ssim_scores) / len(max_ssim_scores) if max_ssim_scores else 0.0


_lpips_metrics = {}


def _lpips_metric(net_type: str, device: str) -> LearnedPerceptualImagePatchSimilarity:
    """Build the LPIPS network once per (net, device) and reuse it.

    Constructing it downloads and instantiates a pretrained backbone, which is far too expensive to
    redo on every evaluation inside the attack loop.
    """
    key = (net_type, device)
    if key not in _lpips_metrics:
        _lpips_metrics[key] = LearnedPerceptualImagePatchSimilarity(net_type=net_type, normalize=True).to(device)
    return _lpips_metrics[key]


def dataloaders_lpips(original_dataloader: DataLoader, recreated_dataloader: DataLoader,
                      data_mean: Tensor = None, data_std: Tensor = None, net_type: str = "alex") -> float:
    """Calculate the average min LPIPS distance for each recreated image over all original images.

    LPIPS is a perceptual *distance*: 0 means identical and larger means less alike, so each recreated
    image takes the minimum over the originals in the batch, mirroring the maximum that
    `dataloaders_ssim_ignite` takes. The network expects images in [0, 1], so the loaders'
    normalization is undone first when the mean and std are known.
    """
    device = "cuda" if cuda.is_available() else "cpu"
    metric = _lpips_metric(net_type, device)

    def denormalize(x: Tensor) -> Tensor:
        if data_mean is None or data_std is None:
            return x.clamp(0.0, 1.0)
        return (x * data_std.to(x.device) + data_mean.to(x.device)).clamp(0.0, 1.0)

    min_lpips_scores = []

    with no_grad():
        for orig_batch, rec_batch in zip(original_dataloader, recreated_dataloader):
            orig_images = denormalize(orig_batch[0].to(device))
            rec_images = denormalize(rec_batch[0].to(device))

            for rec_image in rec_images:
                pair_scores = []
                for orig_image in orig_images:
                    # forward() accumulates into the metric's global state, which we never use.
                    metric.reset()
                    pair_scores.append(float(metric(rec_image.unsqueeze(0), orig_image.unsqueeze(0))))
                min_lpips_scores.append(min(pair_scores))
    metric.reset()
    # 1.0 is the "nothing to compare against" distance; the loaders are never empty in practice.
    return sum(min_lpips_scores) / len(min_lpips_scores) if min_lpips_scores else 1.0


def dataloaders_similarity(metric: str, original_dataloader: DataLoader, recreated_dataloader: DataLoader,
                           data_mean: Tensor = None, data_std: Tensor = None) -> float:
    """Score a reconstruction with the named metric, oriented so that higher is always better.

    The attack loop keeps the best scoring reconstruction and optuna maximizes the yielded value, so a
    distance metric has to enter negated.
    """
    if metric == "ssim":
        return dataloaders_ssim_ignite(original_dataloader, recreated_dataloader)
    if metric == "lpips":
        return -dataloaders_lpips(original_dataloader, recreated_dataloader, data_mean, data_std)
    raise ValueError(f"Unknown similarity metric '{metric}', expected 'ssim' or 'lpips'.")


def text_reconstruciton_score(original_dataloader: DataLoader, recreated_dataloader: DataLoader, token_used: Tensor) -> float:
    """Calculate the reconstruction text statistics from two dataloaders (original and recreated).

    Args:
    ----
        original_dataloader (torch.utils.data.DataLoader): Dataloader containing original text.
        recreated_dataloader (torch.utils.data.DataLoader): Dataloader containing recreated text.
        token_used (torch.Tensor): Tensor for token used.

    Returns:
    -------
        avg_sim (float): Average SIM value over the dataset.

    """
    pred_orders = []
    for orig_x, rec_x in zip(original_dataloader, recreated_dataloader):

        true_tokens = orig_x["embedding"][0].cpu().numpy()
        true_labels = orig_x["labels"][0].cpu().numpy()

        predict_tokens = rec_x["embedding"][0].detach().cpu().numpy()
        _ = rec_x["labels"][0].cpu().numpy()
        ind = np.where(np.array(true_labels)!=0)[0]
        predict_tokens = predict_tokens[ind]
        true_tokens = true_tokens[ind]

        for i in range(len(true_tokens)):
            true_token = np.argmax(true_tokens[i,token_used])
            pred_order = np.where(np.argsort(-predict_tokens[i,token_used])==true_token)[0][0]
            pred_orders.append(pred_order)


    return np.mean(pred_orders)
