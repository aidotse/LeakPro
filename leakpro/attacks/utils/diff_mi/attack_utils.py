"""Utility functions for the Diff-MI attack."""

import copy
import math
import os
import statistics
from typing import Optional, Tuple, Union

import kornia.augmentation as k_aug
import lpips
import numpy as np
import torch
import torchvision
import torchvision.transforms as augmentation
from robustness import model_utils
from scipy import linalg
from torch.nn import functional
from torch.utils.data import TensorDataset
from torchvision.models import inception_v3
from tqdm import tqdm

from leakpro.utils.logger import logger

from .diffusion import GaussianDiffusion, InferenceModel
from .losses import p_reg_loss, topk_loss


def _compute_epsilon_pred(
    diff_net: torch.nn.Module,
    xt: torch.Tensor,
    t: torch.Tensor,
    classes: torch.Tensor,
    device: Union[torch.device, str],
) -> tuple[torch.Tensor, torch.Tensor]:
    eps = diff_net(xt.float().to(device=device), t.to(device=device), classes.to(device=device))
    non_eps = diff_net(
        xt.float().to(device=device),
        t.to(device=device),
        torch.ones_like(classes, device=device) * (diff_net.num_classes - 1),
    )
    return eps, non_eps


def _apply_attribute_guidance(
    model: InferenceModel,
    classifier: torch.nn.Module,
    classes: torch.Tensor,
    p_reg: torch.Tensor,
    aug: k_aug.container.ImageSequential,
    args: object,
    opt: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LinearLR,
    norm_track: torch.Tensor,
    device: Union[torch.device, str],
) -> None:
    attr_input_batch = []
    for _ in range(args.aug_times):
        attr_input = model.encode()
        attr_input = aug(attr_input).clamp(-1, 1)
        attr_input_batch.append(attr_input)

    attr_input_batch = torch.cat(attr_input_batch, dim=0).to(device=device)
    feats, logits = classifier.forward((attr_input_batch + 1) / 2)

    loss = topk_loss(logits, classes.repeat(args.aug_times), k=args.k) + (
        args.alpha * p_reg_loss(feats, classes.repeat(args.aug_times), p_reg)
    )

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.25 * norm_track)
    opt.step()
    scheduler.step()


def iterative_reconstruction(
    args: object,
    diff_net: torch.nn.Module,
    classifier: torch.nn.Module,
    classes: torch.Tensor,
    p_reg: torch.Tensor,
    device: Union[torch.device, str] = "cuda",
    diffusion_steps: Optional[int] = 1000,
    data_channels: int = 3,
    data_height: int = 64,
    data_width: int = 64,
) -> torch.Tensor:
    """Perform iterative image reconstruction using diffusion model and classifier guidance."""
    if diffusion_steps is None:
        diffusion_steps = getattr(args, "diffusion_steps", None)
    if diffusion_steps is None:
        diffusion_steps = classifier.num_classes
    diffusion = GaussianDiffusion(num_steps=diffusion_steps, schedule="linear")
    model = InferenceModel(
        batch_size=classes.shape[0],
        data_channels=data_channels,
        data_height=data_height,
        data_width=data_width,
    ).to(device=device)
    model.train()

    steps = args.steps
    opt = torch.optim.Adamax(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=1.0, total_iters=steps)

    norm_track: Union[torch.Tensor, float] = 0

    aug = k_aug.container.ImageSequential(
        k_aug.RandomHorizontalFlip(),
        k_aug.ColorJitter(brightness=0.2, p=0.5),
        k_aug.RandomGaussianBlur((7, 7), (3, 3), p=0.5),
    )
    bar = range(steps)
    for i, _ in enumerate(bar):
        t = ((steps - i) / 1.5 + (steps - i) / 3 * math.cos(3 / (10 * math.pi) * i)) / steps
        t = t * (0.8 * diffusion_steps) + 0.2 * diffusion_steps
        t = np.array([t + np.random.randint(-50, 51) for _ in range(1)]).astype(int)
        t = np.clip(t, 1, diffusion_steps)

        sample_img = model.encode()
        xt, epsilon = diffusion.sample(sample_img, t)
        t = torch.from_numpy(t).float().view(1)
        eps, non_eps = _compute_epsilon_pred(diff_net, xt, t, classes, device)
        epsilon_pred = args.w * eps - (args.w - 1) * non_eps

        loss = 1 * functional.mse_loss(epsilon_pred, epsilon)

        opt.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad_norm = torch.linalg.norm(model.img.grad)
            if i > 0:
                alpha = 0.5
                norm_track = alpha * norm_track + (1 - alpha) * grad_norm
            else:
                norm_track = grad_norm
        opt.step()

        _apply_attribute_guidance(
            model=model,
            classifier=classifier,
            classes=classes,
            p_reg=p_reg,
            aug=aug,
            args=args,
            opt=opt,
            scheduler=scheduler,
            norm_track=norm_track,
            device=device,
        )

    with torch.no_grad():
        ddim_step = min(args.ddim_step, diffusion_steps)
        if ddim_step == diffusion_steps:
            t = np.array([ddim_step]).astype(int)
            xt = model.encode()
        else:
            t = np.array([ddim_step]).astype(int)
            xt, _ = diffusion.sample(model.encode(), t)
        fine_tuned = diffusion.inverse_ddim(diff_net, x=xt, start_t=t[0], w=args.w, y=classes, device=device)

    return (fine_tuned + 1) / 2

def get_pgd(model: torch.nn.Module) -> torch.nn.Module:
    """Set the model to use PGD adversarial training.

    Args:
    ----
        model: Pre-trained classifier model.

    Returns:
    -------
        pgd_model: Model set up for PGD adversarial training.

    """
    logger.info("Making PGD version of model for the Diff-MI attack.")
    class MeanAndStd:
        def __init__(self) -> None:
            self.mean = torch.tensor([0.0, 0.0, 0.0])
            self.std = torch.tensor([1.0, 1.0, 1.0])

    # Create a copy of the model to avoid modifying the original
    model_copy = copy.deepcopy(model)

    # Set to False to return only the output logits
    model_copy.return_feature = False

    pgd_model, _ = model_utils.make_and_restore_model(arch=model_copy, dataset=MeanAndStd())
    pgd_model.eval()

    return pgd_model

def calc_acc(
    classifier: torch.nn.Module,
    data: torch.Tensor,
    labels: torch.Tensor,
    bs: int = 64,
    anno: str = "",
    with_success: bool = False,
    enable_print: bool = False,
) -> Union[Tuple[float, float], Tuple[float, float, torch.Tensor]]:
    """Calculate top-1 and top-5 accuracy of the classifier on the given data and labels.

    Args:
    ----
        classifier: Pre-trained classifier model.
        data: Input data (Tensor).
        labels: True labels (Tensor).
        bs: Batch size for processing data.
        anno: Annotation string for logging.
        batch_size: Batch size for processing data.
        with_success: Whether to return indices of successful predictions.
        enable_print: Whether to print the accuracy results.

    Returns:
    -------
        top1_count: Number of correct top-1 predictions.
        top5_count: Number of correct top-5 predictions.
        success_idx (optional): Indices of successful predictions if with_success is True.

    """
    # Resize data if classifier has a resize method
    if hasattr(classifier, "resize") and callable(getattr(classifier, "resize", None)):
        data = classifier.resize(data)
    output, img_dataset = [], TensorDataset(data)
    for x in torch.utils.data.DataLoader(img_dataset, batch_size=bs, shuffle=False):
        output.append(classifier(x[0])[-1])
    output = torch.cat(output)
    topk_val = min(5, int(output.shape[1]))
    top1_count = torch.eq(torch.topk(output, k=1)[1], labels.view(-1,1)).float()
    top5_count = torch.eq(torch.topk(output, k=topk_val)[1], labels.view(-1,1)).float()
    if enable_print:
        logger.info(
            f"Acculating accuracy: top1_acc - {top1_count.mean().item():.2%}, "
            f"top5_acc: {topk_val * top5_count.mean().item():.2%} {anno}"
        )
    if with_success:
        success_idx = torch.nonzero((output.max(1)[1] == labels).int()).squeeze(1)
        return top1_count.sum().item(), top5_count.sum().item(), success_idx
    return top1_count.sum().item(), top5_count.sum().item()


def calc_acc_std(
    data: torch.Tensor,
    labels: torch.Tensor,
    classifier: torch.nn.Module,
    label_num: int,
) -> tuple[float, float, float, float]:
    """Calculate top-1 and top-5 accuracy and their standard deviations over groups of data.

    Args:
    ----
        data: Input data (Tensor).
        labels: True labels (Tensor).
        classifier: Pre-trained classifier model.
        label_num: Number of labels per group.
        dims: Dimensions to resize data for the classifier.

    Returns:
    -------
        acc1: Mean top-1 accuracy.
        acc5: Mean top-5 accuracy.
        var1: Standard deviation of top-1 accuracy.
        var5: Standard deviation of top-5 accuracy.

    """
    # Resize data if classifier has a resize method
    if hasattr(classifier, "resize") and callable(getattr(classifier, "resize", None)):
        data = classifier.resize(data)

    top1_list, top5_list = [], []
    assert data.shape[0] % label_num == 0
    for i in range(int(data.shape[0]/label_num)):
        data_ = data[i * label_num: (i+1) * label_num]
        labels_ = labels[i * label_num: (i+1) * label_num]
        assert torch.max(labels_) - torch.min(labels_) == label_num - 1
        top1_count, top5_count = calc_acc(classifier, data_, labels_, enable_print=False)
        top1_list.append(top1_count/label_num)
        top5_list.append(top5_count/label_num)
    try:
        acc1 = statistics.mean(top1_list)
        acc5 = statistics.mean(top5_list)
    except Exception:
        acc1, acc5 = top1_list[0], top5_list[0]

    try:
        var1 = statistics.stdev(top1_list)
        var5 = statistics.stdev(top5_list)
    except Exception:
        var1, var5 = 0.0, 0.0

    return acc1, acc5, var1, var5

def save_tensor(
    data: torch.Tensor,
    labels: torch.Tensor,
    save_path: str,
    file_extension: str = ".pt",
) -> list[str]:
    """Save a batch of tensors as individual files organized by labels.

    Args:
    ----
        data: Input data (Tensor).
        labels: Corresponding labels (Tensor).
        save_path: Directory to save the files.
        file_extension: File extension for saved files (default: ".pt").

    Returns:
    -------
        label_paths: List of paths to the saved files.

    """
    label_paths = []
    dataset = TensorDataset(data.cpu(), labels.cpu())
    for i, (x, y) in enumerate(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)):
        label_path = os.path.join(save_path, str(y.item()))
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        label_paths.append(label_path)

        # Save based on file extension
        if file_extension in [".png", ".jpg", ".jpeg"]:
            # Handle image data
            torchvision.utils.save_image(x.detach()[0, :, :, :],
                                        os.path.join(label_path, f"{i}_attack{file_extension}"),
                                        padding=0)
        else:
            # Handle general tensor data
            torch.save(x.detach()[0], os.path.join(label_path, f"{i}_attack{file_extension}"))

    return label_paths


def calc_lpips(
    private_data: torch.utils.data.DataLoader,
    fakes: torch.Tensor,
    fake_targets: torch.Tensor,
    anno: str = "",
    device: Union[torch.device, str] = "cuda",
) -> tuple[float, float]:
    """Calculate LPIPS distance between reconstructed data and target data.

    Args:
    ----
        private_data: DataLoader containing private/real data.
        fakes: Reconstructed data (Tensor).
        fake_targets: Corresponding target labels (Tensor).
        anno: Annotation string for logging.
        device: Device to run the computations on.

    Returns:
    -------
        value_a: Average LPIPS distance using AlexNet.
        value_v: Average LPIPS distance using VGG.

    """
    try:
        _ = anno
        loss_fn_alex = lpips.LPIPS(net="alex").to(device)  # best forward scores
        loss_fn_vgg = lpips.LPIPS(net="vgg").to(device)  # closer to "traditional" perceptual loss, when used for optimization

        # Load real data and labels from private_data dataloader
        real_data, real_targets = None, None
        for x in private_data:
            real_data = x[0] if real_data is None else torch.cat([real_data, x[0]], dim=0)
            real_targets = x[1] if real_targets is None else torch.cat([real_targets, x[1]], dim=0)

        real_data = real_data.to(device)
        real_targets = real_targets.to(device)

        # Get fake data and labels
        fakes = fakes.to(device)
        fake_targets = fake_targets.to(device)
        bs = fake_targets.size(0)
        value_a, value_v = 0, 0

        # Calculate metric
        for i in range(bs):
            single_value_a, single_value_v = -1, -1
            idx = torch.nonzero(real_targets == fake_targets[i]).squeeze(1)
            for j in idx:
                temp_value_a = loss_fn_alex(fakes[i].unsqueeze(0), real_data[j].unsqueeze(0))
                temp_value_v = loss_fn_vgg(fakes[i].unsqueeze(0), real_data[j].unsqueeze(0))
                single_value_a = max(single_value_a, temp_value_a)
                single_value_v = max(single_value_v, temp_value_v)
            value_a += single_value_a
            value_v += single_value_v
        value_a = (value_a / bs).item()
        value_v = (value_v / bs).item()

        return value_a, value_v
    except Exception as e:
        logger.error(f"LPIPS calculation failed: {e}")
        return -1.0, -1.0

def calc_knn(
    fake_data: torch.Tensor,
    fake_targets: torch.Tensor,
    private_feats: np.ndarray,
    private_idents: np.ndarray,
    evaluation_model: torch.nn.Module,
    batch_size: int = 64,
    anno: str = "",
    device: Union[torch.device, str] = "cuda",
    dims: tuple[int, int] = (64, 64),
) -> tuple[float, np.ndarray]:
    """Calculate KNN distance between reconstructed data and target data in feature space.

    Args:
    ----
        fake_data: Reconstructed data (Tensor).
        fake_targets: Corresponding target labels (Tensor).
        private_feats: Fetures from private dataset (np.ndarray).
        private_idents: Labels corresponding to the private features (np.ndarray).
        evaluation_model: Pre-trained feature extractor model.
        batch_size: Batch size for processing data.
        anno: Annotation string for logging.
        device: Device to run the computations on.
        dims: Dimensions to resize data for the feature extractor.

    Returns:
    -------
        knn: Average Minimum KNN distance value.
        knn_arr: Array of Minimum KNN distances for each reconstructed image.

    """

    _ = anno
    # get features of reconstructed data
    inferred_feats = None
    for i, data in enumerate(torch.utils.data.DataLoader(fake_data, batch_size=batch_size)):
        data = augmentation.Resize(dims)(data).to(device)
        feats = evaluation_model(data)[0]
        inferred_feats = feats.detach().cpu() if i == 0 else torch.cat([inferred_feats, feats.detach().cpu()], dim=0)

    # get features of target data
    idens = fake_targets.to(device).long()
    feats = inferred_feats.to(device)
    private_feats = torch.from_numpy(private_feats).float().to(device)
    private_idents = torch.from_numpy(private_idents).view(-1).long().to(device)
    bs = feats.size(0)
    if bs == 0:
        return float("nan"), np.array([])
    knn_dist = 0.0

    # calculate knn dist
    knn_arr = np.full((bs,), np.nan, dtype=np.float32)
    for i in tqdm(range(bs), desc="Calculating KNN Dist"):
        idx = torch.nonzero(private_idents == idens[i]).squeeze(1)
        if idx.numel() == 0:
            continue
        fake_feat = feats[i].repeat(idx.shape[0], 1)
        true_feat = private_feats[idx]
        knn = torch.sum(torch.pow(fake_feat - true_feat, 2), dim=1)
        knn_min = torch.min(knn)
        knn_arr[i] = knn_min.item()
        knn_dist += knn_min.item()

    valid_count = np.sum(~np.isnan(knn_arr))
    if valid_count == 0:
        return float("nan"), knn_arr
    knn = float(knn_dist / valid_count)

    return knn, knn_arr

def calc_mse(
    private_data: torch.utils.data.DataLoader,
    fakes: torch.Tensor,
    fake_labels: torch.Tensor,
    device: Union[torch.device, str] = "cuda",
    anno: str = "",
) -> tuple[float, dict[int, float], list[float], list[torch.Tensor], list[torch.Tensor]]:
    """Calculate mean squared error between real and fake data grouped by labels.

    Args:
    ----
        private_data: DataLoader containing private/real data.
        labels: True labels for real data (Tensor).
        fakes: Reconstructed/fake data (Tensor).
        fake_labels: Labels for fake data (Tensor).
        device: Device to run the computations on.
        anno: Annotation string for logging.

    Returns:
    -------
        avg_mse: Average MSE across all label groups.
        mse_per_label: Dictionary mapping each label to its MSE value.

    """
    _ = anno
    real, labels = None, None
    for x in private_data:
        real = x[0] if real is None else torch.cat([real, x[0]], dim=0)
        labels = x[1] if labels is None else torch.cat([labels, x[1]], dim=0)

    unique_labels = torch.unique(labels)
    mse_values = []
    mse_values_arr = []
    mse_min_real = []
    mse_min_fake = []
    mse_per_label = {}

    for label in unique_labels:
        # Get all real data with this label
        real_idx = torch.nonzero(labels == label).squeeze(1)
        real_data = real[real_idx]

        # Get all fake data with this label
        fake_idx = torch.nonzero(fake_labels == label).squeeze(1)
        fake_data = fakes[fake_idx]

        if len(real_data) == 0 or len(fake_data) == 0:
            continue

        # Calculate MSE
        label_mse = []
        for real_data_point in real_data:
            min_label_mse = 0
            for fake_data_point in fake_data:
                mse = functional.mse_loss(real_data_point.to(device=device), fake_data_point.to(device=device))
                label_mse.append(mse.item())

                # Use the minimum MSE fake image for this real image
                if min_label_mse == 0 or mse.item() < min_label_mse:
                    min_label_mse = mse.item()
                    min_real_data_point = real_data_point
                    min_fake_data_point = fake_data_point
                min_label_mse = mse.item() if min_label_mse == 0 else min(min_label_mse, mse.item())

            # Store all MSE values for this label
            mse_values_arr.append(min_label_mse)

            # Store the real and fake data point with the lowest mse
            mse_min_real.append(min_real_data_point)
            mse_min_fake.append(min_fake_data_point)

        # Average MSE
        avg_label_mse = statistics.mean(label_mse)
        mse_per_label[label.item()] = avg_label_mse
        mse_values.append(avg_label_mse)

    avg_mse = statistics.mean(mse_values) if mse_values else 0.0

    return avg_mse, mse_per_label, mse_values_arr, mse_min_fake, mse_min_real

def calc_pytorch_fid(
    fake_data: torch.Tensor,
    private_data: torch.utils.data.DataLoader,
    batch_size: int = 50,
    device: Union[torch.device, str] = "cuda",
    dims: int = 2048,
    num_workers: int = 1,
    anno: str = "",
) -> float:
    """Calculate FID score between fake data tensors and private dataset.

    Args:
    ----
        fake_data: Reconstructed/fake data (Tensor).
        private_data: DataLoader containing private/real data.
        batch_size: Batch size for processing.
        device: Device to run calculations.
        dims: Dimensionality of Inception features.
        num_workers: Number of parallel dataloader workers.
        anno: Annotation string for logging.

    Returns:
    -------
        fid_value: Computed FID score.

    """

    _ = num_workers
    _ = anno
    # Load InceptionV3 model
    try:
        model = inception_v3(pretrained=True, transform_input=False).to(device)
        model.fc = torch.nn.Identity()  # Remove final classification layer
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load Inception model: {e}")
        return -1.0

    # Get activations for fake data
    m1, s1 = compute_statistics_from_tensor(fake_data, model, batch_size, dims, device)

    # Get activations for private data
    m2, s2 = compute_statistics_from_dataloader(private_data, model, batch_size, dims, device)

    # Calculate FID
    return float(calculate_frechet_distance(m1, s1, m2, s2))


def compute_statistics_from_tensor(
    data: torch.Tensor,
    model: torch.nn.Module,
    batch_size: int,
    dims: int,
    device: Union[torch.device, str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and covariance of Inception features from tensor data.

    Args:
    ----
        data: Input data tensor.
        model: Inception model.
        batch_size: Batch size for processing.
        dims: Dimensionality of features.
        device: Device to run calculations.

    Returns:
    -------
        mu: Mean of activations.
        sigma: Covariance of activations.

    """
    act = get_activations_from_tensor(data, model, batch_size, dims, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_from_dataloader(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    batch_size: int,
    dims: int,
    device: Union[torch.device, str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and covariance of Inception features from dataloader.

    Args:
    ----
        dataloader: DataLoader containing data.
        model: Inception model.
        batch_size: Batch size for processing.
        dims: Dimensionality of features.
        device: Device to run calculations.

    Returns:
    -------
        mu: Mean of activations.
        sigma: Covariance of activations.

    """
    act = get_activations_from_dataloader(dataloader, model, batch_size, dims, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def get_activations_from_tensor(
    data: torch.Tensor,
    model: torch.nn.Module,
    batch_size: int,
    dims: int,
    device: Union[torch.device, str],
) -> np.ndarray:
    """Extract Inception activations from tensor data.

    Args:
    ----
        data: Input data tensor.
        model: Inception model.
        batch_size: Batch size for processing.
        dims: Dimensionality of features.
        device: Device to run calculations.

    Returns:
    -------
        pred_arr: Array of activations.

    """
    model.eval()

    dataset = TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    pred_arr = np.empty((len(data), dims))
    start_idx = 0

    for batch in tqdm(dataloader, desc="Computing activations (fake)"):
        batch = batch[0].to(device)

        # Resize to 299x299 for Inception
        if batch.shape[2] != 299 or batch.shape[3] != 299:
            batch = functional.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)

        with torch.no_grad():
            pred = model(batch)

        # Apply global average pooling if needed
        if len(pred.shape) > 2:
            pred = functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.squeeze(3).squeeze(2)

        pred = pred.cpu().numpy()
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]

    return pred_arr


def get_activations_from_dataloader(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    batch_size: int,
    dims: int,
    device: Union[torch.device, str],
) -> np.ndarray:
    """Extract Inception activations from dataloader.

    Args:
    ----
        dataloader: DataLoader containing data.
        model: Inception model.
        batch_size: Batch size for processing.
        dims: Dimensionality of features.
        device: Device to run calculations.

    Returns:
    -------
        pred_arr: Array of activations.

    """
    _ = batch_size
    model.eval()

    # First pass to count samples
    num_samples = sum(batch[0].shape[0] for batch in dataloader)

    pred_arr = np.empty((num_samples, dims))
    start_idx = 0

    for batch in tqdm(dataloader, desc="Computing activations (private)"):
        batch_data = batch[0].to(device)

        # Resize to 299x299 for Inception
        if batch_data.shape[2] != 299 or batch_data.shape[3] != 299:
            batch_data = functional.interpolate(batch_data, size=(299, 299), mode="bilinear", align_corners=False)

        with torch.no_grad():
            pred = model(batch_data)

        # Apply global average pooling if needed
        if len(pred.shape) > 2:
            pred = functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.squeeze(3).squeeze(2)

        pred = pred.cpu().numpy()
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Calculate Frechet distance between two Gaussian distributions.

    Args:
    ----
        mu1: Mean of first distribution.
        sigma1: Covariance of first distribution.
        mu2: Mean of second distribution.
        sigma2: Covariance of second distribution.
        eps: Epsilon for numerical stability.

    Returns:
    -------
        fid: Frechet distance.

    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        logger.warning(f"FID calculation produces singular product; adding {eps} to diagonal of cov estimates")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
