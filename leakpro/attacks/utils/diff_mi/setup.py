"""Configuration setup for Diff-MI attack."""

import gc
import os
from typing import Optional, Union

import joblib
import numpy as np
import torch
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from leakpro.utils.logger import logger
from leakpro.utils.save_load import hash_model


def clear_cuda_cache(device: Union[torch.device, str, None] = None) -> None:
    """Release cached CUDA blocks after temporary GPU-heavy Diff-MI stages."""
    if device is not None and torch.device(device).type != "cuda":
        return
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


class PreTrainConfig(BaseModel):
    """Configuration for diffusion pretraining."""

    dataset: str = Field(default="celebA", description="Path to the dataset")
    data_dir: str = Field(default="./data", description="Directory containing the data")
    log_dir: str = Field(default="./logging", description="path to save logs and checkpoints")
    schedule_sampler: str = Field(default="uniform", description="Type of schedule sampler")
    lr: float = Field(default=0.0001, description="Learning rate")
    weight_decay: float = Field(default=0.0, description="Weight decay")
    lr_anneal_steps: int = Field(default=0, description="Learning rate annealing steps")
    batch_size: int = Field(default=64, description="Batch size")
    microbatch: int = Field(default=-1, description="Microbatch size")
    ema_rate: str = Field(default="0.9999", description="Exponential moving average rate")
    log_interval: int = Field(default=10, description="Log interval")
    save_interval: int = Field(default=10000, description="Save interval")
    max_steps: int = Field(50000, description="Maximum number of training steps, 50k used from Diff-Mi paper")
    save_name: Optional[str] = Field(None, description="Filename for saving the fine-tuned model")
    resume_checkpoint: str = Field(default="", description="Path to resume checkpoint")
    use_fp16: bool = Field(default=False, description="Use mixed precision training")
    fp16_scale_growth: float = Field(default=0.001, description="FP16 scale growth")
    image_size: int = Field(default=64, description="Image size")
    num_classes: Optional[int] = Field(1001, description="Number of classes including unconditional class")
    num_channels: int = Field(default=128, description="Number of channels")
    num_res_blocks: int = Field(default=3, description="Number of residual blocks")
    num_heads: int = Field(default=4, description="Number of attention heads")
    num_heads_upsample: int = Field(default=-1, description="Number of upsample attention heads")
    num_head_channels: int = Field(default=-1, description="Number of head channels")
    attention_resolutions: str = Field(default="16,8", description="Attention resolutions")
    channel_mult: str = Field(default="", description="Channel multiplier")
    dropout: float = Field(default=0.0, description="Dropout rate")
    class_cond: bool = Field(default=True, description="Class conditional")
    use_checkpoint: bool = Field(default=False, description="Use checkpoint")
    use_scale_shift_norm: bool = Field(default=True, description="Use scale shift norm")
    resblock_updown: bool = Field(default=False, description="Use resblock updown")
    use_new_attention_order: bool = Field(default=False, description="Use new attention order")
    learn_sigma: bool = Field(default=False, description="Learn sigma")
    diffusion_steps: int = Field(default=1000, description="Number of diffusion steps")
    noise_schedule: str = Field(default="linear", description="Noise schedule")
    timestep_respacing: str = Field(default="", description="Timestep respacing")
    use_kl: bool = Field(default=False, description="Use KL divergence")
    predict_xstart: bool = Field(default=False, description="Predict xstart")
    rescale_timesteps: bool = Field(default=False, description="Rescale timesteps")
    rescale_learned_sigmas: bool = Field(default=False, description="Rescale learned sigmas")

class FineTuneConfig(PreTrainConfig):
    """Configuration for Fine-Tuning."""

    class_cond: bool = Field(default=True, description="Class conditional")
    lr: float = Field(default=0.0002, description="Learning rate")
    batch_size: int = Field(default=4, description="Batch size")
    weight_decay: float = Field(default=0.0, description="Weight decay")
    use_fp16: bool = Field(default=False, description="Use mixed precision training")
    fp16_scale_growth: float = Field(default=0.001, description="FP16 scale growth")
    save_name: Optional[str] = Field(default=None, description="Filename for saving the fine-tuned model")
    resume_checkpoint: str = Field(default="", description="Path to resume checkpoint")
    threshold: float = Field(default=0.99, description="Threshold for fine-tuning")
    epochs: int = Field(default=100, description="Number of epochs for fine-tuning")
    label_num: int = Field(default=300, description="Number of labels for fine-tuning")

class DiffusionConfig(BaseModel):
    """Configuration for Diffusion."""

    module_path: Optional[str] = Field(None, description="Path to the model script.")
    model_class: Optional[str] = Field(None, description="Class name of the model.")
    diffusion_class: Optional[str] = Field(None, description="Class name of the model.")

class PreProcessingConfig(BaseModel):
    """Configuration for Preprocessing."""

    top_n: int = Field(default=30, description="Top N to select for pseudo labels")
    top_k: int = Field(default=50, description="Top K features for p_reg computation")

class PGDConfig(BaseModel):
    """Configuration for PGD."""

    constraint: str = Field("2", description="PGD constraint")
    eps: float = Field(0.5, description="PGD epsilon")
    step_size: float = Field(0.1, description="PGD step size")
    iterations: int = Field(10, description="PGD iterations")
    random_start: bool = Field(True, description="PGD random start")
    targeted: bool = Field(True, description="PGD targeted")
    use_best: bool = Field(True, description="PGD use best")
    with_image: bool = Field(True, description="PGD with image")
    make_adv: bool = Field(True, description="PGD make adv")

class AttackConfig(BaseModel):
    """Configuration for DiffMi attack phase."""

    pgdconfig: Optional[PGDConfig] = Field(default_factory=PGDConfig)
    steps: int = Field(30, description="Number of sampling steps.")
    w: float = Field(3.0, description="Guidance scale.")
    ddim_step: int = Field(100, description="DDIM steps.")
    aug_times: int = Field(4, description="Number of augmentations per label.")
    k: int = Field(20, description="Top K for p_reg.")
    alpha: float = Field(1.0, description="Weight for p_reg loss.")


class DiffMiConfig(BaseModel):
    """Overall configuration for DiffMi attack."""

    preprocessing: PreProcessingConfig
    pretrain: PreTrainConfig
    finetune: FineTuneConfig
    diffmiattack: Optional[AttackConfig] = Field(default_factory=AttackConfig)
    diffusion: Optional[DiffusionConfig] = Field(default_factory=DiffusionConfig)
    hash_identifiable: Optional[bool] = Field(False, description="Whether the model is hash identifiable.")
    do_fine_tune: Optional[bool] = Field(False, description="Whether to fine-tune the diffusion model.")
    save_path: Optional[str] = Field(None, description="Path to save checkpoints")

def top_k_p_reg(
    model_res: torch.Tensor,
    all_fea: torch.Tensor,
    n_classes: int = 1000,
    top_n: int = 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the mean and standard deviation of the top-k features for each class.

    Args:
    ----
        model_res (torch.Tensor): The output logits from the model.
        all_fea (torch.Tensor): The features extracted from the model.
        n_classes (int): Number of classes in the model.
        top_n (int): Number of top features to consider for each class.

    Returns:
    -------
        mean_feats (torch.Tensor): Mean of the top-k features for each class.
        std_feats (torch.Tensor): Standard deviation of the top-k features for each class.

    """
    if model_res.shape[0] == 0:
        raise ValueError("Cannot compute p_reg from an empty feature set.")
    if n_classes > model_res.shape[1]:
        logger.warning(
            f"Requested n_classes={n_classes} exceeds model output dim={model_res.shape[1]}; clipping."
        )
        n_classes = model_res.shape[1]
    top_n = min(top_n, int(model_res.shape[0]))
    if top_n < 1:
        raise ValueError("top_n must be at least 1 when computing p_reg.")
    for class_idx in range(n_classes):
        # Get the N highest values and their indices for this class
        _, top_indices = torch.topk(model_res[:, class_idx], k=top_n, dim=0)

        if class_idx == 0:
                mean_feats = torch.mean(all_fea[top_indices], dim=0)
                std_feats = torch.std(all_fea[top_indices], dim=0, unbiased=False)
        else:
                mean_feats = torch.vstack((mean_feats, torch.mean(all_fea[top_indices], dim=0)))
                std_feats = torch.vstack((std_feats, torch.std(all_fea[top_indices], dim=0, unbiased=False)))
    return mean_feats, std_feats

def reparameterize(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Reparameterization trick to sample from N(mu, var) from N(0,1).

    Args:
    ----
        mu (Tensor): Mean of the latent Gaussian [B x D].
        var (Tensor): Variance of the latent Gaussian [B x D].

    """
    std = torch.sqrt(torch.clamp(var, min=1e-12))
    eps = torch.randn_like(std)

    return eps * std + mu

def get_p_reg(
    dataloader: DataLoader,
    model: torch.nn.Module,
    device: Union[torch.device, str],
    args: object,
) -> torch.Tensor:
    """Compute p_reg from the target model features.

    Args:
    ----
        dataloader (DataLoader): DataLoader for the dataset.
        model (torch.nn.Module): Target model.
        device (torch.device): Device to perform computation on.
        args: Configuration arguments.

    Returns:
    -------
        p_reg (torch.Tensor): The computed p_reg tensor.

    """
    logger.info("Computing p_reg from target model features")

    # Use first 8 characters of the hash of the target model for unique identification for p_reg
    hash = hash_model(model)[:8]

    # Check if p_reg already exists
    if os.path.exists(os.path.join(args.save_path, f"p_reg_{hash}.pt")):
        p_reg = torch.load(os.path.join(args.save_path, f"p_reg_{hash}.pt"), map_location="cpu")
        logger.info(f"Loaded p_reg from {os.path.join(args.save_path, f'p_reg_{hash}.pt')}")
        return p_reg.to(device=device)

    # If not, compute p_reg
    logger.info(f"p_reg file not found at {os.path.join(args.save_path, f'p_reg_{hash}.pt')}, computing p_reg.")
    model.to(device)
    model.eval()
    cpu_model_res = []
    cpu_all_fea = []
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device, non_blocking=True)
            fea, res = model(data)
            cpu_model_res.append(res.detach().cpu())
            cpu_all_fea.append(fea.detach().cpu())
            del data, fea, res

    if not cpu_model_res or not cpu_all_fea:
        raise ValueError("Cannot compute p_reg from an empty dataloader.")

    model_res = torch.cat(cpu_model_res, dim=0)
    all_fea = torch.cat(cpu_all_fea, dim=0)
    mu, var = top_k_p_reg(model_res, all_fea, n_classes=model.num_classes, top_n=args.preprocessing.top_k)
    p_reg = reparameterize(mu, var)
    torch.save(p_reg, os.path.join(args.save_path, f"p_reg_{hash}.pt"))
    logger.info(f"Saved p_reg to {os.path.join(args.save_path, f'p_reg_{hash}.pt')}")
    del cpu_model_res, cpu_all_fea, model_res, all_fea, mu, var
    clear_cuda_cache(device)
    return p_reg.to(device=device)

class PseudoDataset(Dataset):
    """Dataset for pseudo-labeled data."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Dataset for Pseudo labeled data.

        Args:
            x (torch.Tensor): Tensor of input images.
            y (torch.Tensor): Tensor of labels.

        """
        self.x = x
        self.y = y

        # Flag to return condition dict for pretraining.
        self.return_cond = False

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, object]:
        """Retrieve the image and its corresponding label at index 'idx'."""
        image = self.x[idx]
        label = self.y[idx]

        cond = {}
        # Return condition dict if specified. Diff-Mi pretraining requires this.
        if self.return_cond:
            cond["y"] = np.array(self.y[idx], dtype=np.int64)
            return image, cond

        return image, label

def _load_pseudo_data(path: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if os.path.exists(path):
        pseudo_data, pseudo_labels = joblib.load(path)
        logger.info(f"Load data from {path}")
        return pseudo_data, pseudo_labels
    return None


def _compute_pseudo_data(
    dataloader: DataLoader,
    target_model: torch.nn.Module,
    device: Union[torch.device, str],
    num_classes: int,
    top_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    target_model.to(device)
    cpu_probs: list[torch.Tensor] = []
    cpu_images: list[torch.Tensor] = []

    with torch.no_grad():
        for images, _ in tqdm(dataloader, total=len(dataloader)):
            images = images.to(device=device, non_blocking=True)
            _, prob = target_model(images)
            cpu_probs.append(prob.detach().cpu())
            cpu_images.append(images.detach().cpu())
            del images, prob

        if not cpu_probs or not cpu_images:
            raise ValueError("Public dataloader is empty; cannot create pseudo-labeled dataset.")

        cpu_all_probs = torch.cat(cpu_probs, dim=0)
        cpu_all_images = torch.cat(cpu_images, dim=0)
        top_n_select = min(top_n, int(cpu_all_probs.shape[0]))
        data_list: list[torch.Tensor] = []
        label_list: list[np.ndarray] = []
        for class_idx in tqdm(range(num_classes)):
            _, top_n_indices = torch.topk(cpu_all_probs[:, class_idx], k=top_n_select, dim=0)
            data_list.append(cpu_all_images[top_n_indices])
            label_list.append(np.full((top_n_select,), class_idx, dtype=np.int32))
        pseudo_data = np.asarray(torch.cat(data_list, dim=0), dtype=np.float32)
        pseudo_labels = np.concatenate(label_list, axis=0)
    del cpu_probs, cpu_images, cpu_all_probs, cpu_all_images, data_list, label_list
    clear_cuda_cache(device)
    return pseudo_data, pseudo_labels


def top_n_pseudo_label_dataset(
    dataloader: DataLoader,
    target_model: torch.nn.Module,
    device: Union[torch.device, str],
    num_classes: int = 1000,
    top_n: int = 30,
    save_dir: str = "./data/",
) -> DataLoader:
    """Select top-n pseudo labels from the public dataset using the target model.

    Args:
    ----
        dataloader (DataLoader): DataLoader for the public dataset.
        target_model (torch.nn.Module): The target model.
        device (torch.device): Device to perform computation on.
        num_classes (int): Number of classes in the target model.
        top_n (int): Number of top samples to select per class.
        save_dir (str): Directory to save/load the pseudo labeled data.

    Returns:
    -------
        pseudo_dataset (PseudoDataset): Dataset with pseudo labeled data.

    """
    logger.info("Performing top-n selection for pseudo labels")
    target_model.eval()

    # Use first 8 characters of the hash of the target model for unique identification
    model_hash = hash_model(target_model)[:8]

    if hasattr(dataloader, "num_workers") and dataloader.num_workers != 0:
        dataloader = DataLoader(
            dataset=dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=getattr(dataloader, "drop_last", False),
        )

    # Check if pseudo labeled data already exists
    pseudo_path = os.path.join(save_dir, f"pseudo_data_{model_hash}.pkl")
    maybe_data = _load_pseudo_data(pseudo_path)
    if maybe_data is None:
        pseudo_data, pseudo_labels = _compute_pseudo_data(
            dataloader=dataloader,
            target_model=target_model,
            device=device,
            num_classes=num_classes,
            top_n=top_n,
        )
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump((pseudo_data, pseudo_labels), pseudo_path)
        logger.info(f"Save data to {pseudo_path}")
    else:
        pseudo_data, pseudo_labels = maybe_data

    return PseudoDataset(x=np.asarray(pseudo_data), y=np.asarray(pseudo_labels))

def extract_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: Union[torch.device, str],
    save_dir: str = "./data/",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features from the target model.

    Args:
    ----
        model (torch.nn.Module): The target model.
        dataloader (DataLoader): DataLoader for the dataset.
        device (torch.device): Device to perform computation on.
        save_dir (str): Directory to save/load the private feature data.

    Returns:
    -------
        priv_features (np.ndarray): Array with extracted features.
        idents (np.ndarray): Array with corresponding labels.

    """
    model.eval()
    # Use first 8 characters of the hash of the target model for unique identification
    model_hash = hash_model(model)[:8]

    # Check if private feature data already exists
    if os.path.exists(os.path.join(save_dir, f"private_features_{model_hash}.pkl")):
        with open(os.path.join(save_dir, f"private_features_{model_hash}.pkl"), "rb") as file:
            priv_features, idents = joblib.load(file)
            logger.info(f"Load data from {os.path.join(save_dir, f'private_features_{model_hash}.pkl')}")

    # If not, compute the private features
    else:
        model.to(device)

        cpu_feats = []
        cpu_labels = []

        with torch.no_grad():
            for images, labels in tqdm(dataloader, total=len(dataloader)):
                images = images.to(device=device, non_blocking=True)
                feats, _ = model(images)
                cpu_feats.append(feats.detach().cpu())
                cpu_labels.append(labels.detach().cpu())
                del images, feats

            if not cpu_feats or not cpu_labels:
                raise ValueError("Private dataloader is empty; cannot extract features.")

        priv_features = np.asarray(torch.cat(cpu_feats, dim=0), dtype=np.float32)
        idents = np.asarray(torch.cat(cpu_labels, dim=0), dtype=np.int32)
        del cpu_feats, cpu_labels
        clear_cuda_cache(device)
        # Save the private features for future use
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"private_features_{model_hash}.pkl"), "wb") as file:
            joblib.dump((priv_features, idents), file)
            logger.info(f"Save data to {os.path.join(save_dir, f'private_features_{model_hash}.pkl')}")

    return priv_features, idents

def args_to_dict(args: object, keys: list[str]) -> dict[str, object]:
    """Convert selected attributes of an object to a dictionary.

    Args:
    ----
        args: The object containing attributes.
        keys: The list of attribute names to include in the dictionary.

    Returns:
    -------
        dict: A dictionary with the selected attributes and their values.

    """
    return {k: getattr(args, k) for k in keys}
