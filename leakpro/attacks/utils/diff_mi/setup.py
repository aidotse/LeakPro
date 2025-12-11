"""Configuration setup for Diff-MI attack."""

import os
import pickle
from typing import Optional

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from leakpro.utils.logger import logger
from leakpro.utils.save_load import hash_model


class PreTrainConfig(BaseModel):
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

    class_cond: bool = Field(True, description="Class conditional")
    lr: float = Field(0.0002, description="Learning rate")
    batch_size: int = Field(4, description="Batch size")
    weight_decay: float = Field(0.0, description="Weight decay")
    use_fp16: bool = Field(False, description="Use mixed precision training")
    fp16_scale_growth: float = Field(0.001, description="FP16 scale growth")
    save_name: Optional[str] = Field(None, description="Filename for saving the fine-tuned model")
    resume_checkpoint: Optional[str] = Field(None, description="Path to resume checkpoint")
    threshold: float = Field(0.99, description="Threshold for fine-tuning")
    epochs: int = Field(100, description="Number of epochs for fine-tuning")

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
    label_num: int = Field(300, description="First N labels for the attack.")
    repeat_N: int = Field(5, description="Number of times to repeat the attack.")
    steps: int = Field(30, description="Number of sampling steps.")
    w: float = Field(3.0, description="Guidance scale.")
    ddim_step: int = Field(100, description="DDIM steps.")
    batch_size: int = Field(8, description="Batch size for sampling.")
    aug_times: int = Field(4, description="Number of augmentations per label.")
    k: int = Field(20, description="Top K for p_reg.")
    alpha: float = Field(1.0, description="Weight for p_reg loss.")
    calc_knn: bool = Field(True, description="Whether to calculate KNN accuracy.")
    calc_lpips: bool = Field(True, description="Whether to calculate LPIPS.")
    calc_mse: bool = Field(True, description="Whether to calculate MSE.")
    calc_fid: bool = Field(True, description="Whether to calculate FID.")


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

def top_k_p_reg(model_res, all_fea, n_classes=1000, top_N=30):
    """Compute the mean and standard deviation of the top-k features for each class.

    Args:
    ----
        model_res (torch.Tensor): The output logits from the model.
        all_fea (torch.Tensor): The features extracted from the model.
        n_classes (int): Number of classes in the model.
        top_N (int): Number of top features to consider for each class.

    Returns:
    -------
        mean_feats (torch.Tensor): Mean of the top-k features for each class.
        std_feats (torch.Tensor): Standard deviation of the top-k features for each class.

    """
    for class_idx in range(n_classes):
        # Get the N highest values and their indices for this class
        _, top_indices = torch.topk(model_res[:, class_idx], k=top_N, dim=0)

        if class_idx == 0:
                mean_feats = torch.mean(all_fea[top_indices], dim=0)
                std_feats = torch.std(all_fea[top_indices], dim=0)
        else:
                mean_feats = torch.vstack((mean_feats, torch.mean(all_fea[top_indices], dim=0)))
                std_feats = torch.vstack((std_feats, torch.std(all_fea[top_indices], dim=0)))
    return mean_feats, std_feats

def reparameterize(mu, logvar):
    """Reparameterization trick to sample from N(mu, var) from
    N(0,1).

    Args:
    ----
    mu (Tensor) Mean of the latent Gaussian [B x D]
    logvar (Tensor) Standard deviation of the latent Gaussian [B x D]
    return (Tensor) [B x D]

    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    return eps * std + mu

def get_p_reg(dataloader, model, device, args) -> torch.Tensor:
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
        p_reg = torch.load(os.path.join(args.save_path, f"p_reg_{hash}.pt"))
        logger.info(f"Loaded p_reg from {os.path.join(args.save_path, f'p_reg_{hash}.pt')}")
        return p_reg.to(device=device)

    # If not, compute p_reg
    logger.info(f"p_reg file not found at {os.path.join(args.save_path, f'p_reg_{hash}.pt')}, computing p_reg.")
    model.to(device)
    model.eval()
    all_fea = []
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(dataloader):
            data  = data.to(device, non_blocking=True)
            fea, res = model(data)
            if batch_idx == 0:
                model_res = res
                all_fea = fea
            else:
                model_res = torch.cat((model_res, res))
                all_fea = torch.cat((all_fea, fea))

    mu, var = top_k_p_reg(model_res, all_fea, n_classes=model.num_classes, top_N=args.preprocessing.top_k)
    p_reg = reparameterize(mu, var)
    torch.save(p_reg, os.path.join(args.save_path, f"p_reg_{hash}.pt"))
    logger.info(f"Saved p_reg to {os.path.join(args.save_path, f'p_reg_{hash}.pt')}")
    return p_reg

class PseudoDataset(Dataset):
    def __init__(self, x, y):
        """Dataset for Pseudo labeled data.

        Args:
            x (torch.Tensor): Tensor of input images.
            y (torch.Tensor): Tensor of labels.

        """
        self.x = x
        self.y = y

        # Flag to return condition dict for pretraining.
        self.return_cond = False

    def __len__(self):
        """Return the total number of samples."""
        return len(self.y)

    def __getitem__(self, idx):
        """Retrieve the image and its corresponding label at index 'idx'."""
        image = self.x[idx]
        label = self.y[idx]

        cond = {}
        # Return condition dict if specified. Diff-Mi pretraining requires this.
        if self.return_cond:
            cond["y"] = np.array(self.y[idx], dtype=np.int64)
            return image, cond

        return image, label

def top_n_pseudo_label_dataset(
        dataloader: torch.utils.data.DataLoader,
        target_model: torch.nn.Module,
        device: torch.device,
        num_classes: int= 1000,
        top_n: int = 30,
        save_dir: str = "./data/"
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

    # Ensure num_workers is 0
    dataloader.num_workers = 0

    # Check if pseudo labeled data already exists
    if os.path.exists(os.path.join(save_dir, f"pseudo_data_{model_hash}.pkl")):
        with open(os.path.join(save_dir, f"pseudo_data_{model_hash}.pkl"), "rb") as file:
            pseudo_data, pseudo_labels = pickle.load(file)
            logger.info(f"Load data from {os.path.join(save_dir, f'pseudo_data_{model_hash}.pkl')}")

    # If not, compute the pseudo labeled data
    else:
        target_model.to(device)
        pseudo_data = []

        # Initialize variables to store all probabilities and images
        cpu_all_probs = None
        cpu_all_images = None

        with torch.no_grad():
            for i, (images, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
                images = images.to(device=device, non_blocking=True)
                _, prob = target_model(images)

                if i == 0 or all_probs is None:
                    all_probs = prob
                    all_images = images
                else:
                    all_probs = torch.cat((all_probs, prob), dim=0)
                    all_images = torch.cat((all_images, images), dim=0)

                # Save memory by offloading to CPU every 100 batches
                if i % 100 == 0 and i > 0:
                    if cpu_all_images is None:
                        cpu_all_probs = all_probs.cpu()
                        cpu_all_images = all_images.cpu()
                    else:
                        cpu_all_probs =  torch.cat((cpu_all_probs, all_probs.cpu()), dim=0)
                        cpu_all_images =  torch.cat((cpu_all_images, all_images.cpu()), dim=0)
                    all_probs = None
                    all_images = None

            # Handle the last batch if it exists
            if all_probs is not None:
                if cpu_all_probs is None:
                    cpu_all_probs = all_probs.cpu()
                    cpu_all_images = all_images.cpu()
                else:
                    cpu_all_probs = torch.cat((cpu_all_probs, all_probs.cpu()), dim=0)
                    cpu_all_images = torch.cat((cpu_all_images, all_images.cpu()), dim=0)

            # Fetch top-n samples per class
            for class_idx in tqdm(range(num_classes)):
                _, top_n_indices = torch.topk(cpu_all_probs[:, class_idx], k=top_n, dim=0)
                if class_idx == 0:
                    pseudo_data = cpu_all_images[top_n_indices]
                    pseudo_labels = np.array([class_idx]*top_n, dtype=np.int32)
                else:
                    pseudo_data = torch.cat((pseudo_data, cpu_all_images[top_n_indices]), dim=0)
                    pseudo_labels = np.concatenate((pseudo_labels, np.array([class_idx]*top_n, dtype=np.int32)), axis=0)
            pseudo_data = np.asarray(pseudo_data, dtype=np.float32)

        # Save the pseudo labeled data for future use
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"pseudo_data_{model_hash}.pkl"), "wb") as file:
            pickle.dump((pseudo_data, pseudo_labels), file)
            logger.info(f"Save data to {os.path.join(save_dir, f'pseudo_data_{model_hash}.pkl')}")

    return PseudoDataset(x=np.asarray(pseudo_data), y=np.asarray(pseudo_labels))

def extract_features(model, dataloader, device, save_dir: str = "./data/") -> None:
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
            priv_features, idents = pickle.load(file)
            logger.info(f"Load data from {os.path.join(save_dir, f'private_features_{model_hash}.pkl')}")

    # If not, compute the private features
    else:
        model.to(device)

        # Initialize variables to store all probabilities and images
        cpu_all_feats = None
        cpu_all_labels = None

        with torch.no_grad():
            for i, (images, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
                images = images.to(device=device, non_blocking=True)
                feats, _ = model(images)

                if i == 0 or all_feats is None:
                    all_feats = feats
                    all_labels = labels
                else:
                    all_feats = torch.cat((all_feats, feats), dim=0)
                    all_labels = torch.cat((all_labels, labels), dim=0)

                # Save memory by offloading to CPU every 100 batches
                if i % 100 == 0 and i > 0:
                    if cpu_all_labels is None:
                        cpu_all_feats = all_feats.cpu()
                        cpu_all_labels = all_labels.cpu()
                    else:
                        cpu_all_feats =  torch.cat((cpu_all_feats, all_feats.cpu()), dim=0)
                        cpu_all_labels =  torch.cat((cpu_all_labels, all_labels.cpu()), dim=0)
                    all_feats = None
                    all_labels = None
            cpu_all_feats =  torch.cat((cpu_all_feats, all_feats.cpu()), dim=0)
            cpu_all_labels =  torch.cat((cpu_all_labels, all_labels.cpu()), dim=0)

            # Handle the last batch if it exists
            if all_feats is not None:
                if cpu_all_feats is None:
                    cpu_all_feats = all_feats.cpu()
                    cpu_all_labels = all_labels.cpu()
                else:
                    cpu_all_feats = torch.cat((cpu_all_feats, all_feats.cpu()), dim=0)
                    cpu_all_labels = torch.cat((cpu_all_labels, all_labels.cpu()), dim=0)

        priv_features = np.asarray(cpu_all_feats, dtype=np.float32)
        idents = np.asarray(cpu_all_labels, dtype=np.int32)
        # Save the private features for future use
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"private_features_{model_hash}.pkl"), "wb") as file:
            pickle.dump((priv_features, idents), file)
            logger.info(f"Save data to {os.path.join(save_dir, f'private_features_{model_hash}.pkl')}")

    return priv_features, idents

def args_to_dict(args, keys):
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
