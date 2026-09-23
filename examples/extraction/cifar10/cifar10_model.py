#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""CIFAR-10 target model and data helpers for the extraction example."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from improved_diffusion.script_util import (
    create_gaussian_diffusion,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet18_Weights, resnet18
from tqdm.auto import tqdm

from leakpro.attacks.extraction_attacks.adapters import CallableDiffusionAdapter
from leakpro.attacks.extraction_attacks.protocols import ConditionGradient, FeatureTransform
from leakpro.utils.device import get_device
from leakpro.utils.seed import seed_everything


@dataclass(frozen=True)
class TrainConfig:
    """Training and audit sizes for one notebook run."""

    seed: int
    train_size: int
    epochs: int
    train_batch_size: int
    learning_rate: float
    timesteps: int
    sampling_steps: int
    model_channels: int
    reference_size: int
    num_res_blocks: int = 3
    dropout: float = 0.3
    microbatch: int = 4
    ema_decay: float = 0.9999

    def __post_init__(self) -> None:
        """Reject configurations unsupported by the official CIFAR model."""
        for key in ("train_size", "epochs", "train_batch_size", "timesteps", "sampling_steps",
                    "model_channels", "reference_size", "num_res_blocks", "microbatch"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer.")
        if self.model_channels % 32:
            raise ValueError("model_channels must be a multiple of 32 for the published U-Net.")
        if not 2 <= self.sampling_steps <= self.timesteps:
            raise ValueError("sampling_steps must be between 2 and timesteps.")
        if self.reference_size > self.train_size:
            raise ValueError("reference_size must not exceed train_size.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1).")


MODEL_FORMAT_VERSION = 3
CHECKPOINT_EVERY_EPOCHS = 100


def select_device(requested: str = "auto") -> torch.device:
    """Resolve the device name from ``train_config.yaml``."""
    device = get_device() if requested == "auto" else torch.device(requested)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("The official Improved DDPM backend supports CPU and CUDA; use CPU on a Mac.")
    return device


def _model_options(train: TrainConfig) -> dict[str, Any]:
    """Use the Improved DDPM CIFAR-10 L_hybrid model configuration."""
    options = model_and_diffusion_defaults()
    options.update(
        image_size=32,
        num_channels=train.model_channels,
        num_res_blocks=train.num_res_blocks,
        dropout=train.dropout,
        learn_sigma=True,
        diffusion_steps=train.timesteps,
        noise_schedule="cosine",
        attention_resolutions="16,8",
        num_heads=4,
        use_scale_shift_norm=True,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
    )
    return options


def create_target_model(train: TrainConfig, device: torch.device) -> nn.Module:
    """Construct the official U-Net."""
    device = select_device(str(device))
    model, _diffusion = create_model_and_diffusion(**_model_options(train))
    return model.to(device)


class GaussianDiffusion:
    """Wrap OpenAI's cosine diffusion and eta-zero DDIM for the attack adapters."""

    def __init__(self, timesteps: int, sampling_steps: int, device: torch.device) -> None:
        if not 2 <= sampling_steps <= timesteps:
            raise ValueError("sampling_steps must be between 2 and timesteps.")
        device = select_device(str(device))
        self.num_timesteps = timesteps
        self.sampling_steps = sampling_steps
        self.device = device
        options = {
            "steps": timesteps,
            "learn_sigma": True,
            "noise_schedule": "cosine",
            "rescale_timesteps": True,
            "rescale_learned_sigmas": True,
        }
        self.process = create_gaussian_diffusion(**options)
        # Integer section respacing includes the terminal training timestep.
        self.sampler = create_gaussian_diffusion(**options, timestep_respacing=[sampling_steps])
        self.sample_schedule = torch.tensor(self.sampler.timestep_map, dtype=torch.long)
        self.betas = torch.tensor(self.process.betas, dtype=torch.float32, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.tensor(self.process.alphas_cumprod, dtype=torch.float32, device=device)

    @staticmethod
    def _at(values: Tensor, timesteps: Tensor, image_shape: tuple[int, ...]) -> Tensor:
        """Gather one scalar per image and add broadcast dimensions."""
        return values.gather(0, timesteps).reshape(timesteps.shape[0], *((1,) * (len(image_shape) - 1)))

    def q_sample(self, clean_images: Tensor, timesteps: Tensor, noise: Tensor) -> Tensor:
        """Use native integer timesteps for SIDE's forward noising."""
        return self.process.q_sample(clean_images, timesteps, noise=noise)

    def sample(
        self,
        model: nn.Module,
        batch_size: int,
        seed: int,
        *,
        labels: Tensor | None = None,
        gradient_fn: ConditionGradient | None = None,
    ) -> Tensor:
        """Sample with official clipping and optional SIDE score guidance.

        The eta-zero update follows improved_diffusion.gaussian_diffusion.ddim_sample.
        Guidance follows guided_diffusion.gaussian_diffusion.condition_score, while
        the SIDE classifier receives native indices rather than U-Net rescaled times.
        Unlike condition_score, this example projects the guided clean prediction
        to [-1, 1] and recomputes epsilon consistently. This deliberate stabilization
        limits terminal cosine-step amplification of classifier error; it is not
        part of the paper's guidance procedure. Additive clean-image increments
        avoid cancellation at near-zero alpha and leave zero guidance unchanged.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if gradient_fn is not None and (labels is None or labels.shape != (batch_size,)):
            raise ValueError("Classifier-guided sampling requires one label per image.")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        images = torch.randn((batch_size, 3, 32, 32), generator=generator, device="cpu").to(self.device)
        model.eval()
        for index in reversed(range(self.sampling_steps)):
            timesteps = torch.full((batch_size,), index, device=self.device, dtype=torch.long)
            with torch.no_grad():
                output = self.sampler.p_mean_variance(model, images, timesteps, clip_denoised=True)
                predicted_clean = output["pred_xstart"]
                predicted_noise = self.sampler._predict_eps_from_xstart(images, timesteps, predicted_clean)
            if gradient_fn is not None:
                native_step = int(self.sample_schedule[index])
                native_timesteps = torch.full_like(timesteps, native_step)
                gradient = gradient_fn(images, native_timesteps, labels).detach()
                if gradient.shape != images.shape or not torch.isfinite(gradient).all():
                    raise ValueError("The guidance gradient must match the images and contain finite values.")
                alpha = float(self.sampler.alphas_cumprod[index])
                guided_clean = (predicted_clean + (1.0 - alpha) / math.sqrt(alpha) * gradient).clamp(-1.0, 1.0)
                clean_change = guided_clean - predicted_clean
                predicted_noise = predicted_noise - math.sqrt(alpha / (1.0 - alpha)) * clean_change
                predicted_clean = guided_clean
            previous_alpha = float(self.sampler.alphas_cumprod_prev[index])
            images = math.sqrt(previous_alpha) * predicted_clean + math.sqrt(1.0 - previous_alpha) * predicted_noise
        return images.clamp(-1.0, 1.0).detach()


def load_cifar10(
    train: TrainConfig,
    work_dir: Path,
    data_dir: Path | None = None,
) -> tuple[Dataset, Tensor]:
    """Download CIFAR-10 and return the audited training subset and references."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda image: image.mul(2.0).sub(1.0)),
        ]
    )
    dataset_root = data_dir if data_dir is not None else work_dir / "data"
    full_train = CIFAR10(root=str(dataset_root), train=True, download=True, transform=transform)
    if train.train_size > len(full_train):
        raise ValueError("The requested CIFAR-10 training subset is too large.")
    if train.reference_size > train.train_size:
        raise ValueError("reference_size must not exceed the audited training subset size.")
    audited_train = Subset(full_train, range(train.train_size))
    references = torch.stack([audited_train[index][0] for index in range(train.reference_size)])
    return audited_train, references


def _update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """Update the sampling model after one optimizer step."""
    with torch.no_grad():
        for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
            ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)


def _save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Replace a checkpoint only after its new file is fully written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    torch.save(state, temporary_path)
    temporary_path.replace(path)


def _device_rng_state(device: torch.device) -> Tensor | None:
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device)
    return None


def _restore_device_rng(device: torch.device, state: Tensor | None) -> None:
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device)


def _train_epoch(
    model: nn.Module,
    ema_model: nn.Module,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    noise_generator: torch.Generator,
    train: TrainConfig,
    device: torch.device,
) -> float:
    """Accumulate each effective batch before updating AdamW and EMA."""
    model.train()
    loss_sum = 0.0
    image_count = 0
    for clean_images, _labels in loader:
        clean_images = clean_images.to(device)
        timesteps = torch.randint(
            0,
            train.timesteps,
            (clean_images.shape[0],),
            generator=noise_generator,
            device="cpu",
        ).to(device)
        noise = torch.randn(clean_images.shape, generator=noise_generator, device="cpu").to(device)
        optimizer.zero_grad(set_to_none=True)
        batch_count = clean_images.shape[0]
        for start in range(0, batch_count, train.microbatch):
            stop = min(start + train.microbatch, batch_count)
            losses = diffusion.process.training_losses(
                model,
                clean_images[start:stop],
                timesteps[start:stop],
                noise=noise[start:stop],
            )["loss"]
            if not torch.isfinite(losses).all():
                raise RuntimeError("Diffusion training produced a non-finite loss; no checkpoint was saved.")
            # Weight the final short microbatch by its image count.
            (losses.sum() / batch_count).backward()
            loss_sum += float(losses.detach().sum())
        optimizer.step()
        _update_ema(ema_model, model, decay=train.ema_decay)
        image_count += batch_count
    return loss_sum / image_count


def train_or_load_target(
    train: TrainConfig,
    train_dataset: Dataset,
    work_dir: Path,
    device: torch.device,
    *,
    force_retrain: bool = False,
) -> tuple[nn.Module, GaussianDiffusion, Path, list[float]]:
    """Load a completed target or resume training from a completed epoch.

    A separate resume file preserves training state every 100 epochs. Only the
    final checkpoint is returned as an audit target.
    """
    checkpoint_path = work_dir / "cifar10_ddpm.pt"
    resume_path = checkpoint_path.with_suffix(".resume.pt")
    if len(train_dataset) != train.train_size:
        raise ValueError("The training dataset length must match train.train_size.")
    if force_retrain:
        checkpoint_path.unlink(missing_ok=True)
        resume_path.unlink(missing_ok=True)
    training_settings = {
        key: value for key, value in asdict(train).items()
        if key not in ("sampling_steps", "reference_size")
    }
    expected_metadata = {"format_version": MODEL_FORMAT_VERSION, "training": training_settings}
    seed_everything(train.seed)
    model = create_target_model(train, device)
    diffusion = GaussianDiffusion(train.timesteps, train.sampling_steps, device)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("metadata") != expected_metadata:
            raise ValueError(f"Checkpoint metadata does not match the training configuration: {checkpoint_path}")
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return model, diffusion, checkpoint_path, list(checkpoint.get("epoch_losses", []))

    loader_generator = torch.Generator(device="cpu").manual_seed(train.seed)
    loader = DataLoader(
        train_dataset,
        batch_size=train.train_batch_size,
        shuffle=True,
        num_workers=0,
        generator=loader_generator,
    )
    noise_generator = torch.Generator(device="cpu").manual_seed(train.seed + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train.learning_rate, weight_decay=0.0)
    ema_model = copy.deepcopy(model).eval()
    epoch_losses: list[float] = []
    if resume_path.exists():
        resume = torch.load(resume_path, map_location="cpu", weights_only=True)
        if resume.get("metadata") != expected_metadata or resume.get("device_type") != device.type:
            raise ValueError(f"Resume metadata or device does not match this training run: {resume_path}")
        epoch_losses = list(resume["epoch_losses"])
        if not 0 < len(epoch_losses) <= train.epochs:
            raise ValueError(f"Resume checkpoint has an invalid completed epoch count: {resume_path}")
        model.load_state_dict(resume["model"])
        ema_model.load_state_dict(resume["ema_model"])
        optimizer.load_state_dict(resume["optimizer"])
        loader_generator.set_state(resume["loader_rng"])
        noise_generator.set_state(resume["noise_rng"])
        torch.set_rng_state(resume["torch_rng"])
        _restore_device_rng(device, resume["device_rng"])
    progress = tqdm(range(len(epoch_losses), train.epochs), desc="DDPM training epochs",
                    initial=len(epoch_losses), total=train.epochs)
    for _epoch in progress:
        epoch_losses.append(_train_epoch(model, ema_model, diffusion, loader, optimizer,
                                        noise_generator, train, device))
        progress.set_postfix(loss=f"{epoch_losses[-1]:.4f}")
        if len(epoch_losses) % CHECKPOINT_EVERY_EPOCHS == 0:
            _save_checkpoint(resume_path, {
                "metadata": expected_metadata,
                "device_type": device.type,
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch_losses": epoch_losses,
                "loader_rng": loader_generator.get_state(),
                "noise_rng": noise_generator.get_state(),
                "torch_rng": torch.get_rng_state(),
                "device_rng": _device_rng_state(device),
            })

    model.load_state_dict(ema_model.state_dict())
    model.eval()
    _save_checkpoint(checkpoint_path, {
        "metadata": expected_metadata,
        "model": model.state_dict(),
        "epoch_losses": epoch_losses,
    })
    resume_path.unlink(missing_ok=True)
    return model, diffusion, checkpoint_path, epoch_losses


def make_adapter(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    device: torch.device,
) -> CallableDiffusionAdapter:
    """Expose the notebook DDPM through LeakPro's white-box adapter."""

    def sample(batch_size: int, conditions: Sequence[Any] | None, seed: int) -> Tensor:
        if conditions is not None:
            raise ValueError("The CIFAR-10 DDPM is unconditional.")
        return diffusion.sample(model, batch_size, seed)

    def q_sample(clean_images: Tensor, timesteps: Tensor, noise: Tensor) -> Tensor:
        return diffusion.q_sample(clean_images.to(device), timesteps.to(device), noise.to(device))

    def guided_sample(
        batch_size: int,
        labels: Tensor,
        gradient_fn: ConditionGradient,
        seed: int,
    ) -> Tensor:
        return diffusion.sample(model, batch_size, seed, labels=labels, gradient_fn=gradient_fn)

    return CallableDiffusionAdapter(
        image_shape=(3, 32, 32),
        sample_fn=sample,
        num_timesteps=diffusion.num_timesteps,
        q_sample_fn=q_sample,
        guided_sample_fn=guided_sample,
    )


class ImageNetFeatureTransform(nn.Module):
    """Resize CIFAR-10 tensors and apply ImageNet normalization."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    def forward(self, images: Tensor) -> Tensor:
        """Return ImageNet-sized normalized tensors on the input device."""
        resized = functional.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        return (resized - self.mean.to(images.device)) / self.std.to(images.device)


def make_feature_extractor() -> tuple[nn.Module, FeatureTransform]:
    """Load a frozen ImageNet ResNet-18 for SIDE clustering."""
    extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
    extractor.fc = nn.Identity()
    extractor.eval()
    for parameter in extractor.parameters():
        parameter.requires_grad_(False)
    return extractor, ImageNetFeatureTransform()


def sha256_file(path: Path) -> str:
    """Return a checkpoint hash for the extraction result identity."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor(tensor: Tensor) -> str:
    """Hash tensor metadata and bytes in a device-independent representation."""
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    metadata = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)}, sort_keys=True).encode()
    raw_value = value.numpy().tobytes()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(len(raw_value).to_bytes(8, "big"))
    digest.update(raw_value)
    return digest.hexdigest()


def sha256_module_state(module: nn.Module) -> str:
    """Hash the named tensors that determine a module's outputs."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        header = json.dumps({"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}, sort_keys=True).encode()
        raw_value = value.numpy().tobytes()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(raw_value).to_bytes(8, "big"))
        digest.update(raw_value)
    return digest.hexdigest()
