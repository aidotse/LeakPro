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
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet18_Weights, resnet18
from tqdm.auto import tqdm

from leakpro.attacks.extraction_attacks.adapters import CallableDiffusionAdapter
from leakpro.attacks.extraction_attacks.protocols import ConditionGradient, FeatureTransform


@dataclass(frozen=True)
class RunProfile:
    """Training and audit sizes for one notebook run."""

    name: str
    seed: int
    train_size: int
    epochs: int
    train_batch_size: int
    learning_rate: float
    timesteps: int
    sampling_steps: int
    model_channels: int
    reference_size: int


MODEL_FORMAT_VERSION = 2


def select_device(requested: str = "auto") -> torch.device:
    """Resolve the device name from ``train_config.yaml``."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed the libraries used by this example."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SinusoidalTimeEmbedding(nn.Module):
    """Embed integer diffusion steps with fixed sinusoidal features."""

    def __init__(self, dimensions: int) -> None:
        super().__init__()
        self.dimensions = dimensions

    def forward(self, timesteps: Tensor) -> Tensor:
        """Return one embedding per timestep."""
        half = self.dimensions // 2
        scale = math.log(10_000) / max(half - 1, 1)
        frequencies = torch.exp(
            -scale * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        if embedding.shape[1] < self.dimensions:
            embedding = functional.pad(embedding, (0, self.dimensions - embedding.shape[1]))
        return embedding


class ResidualTimeBlock(nn.Module):
    """Apply two convolutions with an additive timestep projection."""

    def __init__(self, in_channels: int, out_channels: int, time_dimensions: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time = nn.Linear(time_dimensions, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, images: Tensor, time_embedding: Tensor) -> Tensor:
        """Return the residual block output."""
        hidden = self.conv1(functional.silu(self.norm1(images)))
        hidden = hidden + self.time(functional.silu(time_embedding)).unsqueeze(-1).unsqueeze(-1)
        hidden = self.conv2(functional.silu(self.norm2(hidden)))
        return hidden + self.skip(images)


class SmallTimeUNet(nn.Module):
    """Predict CIFAR-10 diffusion noise with a two-level U-Net."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        time_dimensions = channels * 4
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(channels),
            nn.Linear(channels, time_dimensions),
            nn.SiLU(),
            nn.Linear(time_dimensions, time_dimensions),
        )
        self.input = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.encoder1 = ResidualTimeBlock(channels, channels, time_dimensions)
        self.down1 = nn.Conv2d(channels, channels * 2, kernel_size=4, stride=2, padding=1)
        self.encoder2 = ResidualTimeBlock(channels * 2, channels * 2, time_dimensions)
        self.down2 = nn.Conv2d(channels * 2, channels * 4, kernel_size=4, stride=2, padding=1)
        self.middle = ResidualTimeBlock(channels * 4, channels * 4, time_dimensions)
        self.up2 = nn.ConvTranspose2d(channels * 4, channels * 2, kernel_size=4, stride=2, padding=1)
        self.decoder2 = ResidualTimeBlock(channels * 4, channels * 2, time_dimensions)
        self.up1 = nn.ConvTranspose2d(channels * 2, channels, kernel_size=4, stride=2, padding=1)
        self.decoder1 = ResidualTimeBlock(channels * 2, channels, time_dimensions)
        self.output = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, 3, kernel_size=3, padding=1),
        )

    def forward(self, images: Tensor, timesteps: Tensor) -> Tensor:
        """Predict the noise added at each supplied timestep."""
        time_embedding = self.time_embedding(timesteps)
        first = self.encoder1(self.input(images), time_embedding)
        second = self.encoder2(self.down1(first), time_embedding)
        middle = self.middle(self.down2(second), time_embedding)
        decoded_second = self.decoder2(torch.cat((self.up2(middle), second), dim=1), time_embedding)
        decoded_first = self.decoder1(torch.cat((self.up1(decoded_second), first), dim=1), time_embedding)
        return self.output(decoded_first)


class GaussianDiffusion:
    """Implement DDPM forward noising and deterministic DDIM sampling."""

    def __init__(self, timesteps: int, sampling_steps: int, device: torch.device) -> None:
        if sampling_steps > timesteps:
            raise ValueError("sampling_steps must not exceed timesteps.")
        self.num_timesteps = timesteps
        self.sampling_steps = sampling_steps
        self.device = device
        self.betas = torch.linspace(1e-4, 2e-2, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sample_schedule = torch.linspace(0, timesteps - 1, sampling_steps).round().long().unique()

    @staticmethod
    def _at(values: Tensor, timesteps: Tensor, image_shape: tuple[int, ...]) -> Tensor:
        """Gather one scalar per image and add broadcast dimensions."""
        return values.gather(0, timesteps).reshape(timesteps.shape[0], *((1,) * (len(image_shape) - 1)))

    def q_sample(self, clean_images: Tensor, timesteps: Tensor, noise: Tensor) -> Tensor:
        """Sample q(x_t | x_0) for SIDE classifier training."""
        alpha_bar = self._at(self.alpha_bars, timesteps, tuple(clean_images.shape))
        return alpha_bar.sqrt() * clean_images + (1.0 - alpha_bar).sqrt() * noise

    def sample(
        self,
        model: nn.Module,
        batch_size: int,
        seed: int,
        *,
        labels: Tensor | None = None,
        gradient_fn: ConditionGradient | None = None,
    ) -> Tensor:
        """Run eta-zero DDIM sampling, optionally with SIDE score guidance."""
        generator = torch.Generator(device="cpu").manual_seed(seed)
        image_shape = (batch_size, 3, 32, 32)
        images = torch.randn(image_shape, generator=generator, device="cpu").to(self.device)
        model.eval()
        for schedule_index in reversed(range(self.sample_schedule.shape[0])):
            step = int(self.sample_schedule[schedule_index])
            timesteps = torch.full((batch_size,), step, device=self.device, dtype=torch.long)
            with torch.no_grad():
                predicted_noise = model(images, timesteps)
            alpha_bar = self._at(self.alpha_bars, timesteps, image_shape)
            if gradient_fn is not None:
                if labels is None:
                    raise ValueError("Classifier-guided sampling requires labels.")
                predicted_noise = predicted_noise - (1.0 - alpha_bar).sqrt() * gradient_fn(
                    images,
                    timesteps,
                    labels,
                )
            predicted_clean = (images - (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
            predicted_clean = predicted_clean.clamp(-1.0, 1.0)
            previous_alpha_bar = (
                self.alpha_bars[int(self.sample_schedule[schedule_index - 1])]
                if schedule_index > 0
                else torch.ones((), device=self.device)
            )
            images = previous_alpha_bar.sqrt() * predicted_clean + (1.0 - previous_alpha_bar).sqrt() * predicted_noise
        return images.clamp(-1.0, 1.0)


def load_cifar10(
    profile: RunProfile,
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
    if profile.train_size > len(full_train):
        raise ValueError("The requested CIFAR-10 training subset is too large.")
    if profile.reference_size > profile.train_size:
        raise ValueError("reference_size must not exceed the audited training subset size.")
    audited_train = Subset(full_train, range(profile.train_size))
    references = torch.stack([audited_train[index][0] for index in range(profile.reference_size)])
    return audited_train, references


def _update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """Update the sampling model after one optimizer step."""
    with torch.no_grad():
        for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
            ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)


def train_or_load_target(
    profile: RunProfile,
    train_dataset: Dataset,
    work_dir: Path,
    device: torch.device,
) -> tuple[SmallTimeUNet, GaussianDiffusion, Path, list[float]]:
    """Load a matching checkpoint or train and atomically save one."""
    checkpoint_path = work_dir / f"cifar10_ddpm_{profile.name}.pt"
    expected_metadata = {"format_version": MODEL_FORMAT_VERSION, "profile": asdict(profile)}
    model = SmallTimeUNet(profile.model_channels).to(device)
    diffusion = GaussianDiffusion(profile.timesteps, profile.sampling_steps, device)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("metadata") != expected_metadata:
            raise ValueError(f"Checkpoint metadata does not match the {profile.name!r} profile: {checkpoint_path}")
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return model, diffusion, checkpoint_path, list(checkpoint.get("epoch_losses", []))

    loader_generator = torch.Generator(device="cpu").manual_seed(profile.seed)
    loader = DataLoader(
        train_dataset,
        batch_size=profile.train_batch_size,
        shuffle=True,
        num_workers=0,
        generator=loader_generator,
    )
    noise_generator = torch.Generator(device="cpu").manual_seed(profile.seed + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=profile.learning_rate)
    ema_model = copy.deepcopy(model).eval()
    epoch_losses: list[float] = []
    progress = tqdm(range(profile.epochs), desc="DDPM training epochs")
    for _epoch in progress:
        model.train()
        loss_sum = 0.0
        image_count = 0
        for clean_images, _labels in loader:
            clean_images = clean_images.to(device)
            timesteps = torch.randint(
                0,
                profile.timesteps,
                (clean_images.shape[0],),
                generator=noise_generator,
                device="cpu",
            ).to(device)
            noise = torch.randn(clean_images.shape, generator=noise_generator, device="cpu").to(device)
            noisy_images = diffusion.q_sample(clean_images, timesteps, noise)
            predicted_noise = model(noisy_images, timesteps)
            loss = functional.mse_loss(predicted_noise, noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            _update_ema(ema_model, model, decay=0.995)
            loss_sum += float(loss.detach()) * clean_images.shape[0]
            image_count += clean_images.shape[0]
        epoch_losses.append(loss_sum / image_count)
        progress.set_postfix(loss=f"{epoch_losses[-1]:.4f}")

    model.load_state_dict(ema_model.state_dict())
    model.eval()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(".tmp")
    torch.save(
        {"metadata": expected_metadata, "model": model.state_dict(), "epoch_losses": epoch_losses},
        temporary_path,
    )
    temporary_path.replace(checkpoint_path)
    return model, diffusion, checkpoint_path, epoch_losses


def make_adapter(
    model: SmallTimeUNet,
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
    """Return a checkpoint fingerprint for the extraction result identity."""
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


def sha256_mapping(value: dict[str, str]) -> str:
    """Hash a string mapping using one canonical JSON representation."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
