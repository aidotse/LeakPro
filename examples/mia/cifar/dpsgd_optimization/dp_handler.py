#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""CIFAR DP-SGD handler for privacy-utility optimization runs.

This is *training* code, which LeakPro always leaves to the user's handler — the
attack is not here. The one thing it does specially is read the DP-SGD noise
multiplier and clipping norm *directly* from the ``dpsgd_dic.pkl`` the optimization run
writes, because those two are optimization knobs. A single DP-SGD training
function (:func:`dp_train`) is used for both the target and, through
:meth:`CifarDPHandler.train`, the RMIA shadow models — so shadow models are
trained under exactly the candidate target's configuration.
"""

import pickle
from pathlib import Path

import torch
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch import nn
from torch.utils.data import DataLoader

from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.schemas import EvalOutput, TrainingOutput
from leakpro.utils.logger import logger

_DEFAULT_DELTA = 1e-5
_MAX_PHYSICAL_BATCH = 256


class SmallCNN(nn.Module):
    """Opacus-compatible CIFAR CNN (GroupNorm, no BatchNorm); memorizes visibly, trains fast."""

    def __init__(self, num_classes: int = 10, dpsgd: bool = True) -> None:
        super().__init__()
        # Stored so LeakPro's get_model_init_params can rebuild identical replicas
        # (num_classes, dpsgd) for the shadow models.
        self.num_classes = num_classes
        self.dpsgd = dpsgd
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 8 * 8, 256), nn.ReLU(), nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Class logits for a batch of images."""
        return self.classifier(self.features(x))


def _load_dpsgd_config(dpsgd_path: str) -> dict:
    """Read the direct DP-SGD knobs (noise multiplier, clip norm) written by the optimization run."""
    path = Path(dpsgd_path)
    if not path.exists():
        raise FileNotFoundError(f"DP-SGD config not found: {dpsgd_path}")
    with path.open("rb") as f:
        return pickle.load(f)


def dp_train(  # noqa: PLR0913
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    dpsgd_path: str,
    device: str | None = None,
) -> TrainingOutput:
    """Train one model with DP-SGD (or plain SGD when noise multiplier is 0).

    The noise multiplier and clipping norm come straight from ``dpsgd_path``; the
    learning rate and batch size are already baked into ``optimizer`` and
    ``dataloader``. This is the single training path shared by the target and the
    RMIA reference models, which is what makes the references mimic the target.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _load_dpsgd_config(dpsgd_path)
    noise_multiplier = float(cfg["noise_multiplier"])
    max_grad_norm = float(cfg["max_grad_norm"])
    delta = float(cfg.get("delta", _DEFAULT_DELTA))

    model = model.to(device)
    engine = None
    if noise_multiplier > 0:
        engine = PrivacyEngine(accountant="rdp")
        model, optimizer, dataloader = engine.make_private(
            module=model, optimizer=optimizer, data_loader=dataloader,
            noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm,
        )

    def _epochs(loader: DataLoader) -> tuple[float, float]:
        last_acc, last_loss = 0.0, 0.0
        model.train()
        for _ in range(epochs):
            total_loss, correct, seen = 0.0, 0, 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device).long().view(-1)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * yb.size(0)
                correct += (out.argmax(1) == yb).sum().item()
                seen += yb.size(0)
            last_acc, last_loss = correct / max(seen, 1), total_loss / max(seen, 1)
        return last_acc, last_loss

    if noise_multiplier > 0:
        with BatchMemoryManager(data_loader=dataloader, max_physical_batch_size=_MAX_PHYSICAL_BATCH,
                                optimizer=optimizer) as mem_loader:
            acc, loss = _epochs(mem_loader)
        epsilon = engine.get_epsilon(delta=delta)
    else:
        acc, loss = _epochs(dataloader)
        epsilon = float("inf")

    model.to("cpu")
    if hasattr(model, "_module"):  # unwrap Opacus GradSampleModule before saving
        model = model._module
    logger.info(f"Trained CNN: acc={acc:.4f}, formal epsilon={epsilon:.2f} (delta={delta}).")
    return TrainingOutput(model=model, metrics=EvalOutput(accuracy=acc, loss=loss,
                                                          extra={"epsilon": epsilon, "delta": delta}))


class CifarDPHandler(AbstractInputHandler, role="full"):
    """Data + DP-SGD training handler for the CIFAR DP-SGD optimization example."""

    def train(  # noqa: PLR0913
        self,
        dataloader: DataLoader,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
    ) -> TrainingOutput:
        """Train a (shadow or target) model, reading DP knobs from the target's dpsgd config."""
        dpsgd_path = getattr(self.configs.target, "dpsgd_path", None)
        if not dpsgd_path:
            raise ValueError("target.dpsgd_path must be set for the DP-SGD optimization example.")
        return dp_train(model, dataloader, criterion, optimizer, epochs, dpsgd_path)

    def eval(self, dataloader: DataLoader, model: nn.Module, criterion: nn.Module) -> EvalOutput:
        """Accuracy and mean loss over ``dataloader``."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        total_loss, correct, seen = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in dataloader:
                xb, yb = xb.to(device), yb.to(device).long().view(-1)
                out = model(xb)
                total_loss += criterion(out, yb).item() * yb.size(0)
                correct += (out.argmax(1) == yb).sum().item()
                seen += yb.size(0)
        return EvalOutput(accuracy=correct / max(seen, 1), loss=total_loss / max(seen, 1))

    class UserDataset(AbstractInputHandler.UserDataset):
        """CIFAR images already normalized to standard scores; stores mean/std for reconstruction."""

        def __init__(self, data: torch.Tensor, targets: torch.Tensor, **kwargs: dict) -> None:
            self.data = data.float()
            self.targets = targets.long()
            for key, value in kwargs.items():
                setattr(self, key, value)

        def __getitem__(self, index: int) -> tuple:
            return self.data[index], self.targets[index]

        def __len__(self) -> int:
            return len(self.targets)
