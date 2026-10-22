#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""CIFAR training handler for the webapp: upload this as the training handler in Step 2.

Provides ``train()`` and ``eval()``. Training uses DP-SGD when the webapp passes
``dpsgd_metadata_path``, and the plain loop otherwise; LeakPro's shadow models always take the plain
loop.

Two constraints come from how the webapp uses this file:

- It grafts ``train`` and ``eval`` onto its own handler class, so they must not rely on ``self``.
  The loops therefore live in module-level functions.
- It picks the first class in ``dir()`` of this module that has a ``train`` attribute. Importing an
  ``nn.Module`` subclass here (they all have ``train``) could make it pick that instead.

Unlike ``../cifar_model_handler.py``, this does not hold out part of the training set for early
stopping. Records held out that way are labelled members in the model metadata but never trained
on, which mislabels membership in the audit.
"""

import os
import pickle

import torch
from torch import no_grad, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakpro.schemas import EvalOutput, TrainingOutput
from leakpro.utils.device import get_device, mark_step


class CifarModelHandler:
    """Training handler for CIFAR, compatible with the LeakPro webapp."""

    def train(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module = None,
        criterion: torch.nn.Module = None,
        optimizer: optim.Optimizer = None,
        epochs: int = None,
        dpsgd_metadata_path: str = None,
        virtual_batch_size: int = 16,
    ) -> TrainingOutput:
        """Train the model; pass ``dpsgd_metadata_path`` to train with DP-SGD."""
        if epochs is None:
            raise ValueError("epochs must be provided")
        if dpsgd_metadata_path:
            return _train_dpsgd(dataloader, model, criterion, optimizer, epochs,
                                dpsgd_metadata_path, virtual_batch_size)
        return _train_standard(dataloader, model, criterion, optimizer, epochs)

    def eval(self, loader: DataLoader, model: torch.nn.Module, criterion: torch.nn.Module) -> EvalOutput:
        """Return loss and accuracy of ``model`` on ``loader``."""
        device = get_device()
        model.to(device)
        model.eval()
        loss, correct, total = 0.0, 0, 0
        with no_grad():
            for inputs, labels in loader:
                labels = labels.long().view(-1).to(device)
                outputs = model(inputs.to(device))
                mark_step(device)
                loss += criterion(outputs, labels).item() * labels.size(0)
                correct += outputs.argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        model.to("cpu")
        return EvalOutput(accuracy=correct / total if total else 0.0,
                          loss=loss / total if total else 0.0)


def _run_epochs(batches, model, criterion, optimizer, epochs, device, scheduler=None):  # noqa: ANN001, ANN202, PLR0913
    """Run the training loop and return the per-epoch accuracy and loss histories."""
    accuracy_history, loss_history = [], []
    for epoch in range(epochs):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        for inputs, labels in tqdm(batches, desc=f"Epoch {epoch + 1}/{epochs}"):
            labels = labels.long().view(-1).to(device)
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            mark_step(device)
            correct += outputs.argmax(1).eq(labels).sum().item()
            epoch_loss += loss.item() * labels.size(0)
            total += labels.size(0)
        if scheduler is not None:
            scheduler.step()
        accuracy_history.append(correct / total if total else 0.0)
        loss_history.append(epoch_loss / total if total else 0.0)
    return accuracy_history, loss_history


def _training_output(model, accuracy_history, loss_history) -> TrainingOutput:  # noqa: ANN001
    metrics = EvalOutput(
        accuracy=accuracy_history[-1], loss=loss_history[-1],
        extra={"accuracy_history": accuracy_history, "loss_history": loss_history},
    )
    return TrainingOutput(model=model, metrics=metrics)


def _train_standard(dataloader, model, criterion, optimizer, epochs) -> TrainingOutput:  # noqa: ANN001
    device = get_device()
    model.to(device)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    accuracy_history, loss_history = _run_epochs(dataloader, model, criterion, optimizer, epochs,
                                                 device, scheduler)
    model.to("cpu")
    return _training_output(model, accuracy_history, loss_history)


def _train_dpsgd(dataloader, model, criterion, optimizer, epochs,  # noqa: ANN001, PLR0913
                 dpsgd_metadata_path: str, virtual_batch_size: int) -> TrainingOutput:
    from opacus import PrivacyEngine  # noqa: PLC0415
    from opacus.accountants.utils import get_noise_multiplier  # noqa: PLC0415
    from opacus.utils.batch_memory_manager import BatchMemoryManager  # noqa: PLC0415
    from opacus.validators import ModuleValidator  # noqa: PLC0415

    if ModuleValidator.validate(model, strict=False):
        # fix() returns a new module (BatchNorm -> GroupNorm), so the optimizer must be rebuilt on its
        # parameters. Rebuilding from `defaults` keeps the configured hyperparameters.
        model = ModuleValidator.fix(model)
        optimizer = optimizer.__class__(model.parameters(), **optimizer.defaults)

    if not os.path.exists(dpsgd_metadata_path):
        raise FileNotFoundError(f"DP-SGD config not found: {dpsgd_metadata_path}")
    with open(dpsgd_metadata_path, "rb") as f:
        config = pickle.load(f)  # noqa: S301 - written by the webapp for this job

    accountant = config.get("accountant", "prv")
    noise_multiplier = get_noise_multiplier(
        target_epsilon=config["target_epsilon"],
        target_delta=config["target_delta"],
        sample_rate=1 / len(dataloader),
        epochs=config["epochs"],
        epsilon_tolerance=config.get("epsilon_tolerance", 0.01),
        accountant=accountant,
        eps_error=config.get("eps_error", 0.01),
    )
    privacy_engine = PrivacyEngine(accountant=accountant)
    private_model, optimizer, private_loader = privacy_engine.make_private(
        module=model, optimizer=optimizer, data_loader=dataloader,
        noise_multiplier=noise_multiplier, max_grad_norm=config["max_grad_norm"],
    )

    device = get_device()
    private_model.to(device)
    with BatchMemoryManager(data_loader=private_loader, max_physical_batch_size=virtual_batch_size,
                            optimizer=optimizer) as memory_safe_loader:
        accuracy_history, loss_history = _run_epochs(memory_safe_loader, private_model, criterion,
                                                     optimizer, epochs, device)

    # to_standard_module() removes Opacus' hooks and per-sample gradient buffers. Unwrapping via
    # ._module leaves them attached, and a later backward pass on the model then fails.
    model = private_model.to_standard_module().to("cpu")
    return _training_output(model, accuracy_history, loss_history)
