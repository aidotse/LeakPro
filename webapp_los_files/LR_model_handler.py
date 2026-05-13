"""
LR_model_handler.py — Training handler for LOS/MIMIC-III Logistic Regression.

Upload this as handler.py in Step 2 of the LeakPro webapp.
Supports both standard training and DP-SGD (pass dpsgd_metadata_path to activate).

Loss: BCEWithLogitsLoss (model outputs raw logits, no sigmoid in forward).
"""
import os
import pickle

import torch
from torch import cuda, device, no_grad, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakpro.schemas import EvalOutput, TrainingOutput


class LRModelHandler:
    """Training handler for binary LOS classification with LR."""

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
        """Train the model. Pass dpsgd_metadata_path to activate DP-SGD."""
        if dpsgd_metadata_path:
            return _train_dpsgd(dataloader, model, optimizer, epochs,
                                dpsgd_metadata_path, virtual_batch_size)
        return _train_standard(dataloader, model, optimizer, epochs)

    def eval(self, loader: DataLoader, model: torch.nn.Module, criterion=None) -> EvalOutput:
        dev = device("cuda" if cuda.is_available() else "cpu")
        model.to(dev)
        model.eval()
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss, acc, total = 0.0, 0.0, 0
        with no_grad():
            for data, target in loader:
                target = target.float().view(-1).to(dev)
                data = data.to(dev)
                output = model(data).squeeze(1)
                loss += loss_fn(output, target).item() * target.size(0)
                acc += (output.sigmoid() >= 0.5).float().eq(target).sum().item()
                total += target.size(0)
        model.to("cpu")
        return EvalOutput(
            accuracy=acc / total if total else 0.0,
            loss=loss / total if total else 0.0,
        )


# ---------------------------------------------------------------------------
# Standard training
# ---------------------------------------------------------------------------

def _train_standard(dataloader, model, optimizer, epochs):
    if epochs is None:
        raise ValueError("epochs must be provided")

    criterion = torch.nn.BCEWithLogitsLoss()
    dev = device("cuda" if cuda.is_available() else "cpu")
    model.to(dev)
    accuracy_history, loss_history = [], []

    for epoch in range(epochs):
        model.train()
        train_loss, train_acc, total = 0.0, 0.0, 0
        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            labels = labels.float().view(-1).to(dev)
            inputs = inputs.to(dev)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_acc += (outputs.sigmoid() >= 0.5).float().eq(labels).sum().item()
            train_loss += loss.item() * labels.size(0)
            total += labels.size(0)
        accuracy_history.append(train_acc / total)
        loss_history.append(train_loss / total)

    model.to("cpu")
    metrics = EvalOutput(
        accuracy=accuracy_history[-1], loss=loss_history[-1],
        extra={"accuracy_history": accuracy_history, "loss_history": loss_history},
    )
    return TrainingOutput(model=model, metrics=metrics)


# ---------------------------------------------------------------------------
# DP-SGD training
# ---------------------------------------------------------------------------

def _train_dpsgd(dataloader, model, optimizer, epochs,
                 dpsgd_metadata_path, virtual_batch_size):
    if epochs is None:
        raise ValueError("epochs must be provided")

    from opacus import PrivacyEngine
    from opacus.accountants.utils import get_noise_multiplier
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    from opacus.validators import ModuleValidator

    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        model = ModuleValidator.fix(model)
        opt_cls = optimizer.__class__
        opt_cfg = {k: v for group in optimizer.param_groups
                   for k, v in group.items() if k != "params"}
        optimizer = opt_cls(model.parameters(), **opt_cfg)

    if not os.path.exists(dpsgd_metadata_path):
        raise FileNotFoundError(f"DP-SGD config not found: {dpsgd_metadata_path}")
    with open(dpsgd_metadata_path, "rb") as f:
        cfg = pickle.load(f)

    sample_rate = 1 / len(dataloader)
    noise_multiplier = get_noise_multiplier(
        target_epsilon=cfg["target_epsilon"],
        target_delta=cfg["target_delta"],
        sample_rate=sample_rate,
        epochs=cfg["epochs"],
        epsilon_tolerance=cfg["epsilon_tolerance"],
        accountant="prv",
        eps_error=cfg["eps_error"],
    )
    privacy_engine = PrivacyEngine(accountant="prv")
    model, optimizer, dataloader = privacy_engine.make_private(
        module=model, optimizer=optimizer, data_loader=dataloader,
        noise_multiplier=noise_multiplier, max_grad_norm=cfg["max_grad_norm"],
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    dev = torch.device("cuda" if cuda.is_available() else "cpu")
    model.to(dev)
    accuracy_history, loss_history = [], []

    with BatchMemoryManager(data_loader=dataloader,
                            max_physical_batch_size=virtual_batch_size,
                            optimizer=optimizer) as mem_loader:
        for epoch in range(epochs):
            model.train()
            train_loss, train_acc = 0.0, 0.0
            for inputs, labels in tqdm(mem_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                labels = labels.float().view(-1).to(dev)
                inputs = inputs.to(dev)
                optimizer.zero_grad()
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_acc += (outputs.sigmoid() >= 0.5).float().eq(labels).sum().item()
                train_loss += loss.item() * labels.size(0)
            n = len(mem_loader.dataset)
            accuracy_history.append(train_acc / n)
            loss_history.append(train_loss / n)

    model.to("cpu")
    if hasattr(model, "_module"):
        model = model._module

    metrics = EvalOutput(
        accuracy=accuracy_history[-1], loss=loss_history[-1],
        extra={"accuracy_history": accuracy_history, "loss_history": loss_history},
    )
    return TrainingOutput(model=model, metrics=metrics)
