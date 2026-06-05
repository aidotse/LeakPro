"""CelebA-HQ DP-SGD model handler — train() and eval() with Opacus.

Usage:
    from celebA_data_handler import CelebADataHandler
    from celebA_model_handler_dpsgd import CelebAModelHandlerDPsgd
    leakpro = LeakPro(CelebADataHandler, config_path, model_handler=CelebAModelHandlerDPsgd)
"""

import os
import pickle
from pathlib import Path

import torch
from torch import no_grad, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator

from leakpro import AbstractInputHandler
from leakpro.schemas import EvalOutput, TrainingOutput
from leakpro.utils.device import get_device
from leakpro.utils.logger import logger


def _resolve_dpsgd_metadata_path(configs, explicit_path: str | None = None) -> str:
    """Resolve the DP-SGD metadata path from argument or target config."""
    if explicit_path:
        return explicit_path
    target_cfg = getattr(configs, "target", None)
    if target_cfg is not None:
        target_path = getattr(target_cfg, "dpsgd_path", None)
        if target_path:
            return target_path
        target_folder = getattr(target_cfg, "target_folder", None)
        if target_folder:
            return str(Path(target_folder) / "dpsgd_dic.pkl")
    raise FileNotFoundError(
        "DP-SGD is enabled but no DP-SGD metadata path could be resolved. "
        "Set `target.dpsgd_path` in audit.yaml, or place `dpsgd_dic.pkl` inside `target.target_folder`."
    )


def _rebuild_optimizer(optimizer: optim.Optimizer, model: torch.nn.Module) -> optim.Optimizer:
    opt_cls = optimizer.__class__
    opt_cfg = {k: v for group in optimizer.param_groups
               for k, v in group.items() if k != "params"}
    return opt_cls(model.parameters(), **opt_cfg)


def _disable_inplace_activations(model: torch.nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "inplace") and isinstance(module.inplace, bool):
            module.inplace = False


class CelebAModelHandlerDPsgd(AbstractInputHandler, role="model"):
    """DP-SGD training handler for CelebA-HQ. No dataset logic."""

    def train(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module = None,
        criterion: torch.nn.Module = None,
        optimizer: optim.Optimizer = None,
        epochs: int = None,
        dpsgd_metadata_path: str | None = None,
        virtual_batch_size: int = 16,
    ) -> TrainingOutput:
        if epochs is None:
            raise ValueError("epochs not found in configs")
        if optimizer is None:
            raise ValueError("Optimizer must be provided for training")
        if criterion is None:
            raise ValueError("Criterion must be provided for training")
        if not hasattr(model, "dpsgd"):
            raise ValueError("Model is missing required 'dpsgd' attribute")

        run_dpsgd = bool(model.dpsgd)

        if run_dpsgd:
            dpsgd_metadata_path = _resolve_dpsgd_metadata_path(
                getattr(self, "configs", None), dpsgd_metadata_path)
            errors = ModuleValidator.validate(model, strict=False)
            if errors:
                logger.info("Model has DP incompatibilities — applying ModuleValidator.fix()")
                model = ModuleValidator.fix(model)
                optimizer = _rebuild_optimizer(optimizer, model)
                logger.info(f"Model fixed and {optimizer.__class__.__name__} re-instantiated.")

        _disable_inplace_activations(model)
        model, optimizer, dataloader, _ = _setup_dpsgd(
            model, optimizer, dataloader, dpsgd_metadata_path)

        dev = get_device()
        model.to(dev)
        model.train()

        accuracy_history, loss_history = [], []

        if run_dpsgd:
            with BatchMemoryManager(data_loader=dataloader,
                                    max_physical_batch_size=virtual_batch_size,
                                    optimizer=optimizer) as mem_loader:
                for epoch in range(epochs):
                    train_loss, train_acc = _train_loop(
                        mem_loader, model, criterion, optimizer, dev, epoch + 1, epochs)
                    n = len(mem_loader.dataset)
                    accuracy_history.append(train_acc / n)
                    loss_history.append(train_loss / n)
        else:
            for epoch in range(epochs):
                train_loss, train_acc = _train_loop(
                    dataloader, model, criterion, optimizer, dev, epoch + 1, epochs)
                n = len(dataloader.dataset)
                accuracy_history.append(train_acc / n)
                loss_history.append(train_loss / n)

        model.to("cpu")
        if hasattr(model, "_module"):
            model = model._module

        train_accuracy = accuracy_history[-1] if accuracy_history else 0.0
        avg_train_loss = loss_history[-1] if loss_history else 0.0
        results = EvalOutput(
            accuracy=train_accuracy, loss=avg_train_loss,
            extra={"accuracy_history": accuracy_history, "loss_history": loss_history},
        )
        return TrainingOutput(model=model, metrics=results)

    def eval(self, loader, model, criterion) -> EvalOutput:
        gpu_or_cpu = get_device()
        model.to(gpu_or_cpu)
        model.eval()
        loss, acc, total_samples = 0, 0, 0
        with no_grad():
            for data, target in loader:
                data, target = data.to(gpu_or_cpu), target.to(gpu_or_cpu)
                target = target.view(-1)
                output = model(data)
                loss += criterion(output, target).item() * target.size(0)
                acc += output.argmax(dim=1).eq(target).sum().item()
                total_samples += target.size(0)
        return EvalOutput(accuracy=float(acc) / total_samples, loss=loss / total_samples)


def _train_loop(dataloader, model, criterion, optimizer, dev, epoch, epochs):
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}"):
        labels = labels.long().view(-1)
        inputs, labels = inputs.to(dev, non_blocking=True), labels.to(dev, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        pred = outputs.argmax(dim=1)
        loss.backward()
        optimizer.step()
        train_acc += pred.eq(labels.view_as(pred)).sum().item()
        train_loss += loss.item() * labels.size(0)
    return train_loss, train_acc


def _setup_dpsgd(model, optimizer, dataloader, dpsgd_path):
    if not bool(getattr(model, "dpsgd", False)):
        return model, optimizer, dataloader, None

    sample_rate = 1 / len(dataloader)
    if not os.path.exists(dpsgd_path):
        raise FileNotFoundError(f"DP-SGD config not found: {dpsgd_path}")
    with open(dpsgd_path, "rb") as f:
        cfg = pickle.load(f)
    logger.info(f"DP-SGD config loaded from {dpsgd_path}: {cfg}")
    try:
        noise_multiplier = get_noise_multiplier(
            target_epsilon=cfg["target_epsilon"],
            target_delta=cfg["target_delta"],
            sample_rate=sample_rate,
            epochs=cfg["epochs"],
            epsilon_tolerance=cfg["epsilon_tolerance"],
            accountant="prv",
            eps_error=cfg["eps_error"],
        )
    except Exception as e:
        raise ValueError(
            f"Failed to compute noise multiplier (ε={cfg['target_epsilon']}). "
            f"Try a larger epsilon. Original error: {e}"
        )
    privacy_engine = PrivacyEngine(accountant="prv")
    priv_model, priv_optimizer, priv_dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=cfg["max_grad_norm"],
    )
    return priv_model, priv_optimizer, priv_dataloader, privacy_engine
