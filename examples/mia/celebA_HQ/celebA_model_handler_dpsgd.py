"""CelebA-HQ DP-SGD model handler: train() and eval() with Opacus.

Usage:
    from celebA_data_handler import CelebADataHandler
    from celebA_model_handler_dpsgd import CelebAModelHandlerDPsgd
    leakpro = LeakPro(CelebADataHandler, config_path, model_handler=CelebAModelHandlerDPsgd)
"""

import os
import pickle
from pathlib import Path

import torch
from torch import cuda, device, no_grad, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator

from leakpro import AbstractInputHandler
from leakpro.schemas import EvalOutput, TrainingOutput
from leakpro.utils.logger import logger


_MISSING = object()


def _get_config_value(config, key: str, default=_MISSING):
    """Read a config value from either a dict-like or attribute-based config."""
    if isinstance(config, dict):
        if key in config:
            return config[key]
    elif config is not None and hasattr(config, key):
        return getattr(config, key)

    if default is not _MISSING:
        return default

    raise ValueError(f"Missing training config value: {key}")


def _resolve_train_config(configs, explicit_train_config: dict | None = None):
    """Resolve the training config from an explicit dict or handler configs."""
    if explicit_train_config is not None:
        return explicit_train_config.get("train", explicit_train_config)

    train_cfg = getattr(configs, "train", None)
    if train_cfg is not None:
        return train_cfg

    raise ValueError(
        "No training config could be resolved. Pass `train_config`, provide the "
        "argument explicitly, or initialize the handler with configs containing a `train` section."
    )


def _resolve_epochs(epochs: int | None, configs=None, train_config: dict | None = None) -> int:
    """Resolve epochs from the explicit argument or the training config."""
    if epochs is not None:
        return epochs

    resolved_train_config = _resolve_train_config(configs, train_config)
    return int(_get_config_value(resolved_train_config, "epochs"))


def _build_optimizer(
    model: torch.nn.Module,
    optimizer: optim.Optimizer | None = None,
    configs=None,
    train_config: dict | None = None,
) -> optim.Optimizer:
    """Build an optimizer from config when one is not supplied explicitly."""
    if optimizer is not None:
        return optimizer

    resolved_train_config = _resolve_train_config(configs, train_config)
    optimizer_name = str(_get_config_value(resolved_train_config, "optimizer", "sgd")).lower()
    learning_rate = float(_get_config_value(resolved_train_config, "learning_rate"))
    weight_decay = float(_get_config_value(resolved_train_config, "weight_decay", 0.0))
    momentum = float(_get_config_value(resolved_train_config, "momentum", 0.0))

    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


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
        optimizer: optim.Optimizer | None = None,
        epochs: int | None = None,
        train_config: dict | None = None,
        dpsgd_metadata_path: str | None = None,
        virtual_batch_size: int = 16,
    ) -> TrainingOutput:
        if model is None:
            raise ValueError("Model must be provided for training")
        if criterion is None:
            raise ValueError("Criterion must be provided for training")
        if not hasattr(model, "dpsgd"):
            raise ValueError("Model is missing required 'dpsgd' attribute")

        configs = getattr(self, "configs", None)
        epochs = _resolve_epochs(epochs, configs=configs, train_config=train_config)
        optimizer = _build_optimizer(
            model=model,
            optimizer=optimizer,
            configs=configs,
            train_config=train_config,
        )
        self.optimizer = optimizer

        if getattr(dataloader, "batch_size", None) is not None:
            virtual_batch_size = min(dataloader.batch_size, virtual_batch_size)

        run_dpsgd = bool(model.dpsgd)

        if run_dpsgd:
            dpsgd_metadata_path = _resolve_dpsgd_metadata_path(configs, dpsgd_metadata_path)
            errors = ModuleValidator.validate(model, strict=False)
            if errors:
                logger.info("Model has DP incompatibilities; applying ModuleValidator.fix()")
                model = ModuleValidator.fix(model)
                optimizer = _rebuild_optimizer(optimizer, model)
                self.optimizer = optimizer
                logger.info(f"Model fixed and {optimizer.__class__.__name__} re-instantiated.")
            _disable_inplace_activations(model)
            model, optimizer, dataloader, _ = _setup_dpsgd(
                model, optimizer, dataloader, dpsgd_metadata_path)

        self.training_optimizer = optimizer

        dev = torch.device("cuda" if cuda.is_available() else "cpu")
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
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
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

    if not os.path.exists(dpsgd_path):
        raise FileNotFoundError(f"DP-SGD config not found: {dpsgd_path}")
    with open(dpsgd_path, "rb") as f:
        cfg = pickle.load(f)
    logger.info(f"DP-SGD config loaded from {dpsgd_path}: {cfg}")

    accountant = cfg.get("accountant", "prv")
    sample_rate = cfg.get("sample_rate", 1 / len(dataloader))
    try:
        noise_multiplier = get_noise_multiplier(
            target_epsilon=cfg["target_epsilon"],
            target_delta=cfg["target_delta"],
            sample_rate=sample_rate,
            epochs=cfg["epochs"],
            epsilon_tolerance=cfg.get("epsilon_tolerance", 0.01),
            accountant=accountant,
            eps_error=cfg.get("eps_error", 0.01),
        )
    except Exception as e:
        raise ValueError(
            f"Failed to compute noise multiplier using the '{accountant}' accountant "
            f"(epsilon={cfg['target_epsilon']}). Original error: {e}"
        )
    privacy_engine = PrivacyEngine(accountant=accountant)
    priv_model, priv_optimizer, priv_dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=cfg["max_grad_norm"],
    )
    return priv_model, priv_optimizer, priv_dataloader, privacy_engine
