#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""GDD_ENS DP-SGD model handler — train() and eval() with Opacus (role="model").

Mirrors ``examples/mia/cifar/cifar_model_handler_dpsgd.py``. DP is gated on the model's
``dpsgd`` attribute: when the shadow handler rebuilds a model from the target metadata (which
records ``dpsgd=True``), shadows are trained privately too, so the LiRA/RMIA likelihood-ratio
calibration is computed against DP-trained shadows rather than non-private ones.

The privacy budget is read from a pickled config dict (``dpsgd_dic.pkl``) resolved from
``target.dpsgd_path`` (or ``target.target_folder/dpsgd_dic.pkl``). The required noise multiplier
is computed with Opacus' accountant to hit ``target_epsilon`` at ``target_delta`` for the given
sample rate and epoch count.

Usage:
    from gdd_data_handler import GddDataHandler
    from gdd_model_handler_dpsgd import GddModelHandlerDPsgd
    leakpro = LeakPro(GddDataHandler, "audit_dpsgd.yaml", model_handler=GddModelHandlerDPsgd)
"""

import os
import pickle
from pathlib import Path

import torch
from torch import cuda, no_grad, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator

from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.schemas import EvalOutput, TrainingOutput
from leakpro.utils.logger import logger

DEFAULT_ACCOUNTANT = "rdp"


def _resolve_dpsgd_metadata_path(configs, explicit_path: str | None = None) -> str:
    """Resolve the DP-SGD metadata path from argument or target config.

    Resolution order: explicit argument, then ``target.dpsgd_path``, then
    ``target.target_folder/dpsgd_dic.pkl``.
    """
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
        "DP-SGD is enabled but no DP-SGD metadata path could be resolved. Set `target.dpsgd_path` "
        "in audit_dpsgd.yaml, or place `dpsgd_dic.pkl` inside `target.target_folder`."
    )


def _rebuild_optimizer(optimizer: optim.Optimizer, model: torch.nn.Module) -> optim.Optimizer:
    """Recreate the optimizer with the same hyperparameters for a new model parameter set."""
    opt_cls = optimizer.__class__
    opt_cfg = {k: v for group in optimizer.param_groups
               for k, v in group.items() if k != "params"}
    return opt_cls(model.parameters(), **opt_cfg)


def _disable_inplace_activations(model: torch.nn.Module) -> None:
    """Disable in-place activations that can break Opacus backward hooks."""
    for module in model.modules():
        if hasattr(module, "inplace") and isinstance(module.inplace, bool):
            module.inplace = False


class GddModelHandlerDPsgd(AbstractInputHandler, role="model"):
    """DP-SGD training handler for the GDD_ENS MLP. No dataset logic."""

    def train(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module = None,
        criterion: torch.nn.Module = None,
        optimizer: optim.Optimizer = None,
        epochs: int = None,
        dpsgd_metadata_path: str | None = None,
        virtual_batch_size: int = 32,
    ) -> TrainingOutput:
        if epochs is None:
            raise ValueError("epochs not found in configs")
        if optimizer is None:
            raise ValueError("Optimizer must be provided for training")
        if criterion is None:
            raise ValueError("Criterion must be provided for training")
        if not hasattr(model, "dpsgd"):
            raise ValueError("Model is missing required 'dpsgd' attribute")

        dev = torch.device("cuda" if cuda.is_available() else "cpu")
        run_dpsgd = bool(model.dpsgd)
        virtual_batch_size = min(dataloader.batch_size, virtual_batch_size)

        if run_dpsgd:
            dpsgd_metadata_path = _resolve_dpsgd_metadata_path(
                getattr(self, "configs", None), dpsgd_metadata_path)

            errors = ModuleValidator.validate(model, strict=False)
            if errors:
                logger.info("Model has DP incompatibilities — applying ModuleValidator.fix()")
                model = ModuleValidator.fix(model)
                optimizer = _rebuild_optimizer(optimizer, model)

            _disable_inplace_activations(model)
            model, optimizer, dataloader, _ = _setup_dpsgd(
                model, optimizer, dataloader, dpsgd_metadata_path, epochs)

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
        gpu_or_cpu = torch.device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)
        model.eval()
        loss, correct, total = 0.0, 0, 0
        with no_grad():
            for data, target in loader:
                data = data.to(gpu_or_cpu)
                target = target.to(gpu_or_cpu).long().view(-1)
                output = model(data)
                loss += criterion(output, target).item() * target.size(0)
                correct += output.argmax(dim=1).eq(target).sum().item()
                total += target.size(0)
        return EvalOutput(accuracy=float(correct) / total, loss=loss / total)


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


def _setup_dpsgd(model, optimizer, dataloader, dpsgd_path, epochs):
    if not bool(getattr(model, "dpsgd", False)):
        return model, optimizer, dataloader, None

    sample_rate = 1 / len(dataloader)  # already incorporates batch size
    if not os.path.exists(dpsgd_path):
        raise FileNotFoundError(f"DP-SGD config not found: {dpsgd_path}")
    with open(dpsgd_path, "rb") as f:
        cfg = pickle.load(f)
    logger.info(f"DP-SGD config loaded from {dpsgd_path}: {cfg}")

    # The noise multiplier is calibrated to cfg["epochs"], but the training loop runs `epochs`
    # (from the metadata recipe, which is also what shadow models replay). If these diverge the
    # reported epsilon would be wrong — fail loudly rather than misstate the privacy guarantee.
    if cfg["epochs"] != epochs:
        raise ValueError(
            f"DP-SGD epoch mismatch: privacy budget was calibrated for {cfg['epochs']} epochs "
            f"but training runs {epochs}. The reported epsilon would be invalid. Make "
            f"dpsgd_dic.pkl['epochs'] match the training/metadata epochs."
        )

    # Honor the configured accountant for BOTH calibration and the engine so the epsilon
    # guarantee cannot silently diverge (the cifar reference hardcodes "prv" and ignores
    # the dict's accountant field).
    accountant = cfg.get("accountant", DEFAULT_ACCOUNTANT)
    try:
        noise_multiplier = get_noise_multiplier(
            target_epsilon=cfg["target_epsilon"],
            target_delta=cfg["target_delta"],
            sample_rate=sample_rate,
            epochs=cfg["epochs"],
            epsilon_tolerance=cfg["epsilon_tolerance"],
            accountant=accountant,
            eps_error=cfg["eps_error"],
        )
    except Exception as e:
        raise ValueError(
            f"Failed to compute noise multiplier (ε={cfg['target_epsilon']}). "
            f"Try a larger epsilon. Original error: {e}"
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
