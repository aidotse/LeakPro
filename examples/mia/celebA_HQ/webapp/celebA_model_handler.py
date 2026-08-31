"""CelebA-HQ training handler — upload this as handler.py in Step 2.

Provides train() and eval() for face identity classification.
Works with both standard training and DP-SGD (pass dpsgd_metadata_path to activate).
"""
import os
import pickle

import torch
from torch import cuda, device, no_grad, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakpro.schemas import EvalOutput, TrainingOutput


class CelebAModelHandler:
    """Training handler for CelebA-HQ. Compatible with LeakPro webapp."""

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
            return _train_dpsgd(dataloader, model, criterion, optimizer,
                                epochs, dpsgd_metadata_path, virtual_batch_size)
        return _train_standard(dataloader, model, criterion, optimizer, epochs)

    def eval(self, loader: DataLoader, model: torch.nn.Module, criterion) -> EvalOutput:
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)
        model.eval()
        loss, acc, total = 0.0, 0.0, 0
        with no_grad():
            for data, target in loader:
                target = target.long().view(-1).to(gpu_or_cpu)
                data = data.to(gpu_or_cpu)
                output = model(data)
                loss += criterion(output, target).item() * target.size(0)
                acc += output.argmax(1).eq(target).sum().item()
                total += target.size(0)
        model.to("cpu")
        return EvalOutput(accuracy=acc / total if total else 0.0,
                          loss=loss / total if total else 0.0)


# ---------------------------------------------------------------------------
# Standard training
# ---------------------------------------------------------------------------

def _train_standard(dataloader, model, criterion, optimizer, epochs):
    if epochs is None:
        raise ValueError("epochs must be provided")

    dev = device("cuda" if cuda.is_available() else "cpu")
    model.to(dev)
    accuracy_history, loss_history = [], []

    for epoch in range(epochs):
        model.train()
        train_loss, train_acc, total = 0.0, 0.0, 0
        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            labels = labels.long().view(-1)
            inputs, labels = inputs.to(dev), labels.to(dev)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_acc += outputs.argmax(1).eq(labels).sum().item()
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
# Helpers
# ---------------------------------------------------------------------------

def _fix_resnet_skip_connections(model):
    """Patch ResNet BasicBlock/Bottleneck to use non-inplace skip addition.

    Opacus forbids in-place ops on tensors tracked by its backward hooks.
    torchvision's ResNet uses `out += identity`; replace with `out = out + identity`.
    """
    import types
    try:
        from torchvision.models.resnet import BasicBlock, Bottleneck
    except ImportError:
        return

    def _bb_forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

    def _bn_forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

    for module in model.modules():
        if isinstance(module, BasicBlock):
            module.forward = types.MethodType(_bb_forward, module)
        elif isinstance(module, Bottleneck):
            module.forward = types.MethodType(_bn_forward, module)


# ---------------------------------------------------------------------------
# DP-SGD training
# ---------------------------------------------------------------------------

def _train_dpsgd(dataloader, model, criterion, optimizer, epochs,
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

    for module in model.modules():
        if hasattr(module, "inplace") and isinstance(module.inplace, bool):
            module.inplace = False

    _fix_resnet_skip_connections(model)

    if not os.path.exists(dpsgd_metadata_path):
        raise FileNotFoundError(f"DP-SGD config not found: {dpsgd_metadata_path}")
    with open(dpsgd_metadata_path, "rb") as f:
        cfg = pickle.load(f)

    sample_rate = 1 / len(dataloader)
    _accountant = cfg.get("accountant", "prv")
    noise_multiplier = get_noise_multiplier(
        target_epsilon=cfg["target_epsilon"],
        target_delta=cfg["target_delta"],
        sample_rate=sample_rate,
        epochs=cfg["epochs"],
        epsilon_tolerance=cfg.get("epsilon_tolerance", 0.01),
        accountant=_accountant,
        eps_error=cfg.get("eps_error", 0.01),
    )
    privacy_engine = PrivacyEngine(accountant=_accountant)
    model, optimizer, dataloader = privacy_engine.make_private(
        module=model, optimizer=optimizer, data_loader=dataloader,
        noise_multiplier=noise_multiplier, max_grad_norm=cfg["max_grad_norm"],
    )

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
                labels = labels.long().view(-1)
                inputs, labels = inputs.to(dev), labels.to(dev)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_acc += outputs.argmax(1).eq(labels).sum().item()
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
