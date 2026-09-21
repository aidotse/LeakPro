#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""The one DP-SGD training path shared by every PET optimization run.

Two entry points, one loop:

* :func:`fit_dpsgd` takes a model, loader, criterion and optimizer that the
  caller already built — the shape of ``AbstractInputHandler.train`` — and runs
  them under Opacus. Handlers call it so that the target and the RMIA shadow
  models are trained by literally the same code.
* :func:`train_with_dpsgd` takes a :class:`PETRecipe` (factories) and a sampled
  config, builds the pieces, and calls :func:`fit_dpsgd`.

Every model, private or not, first goes through :func:`make_opacus_compatible`,
so all points on one frontier share an architecture. Two training loops with
different rewrites would make the utility axis incomparable end to end.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from types import MethodType

import numpy as np
import torch
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from torch.nn import Module, Parameter
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from leakpro.utils.logger import logger

DEFAULT_DELTA = 1e-5
DEFAULT_MAX_PHYSICAL_BATCH = 256
DEFAULT_ACCOUNTANT = "prv"


@dataclass(frozen=True)
class PETRecipe:
    """Structured training recipe: the factories a PET optimizer may intervene in.

    Args:
        make_model: config -> a fresh, untrained model.
        make_optimizer: (parameters, config) -> optimizer (reads e.g. ``learning_rate``).
        make_loader: (indices, config) -> training DataLoader (reads e.g. ``batch_size``).
        criterion: loss module shared by target and reference models.
        epochs: fixed number of training epochs.
        output_kind: "logits" for raw multiclass logits (CrossEntropyLoss models),
            "binary_probs" for a single sigmoid output column (BCELoss models).
            Tells downstream signal extraction how to read the model's output.

    """

    make_model: Callable[[dict], Module]
    make_optimizer: Callable[[Iterable[Parameter], dict], Optimizer]
    make_loader: Callable[[np.ndarray, dict], DataLoader]
    criterion: _Loss
    epochs: int
    output_kind: str = field(default="logits")

    def __post_init__(self) -> None:
        """Validate the output kind."""
        if self.output_kind not in ("logits", "binary_probs"):
            raise ValueError(f"output_kind must be 'logits' or 'binary_probs', got '{self.output_kind}'.")


def _patch_residual_blocks(model: Module) -> None:
    """Rewrite torchvision residual blocks to add the skip connection out of place.

    ``ModuleValidator`` cannot fix this: ``out += identity`` lives in the block's
    ``forward``, not in a submodule, and the in-place add overwrites the tensor
    Opacus's backward hooks need. Silently does nothing if torchvision is absent.
    """
    try:
        from torchvision.models.resnet import BasicBlock, Bottleneck
    except ImportError:
        return

    def basic_forward(self: Module, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

    def bottleneck_forward(self: Module, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)

    # Only patch blocks that still use the stock forward. A subclass overriding
    # forward (SE, attention, anything custom) would be silently replaced by the
    # vanilla residual path, changing the model instead of just making it
    # Opacus-safe.
    for module in model.modules():
        if isinstance(module, BasicBlock) and type(module).forward is BasicBlock.forward:
            module.forward = MethodType(basic_forward, module)
        elif isinstance(module, Bottleneck) and type(module).forward is Bottleneck.forward:
            module.forward = MethodType(bottleneck_forward, module)
        elif isinstance(module, (BasicBlock, Bottleneck)):
            logger.warning(
                f"{type(module).__name__} overrides forward; leaving it untouched. If it adds the "
                "residual in place, Opacus will reject it — rewrite that forward out of place."
            )


def make_opacus_compatible(model: Module) -> Module:
    """Rewrite a module until Opacus can attach per-sample gradient hooks to it.

    Three incompatibilities, in the order they bite: BatchNorm mixes samples
    within a batch, so ``ModuleValidator`` swaps it for GroupNorm; in-place
    activations overwrite tensors the hooks still need; and torchvision's
    residual blocks add their skip connection in place (see
    ``_patch_residual_blocks``).

    Returns the fixed module, which may be a different object than the input —
    build the optimizer from *this* model's parameters, not the original's
    (:func:`fit_dpsgd` does that rebinding for you).
    """
    if ModuleValidator.validate(model, strict=False):
        model = ModuleValidator.fix(model)
        logger.info("Model was not Opacus-compatible; ModuleValidator.fix() applied (BatchNorm -> GroupNorm).")

    for module in model.modules():
        if isinstance(getattr(module, "inplace", None), bool):
            module.inplace = False

    _patch_residual_blocks(model)
    return model


def _optimizer_covers(optimizer: Optimizer, model: Module) -> bool:
    """True iff ``optimizer`` steps exactly the parameters ``model`` currently has."""
    in_optimizer = {id(p) for group in optimizer.param_groups for p in group["params"]}
    return in_optimizer == {id(p) for p in model.parameters()}


def _rebind_optimizer(optimizer: Optimizer, model: Module) -> Optimizer:
    """Rebuild ``optimizer`` with the same hyperparameters over ``model``'s parameters.

    Needed after :func:`make_opacus_compatible` swapped submodules: ``ModuleValidator.fix``
    replaces BatchNorm children *in place*, so the root module is the same object but
    the new GroupNorm parameters are not in the optimizer and the detached BatchNorm
    parameters still are. Checking object identity of the root misses that; parameter
    identity does not.
    """
    settings = {k: v for group in optimizer.param_groups for k, v in group.items() if k != "params"}
    return type(optimizer)(model.parameters(), **settings)


def _unwrap(model: Module) -> Module:
    """Strip Opacus's GradSampleModule so callers get back the architecture they passed in."""
    return model._module if hasattr(model, "_module") else model


def fit_dpsgd(  # noqa: PLR0913
    model: Module,
    dataloader: DataLoader,
    criterion: Module,
    optimizer: Optimizer,
    epochs: int,
    noise_multiplier: float,
    max_grad_norm: float,
    device: str,
    delta: float = DEFAULT_DELTA,
    max_physical_batch: int = DEFAULT_MAX_PHYSICAL_BATCH,
    accountant: str = DEFAULT_ACCOUNTANT,
) -> Module:
    """Train an already-built model under DP-SGD; the single shared Opacus loop.

    This has the shape of ``AbstractInputHandler.train`` on purpose: a handler
    forwards its arguments here and the RMIA shadow models are then trained by
    exactly the code that trained the target, which is what makes them mimic it.

    Every model goes through :func:`make_opacus_compatible`, private or not, so
    all points on one frontier share an architecture. If the rewrite changed the
    parameter set, the optimizer is rebuilt over the new parameters. Check the
    log for the ``ModuleValidator`` notice: the submitted architecture is not
    necessarily what gets trained.

    ``noise_multiplier > 0`` wraps the loop in a ``PrivacyEngine`` with the
    physical batch capped at ``max_physical_batch`` (per-example gradients cost
    batch x params memory; the logical batch size and the accounting are
    unchanged). ``noise_multiplier == 0`` runs the plain loop — the non-private
    anchor, ε = ∞.

    ``accountant`` must match whatever else in the pipeline reports ε, or the
    numbers are not comparable: PRV is tighter than RDP, so the same noise reads
    as a smaller ε under "prv". The accountant is therefore stored next to ε.

    Returns the trained model, unwrapped from Opacus, in eval mode, with the
    formal ``(ε, δ)`` and the accountant attached as ``model.dp_accounting``.
    """
    private = noise_multiplier > 0

    model = make_opacus_compatible(model).to(device)
    if not _optimizer_covers(optimizer, model):
        optimizer = _rebind_optimizer(optimizer, model)

    def run_epochs(loader: DataLoader) -> None:
        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb.to(device)), yb.to(device))
                loss.backward()
                optimizer.step()

    if private:
        engine = PrivacyEngine(accountant=accountant)
        model, optimizer, private_loader = engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )
        with BatchMemoryManager(data_loader=private_loader, max_physical_batch_size=max_physical_batch,
                                optimizer=optimizer) as mem_loader:
            run_epochs(mem_loader)
        epsilon = engine.get_epsilon(delta=delta)
    else:
        run_epochs(dataloader)
        epsilon = float("inf")

    logger.info(f"Trained model: formal epsilon = {epsilon:.2f} (delta = {delta}, accountant = "
                f"{accountant if private else 'none'}).")
    model = _unwrap(model).eval()
    # The accountant travels with the number: PRV and RDP epsilons are not
    # comparable, so a record that does not name its accountant cannot be
    # safely compared with one from another run.
    model.dp_accounting = {"epsilon": epsilon, "delta": delta, "accountant": accountant if private else None}
    return model


def train_with_dpsgd(  # noqa: PLR0913
    recipe: PETRecipe,
    config: dict,
    train_indices: np.ndarray,
    device: str,
    delta: float = DEFAULT_DELTA,
    max_physical_batch: int = DEFAULT_MAX_PHYSICAL_BATCH,
    accountant: str = DEFAULT_ACCOUNTANT,
) -> Module:
    """Build model, loader and optimizer from ``recipe`` under ``config`` and run :func:`fit_dpsgd`.

    ``config`` must carry ``noise_multiplier`` and ``max_grad_norm``; the recipe's
    factories read the rest (learning rate, batch size, ...). ``noise_multiplier``
    is never defaulted: a missing or misspelled key would silently train a fully
    non-private model from a function whose whole purpose is DP-SGD. Non-private
    runs must say so by passing an explicit 0.
    """
    if "noise_multiplier" not in config:
        raise KeyError(
            "config has no 'noise_multiplier'. Pass an explicit 0.0 for the non-private anchor; "
            "defaulting it would silently disable DP-SGD."
        )
    if "max_grad_norm" not in config:
        raise KeyError("config has no 'max_grad_norm'; the clipping norm is a required DP-SGD knob.")

    loader = recipe.make_loader(train_indices, config)
    model = recipe.make_model(config)
    optimizer = recipe.make_optimizer(model.parameters(), config)
    return fit_dpsgd(
        model, loader, recipe.criterion, optimizer, recipe.epochs,
        noise_multiplier=float(config["noise_multiplier"]),
        max_grad_norm=float(config["max_grad_norm"]),
        device=device, delta=delta, max_physical_batch=max_physical_batch, accountant=accountant,
    )
