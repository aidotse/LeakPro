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

import numpy as np
import torch
from opacus import GradSampleModule, PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from leakpro.utils.logger import logger

try:
    from torchvision.models.resnet import BasicBlock, Bottleneck
except ImportError:  # torchvision is optional; without it there are no blocks to patch
    BasicBlock = Bottleneck = None

DEFAULT_DELTA = 1e-5
DEFAULT_MAX_PHYSICAL_BATCH = 256
DEFAULT_ACCOUNTANT = "prv"
# GDP is left out on purpose: Opacus's GaussianAccountant is experimental and
# can underestimate the privacy spent (arXiv:2106.02848). An audit tool must
# not offer an option that can report a smaller epsilon than was spent.
ACCOUNTANTS = ("prv", "rdp")

Criterion = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class PETRecipe:
    """Structured training recipe: the factories a PET optimizer may intervene in.

    Args:
        make_model: config -> a fresh, untrained model.
        make_optimizer: (parameters, config) -> optimizer (reads e.g. ``learning_rate``).
        make_loader: (indices, config) -> training DataLoader (reads e.g. ``batch_size``).
        criterion: loss shared by target and reference models — any callable
            ``(output, target) -> loss``, e.g. ``nn.CrossEntropyLoss()`` or ``F.cross_entropy``.
        epochs: fixed number of training epochs.
        output_kind: "logits" for raw multiclass logits (CrossEntropyLoss models),
            "binary_probs" for a single sigmoid output column (BCELoss models).
            Tells downstream signal extraction how to read the model's output.

    """

    make_model: Callable[[dict], Module]
    make_optimizer: Callable[[Iterable[Parameter], dict], Optimizer]
    make_loader: Callable[[np.ndarray, dict], DataLoader]
    criterion: Criterion
    epochs: int
    output_kind: str = field(default="logits")

    def __post_init__(self) -> None:
        """Validate the output kind."""
        if self.output_kind not in ("logits", "binary_probs"):
            raise ValueError(f"output_kind must be 'logits' or 'binary_probs', got '{self.output_kind}'.")


if BasicBlock is not None:
    # Module-level classes, so a patched model still pickles: the class is found
    # by qualified name. (A bound method stored on the instance pickles as
    # getattr(obj, name) and fails on load.)

    class _OpacusBasicBlock(BasicBlock):
        """torchvision ``BasicBlock`` with the residual added out of place."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Stock BasicBlock forward, ``out + identity`` instead of ``out += identity``."""
            identity = x if self.downsample is None else self.downsample(x)
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return self.relu(out + identity)

    class _OpacusBottleneck(Bottleneck):
        """torchvision ``Bottleneck`` with the residual added out of place."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Stock Bottleneck forward, ``out + identity`` instead of ``out += identity``."""
            identity = x if self.downsample is None else self.downsample(x)
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            return self.relu(out + identity)


def _patch_residual_blocks(model: Module) -> None:
    """Rewrite torchvision residual blocks to add the skip connection out of place.

    ``ModuleValidator`` cannot fix this: ``out += identity`` lives in the block's
    ``forward``, not in a submodule, and the in-place add overwrites the tensor
    Opacus's backward hooks need. Each stock block's class is swapped for an
    out-of-place subclass. Does nothing if torchvision is absent.
    """
    if BasicBlock is None:
        return

    # Only exact stock blocks are swapped. A subclass may carry its own forward
    # (SE, attention) or other methods; swapping its class would silently change
    # the model instead of just making it Opacus-safe.
    for module in model.modules():
        if type(module) is BasicBlock:
            module.__class__ = _OpacusBasicBlock
        elif type(module) is Bottleneck:
            module.__class__ = _OpacusBottleneck
        elif isinstance(module, (BasicBlock, Bottleneck)) and not isinstance(
                module, (_OpacusBasicBlock, _OpacusBottleneck)):
            logger.warning(
                f"{type(module).__name__} subclasses a torchvision residual block; leaving it untouched. "
                "If its forward adds the residual in place, Opacus will reject it — rewrite it out of place."
            )


def make_opacus_compatible(model: Module) -> Module:
    """Rewrite a module until Opacus can attach per-sample gradient hooks to it.

    Three incompatibilities, in the order they bite: BatchNorm mixes samples
    within a batch, so ``ModuleValidator`` swaps it for GroupNorm; in-place
    activations overwrite tensors the hooks still need; and torchvision's
    residual blocks add their skip connection in place (see
    ``_patch_residual_blocks``).

    Returns the fixed module. When ``ModuleValidator.fix`` runs it returns a
    *copy* (the input keeps its BatchNorm), so every parameter is a new object:
    an optimizer built on the input no longer steps this model.
    :func:`fit_dpsgd` rebinds it for you.
    """
    if ModuleValidator.validate(model, strict=False):
        model = ModuleValidator.fix(model)
        logger.info("Model was not Opacus-compatible; ModuleValidator.fix() applied (BatchNorm -> GroupNorm).")

    for module in model.modules():
        if isinstance(getattr(module, "inplace", None), bool):
            module.inplace = False

    _patch_residual_blocks(model)
    return model


def _optimizer_is_stale(optimizer: Optimizer, model: Module) -> bool:
    """True iff ``optimizer`` steps a parameter that ``model`` no longer has.

    An optimizer over a *subset* of the model's parameters (e.g. head only) is
    not stale: freezing the rest is the caller's choice.
    """
    current = {id(p) for p in model.parameters()}
    return any(id(p) not in current for group in optimizer.param_groups for p in group["params"])


def _rebind_optimizer(optimizer: Optimizer, names_before: dict[int, str], model: Module) -> Optimizer:
    """Rebuild ``optimizer`` over ``model``, mapping each parameter by its qualified name.

    ``names_before`` maps ``id(param) -> name`` for the model the optimizer was
    built on. Each group keeps its own hyperparameters (including keys such as a
    scheduler's ``initial_lr``) and gets the new parameter that has the old one's
    name, so a BatchNorm's GroupNorm replacement lands in the BatchNorm's group.
    Parameters that were in no group stay out: a frozen part stays frozen.
    Optimizer state (e.g. momentum) is not carried over.
    """
    params_after = dict(model.named_parameters())
    groups, missing = [], []
    for group in optimizer.param_groups:
        new_params = []
        for p in group["params"]:
            name = names_before.get(id(p))
            if name in params_after:
                new_params.append(params_after[name])
            else:
                missing.append(name or "<unnamed>")
        groups.append({**{k: v for k, v in group.items() if k != "params"}, "params": new_params})
    if missing:
        logger.warning(f"Optimizer parameters with no counterpart after the Opacus rewrite were dropped: {missing}")
    if optimizer.state:
        logger.warning("Optimizer state (e.g. momentum) is not carried over to the rebound optimizer.")
    return type(optimizer)([g for g in groups if g["params"]], **optimizer.defaults)


def _unwrap(model: Module) -> Module:
    """Return the plain module from Opacus's GradSampleModule, with its hooks and ``grad_sample`` removed.

    Taking ``._module`` is not enough: the hooks stay registered (any later
    backward fails) and every parameter keeps its last per-sample gradient.
    """
    if not isinstance(model, GradSampleModule):
        return model
    # to_standard_module() deletes grad_sample from every parameter and raises
    # AttributeError on frozen ones that never got it; give them all one first.
    model.set_grad_sample_to_none()
    return model.to_standard_module()


def _warn_if_delta_too_large(delta: float, dataloader: DataLoader) -> None:
    """Warn when ``delta >= 1/n``: (0, delta)-DP then admits publishing each record with probability delta."""
    try:
        n = len(dataloader.dataset)
    except TypeError:  # iterable-style dataset, size unknown
        return
    if n and delta >= 1.0 / n:
        logger.warning(f"delta = {delta} is not below 1/n = {1.0 / n:.2e} for n = {n} training records; "
                       "a mechanism that publishes each record with probability delta, about delta*n >= 1 records in "
                       "the clear, still meets that guarantee. Choose delta well below 1/n.")


def fit_dpsgd(  # noqa: PLR0913
    model: Module,
    dataloader: DataLoader,
    criterion: Criterion,
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
    all points on one frontier share an architecture. If the rewrite replaced
    parameters the optimizer steps, the optimizer is rebuilt over the new ones,
    matched by name and keeping each parameter group's settings. Check the log
    for the ``ModuleValidator`` notice: the submitted architecture is not
    necessarily what gets trained.

    ``noise_multiplier > 0`` wraps the loop in a ``PrivacyEngine`` with the
    physical batch capped at ``max_physical_batch`` (per-example gradients cost
    batch x params memory; the logical batch size and the accounting are
    unchanged). ``noise_multiplier == 0`` runs the plain loop — the non-private
    anchor, ε = ∞.

    ``accountant`` must match whatever else in the pipeline reports ε, or the
    numbers are not comparable: PRV is tighter than RDP, so the same noise reads
    as a smaller ε under "prv". The accountant is therefore stored next to ε.
    Only "prv" and "rdp" are accepted (see ``ACCOUNTANTS``).

    ``delta`` should be well below ``1 / len(dataloader.dataset)``; a larger one
    is logged as a warning: it admits mechanisms that publish whole records.

    Returns the trained model, unwrapped from Opacus, in eval mode, with the
    formal ``(ε, δ)`` and the accountant attached as ``model.dp_accounting``.
    """
    if accountant not in ACCOUNTANTS:
        raise ValueError(f"accountant must be one of {ACCOUNTANTS}, got '{accountant}'.")
    private = noise_multiplier > 0
    if private:
        _warn_if_delta_too_large(delta, dataloader)

    names_before = {id(p): name for name, p in model.named_parameters()}
    model = make_opacus_compatible(model).to(device)
    if _optimizer_is_stale(optimizer, model):
        optimizer = _rebind_optimizer(optimizer, names_before, model)

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
