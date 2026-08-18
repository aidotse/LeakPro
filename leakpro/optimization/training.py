#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Structured PET recipe and the shared DP-SGD training path.

A ``PETRecipe`` is the plan's structured training contract: instead of a
monolithic ``train(data) -> model``, the user provides factories the optimizer
can intervene in. Every factory receives the sampled knob configuration and
picks out what it needs — ``make_loader`` reads ``batch_size``,
``make_optimizer`` reads ``learning_rate``. Unused knobs are ignored, which is
what makes further PETs cheap: a regularization bundle is just ``make_model``
reading ``dropout`` and ``make_optimizer`` reading ``weight_decay``.

``train_with_dpsgd`` is the single Opacus path (privacy engine, physical-batch
cap, ε accounting) previously duplicated per example. ``build_campaign_fns``
turns a recipe plus data splits into the three callables a ``Campaign`` needs;
exotic training loops that do not fit the recipe keep the escape hatch of
writing those callables directly.
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

from leakpro.optimization.objectives import AttackScores, confidence_signal
from leakpro.utils.logger import logger

REQUIRED_SPLIT_KEYS = ("x", "y", "target_train", "ref_pool", "audit_members", "audit_nonmembers", "utility_eval")


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
            Decides how membership confidence is extracted.

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
    build the optimizer from *this* model's parameters, not the original's.
    """
    if ModuleValidator.validate(model, strict=False):
        model = ModuleValidator.fix(model)
        logger.info("Model was not Opacus-compatible; ModuleValidator.fix() applied (BatchNorm -> GroupNorm).")

    for module in model.modules():
        if isinstance(getattr(module, "inplace", None), bool):
            module.inplace = False

    _patch_residual_blocks(model)
    return model


def train_with_dpsgd(  # noqa: PLR0913
    recipe: PETRecipe,
    config: dict,
    train_indices: np.ndarray,
    device: str,
    delta: float = 1e-5,
    max_physical_batch: int = 256,
    accountant: str = "prv",
) -> Module:
    """Train one model under the sampled config; the single shared Opacus path.

    Every model goes through ``make_opacus_compatible`` regardless of noise, so
    all points on one frontier share an architecture: applying the BatchNorm to
    GroupNorm rewrite only to private configs would leave the non-private anchor
    a structurally different model, and its utility gap would then mix the cost
    of DP noise with the cost of an architecture change. The submitted
    architecture is therefore not necessarily what gets trained — check the log
    for the ModuleValidator notice.

    ``noise_multiplier > 0`` wraps the loop in a PrivacyEngine with the physical
    batch capped at ``max_physical_batch`` (per-example gradients cost
    batch x params memory; the sampled batch size and the accounting are
    unchanged). ``noise_multiplier == 0`` runs the plain loop — the non-private
    anchor, ε = ∞. The key is required either way: see below.

    ``accountant`` must match whatever else in the pipeline reports ε, or the
    numbers are not comparable: PRV is tighter than RDP, so the same noise reads
    as a smaller ε under "prv". It defaults to "prv" for agreement with the
    webapp's DP-SGD path.

    The formal (ε, δ) is stored on the returned model as ``campaign_extras``,
    which ``Campaign`` records next to the empirical attack result.
    """
    # Never default this: a missing or misspelled key would silently train a
    # fully non-private model from a function whose whole purpose is DP-SGD.
    # Non-private runs must say so by passing an explicit 0.
    if "noise_multiplier" not in config:
        raise KeyError(
            "config has no 'noise_multiplier'. Pass an explicit 0.0 for the non-private anchor; "
            "defaulting it would silently disable DP-SGD."
        )
    private = config["noise_multiplier"] > 0

    loader = recipe.make_loader(train_indices, config)
    # Applied on both branches so every point on a frontier shares one
    # architecture: rewriting BatchNorm only for private configs would make the
    # non-private anchor a different model, and its utility gap would then mix
    # the cost of DP noise with the cost of an architecture change.
    model = make_opacus_compatible(recipe.make_model(config)).to(device)
    optimizer = recipe.make_optimizer(model.parameters(), config)

    def run_epochs(epoch_loader) -> None:  # noqa: ANN001
        for _ in range(recipe.epochs):
            for xb, yb in epoch_loader:
                optimizer.zero_grad()
                loss = recipe.criterion(model(xb.to(device)), yb.to(device))
                loss.backward()
                optimizer.step()

    model.train()
    if private:
        engine = PrivacyEngine(accountant=accountant)
        model, optimizer, loader = engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=config["noise_multiplier"],
            max_grad_norm=config["max_grad_norm"],
        )
        with BatchMemoryManager(data_loader=loader, max_physical_batch_size=max_physical_batch,
                                optimizer=optimizer) as mem_loader:
            run_epochs(mem_loader)
        epsilon = engine.get_epsilon(delta=delta)
    else:
        run_epochs(loader)
        epsilon = float("inf")

    logger.info(f"Trained model: formal epsilon = {epsilon:.2f} (delta = {delta}).")
    # The accountant travels with the number: PRV and RDP epsilons are not
    # comparable, so a record that does not name its accountant cannot be
    # safely compared with one from another run.
    model.campaign_extras = {"epsilon": epsilon, "delta": delta,
                             "accountant": accountant if private else None}
    return model.eval()


def _free_cuda(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


@torch.no_grad()
def _evaluate_utility(model: Module, metric: str | Callable, x: torch.Tensor,
                      y: torch.Tensor, device: str) -> float:
    if callable(metric):
        return float(metric(model, x, y, device))
    if metric == "accuracy":
        correct = 0
        for i in range(0, len(x), 1024):
            out = model(x[i:i + 1024].to(device)).cpu()
            # A single-logit head is a binary classifier: threshold at 0, don't
            # argmax (argmax over one column is always 0 — silently 0% or 100%).
            pred = (out.reshape(-1) > 0).long() if (out.ndim == 1 or out.shape[-1] == 1) else out.argmax(dim=1)
            correct += int((pred == y[i:i + 1024].reshape(-1).long()).sum())
        return correct / len(x)
    if metric == "auc":
        from sklearn.metrics import roc_auc_score
        outs = [model(x[i:i + 4096].to(device)).cpu() for i in range(0, len(x), 4096)]
        scores = torch.cat(outs)
        # Flattening a multiclass output would interleave class scores with
        # sample labels and produce a meaningless number, so refuse instead.
        if scores.ndim > 1 and scores.shape[-1] != 1:
            raise ValueError(
                f"utility_metric='auc' needs a single-column model output, got shape {tuple(scores.shape)}. "
                "Use 'accuracy' for multiclass models, or pass a callable."
            )
        return float(roc_auc_score(y.numpy().ravel(), scores.reshape(-1).numpy()))
    raise ValueError(f"Unknown utility_metric '{metric}'; use 'accuracy', 'auc' or a callable.")


def build_campaign_fns(  # noqa: PLR0913
    recipe: PETRecipe,
    splits: dict,
    n_refs: int,
    device: str,
    utility_metric: str | Callable = "accuracy",
    ref_seed: int = 1,
    delta: float = 1e-5,
    max_physical_batch: int = 256,
    accountant: str = "prv",
) -> tuple[Callable, Callable, Callable]:
    """Turn a recipe + data splits into a Campaign's (train_fn, utility_fn, attack_fn).

    Args:
        recipe: the structured training recipe.
        splits: dict with keys ``x, y, target_train, ref_pool, audit_members,
            audit_nonmembers, utility_eval`` (audit members must be a subset of
            ``target_train``; the reference pool must be disjoint from it).
        n_refs: matched reference models per attack — full mimicry: references
            are trained with the candidate's config via the same recipe.
        device: torch device string.
        utility_metric: "accuracy", "auc", or a callable (model, x, y, device) -> float.
        ref_seed: seed for reference-pool subsampling.
        delta: privacy-accounting δ, passed to every training run.
        max_physical_batch: per-example-gradient memory cap; does not change the
            sampled batch size or the accounting.
        accountant: Opacus accountant ("prv", "rdp", "gdp"). Reference models use
            the same one as the target, so mimicry stays exact.

    """
    missing = [k for k in REQUIRED_SPLIT_KEYS if k not in splits]
    if missing:
        raise KeyError(f"splits is missing {missing}; required keys are {REQUIRED_SPLIT_KEYS}.")
    if len(splits["ref_pool"]) == 0:
        raise ValueError("Reference pool is empty; the matched attack needs disjoint training data.")
    # Full mimicry means references train on as much data as the target. Quietly
    # shrinking them would weaken the attack and make the audit read safer than
    # it is, so a short reference pool is an error, not a silent downgrade.
    if len(splits["ref_pool"]) < len(splits["target_train"]):
        raise ValueError(
            f"Reference pool ({len(splits['ref_pool'])}) is smaller than the target's training set "
            f"({len(splits['target_train'])}). Full mimicry needs at least as many reference samples; "
            "shrinking them silently would bias the audit optimistic."
        )

    train_kwargs = {"delta": delta, "max_physical_batch": max_physical_batch, "accountant": accountant}

    def train_fn(config: dict) -> Module:
        _free_cuda(device)
        return train_with_dpsgd(recipe, config, splits["target_train"], device, **train_kwargs)

    def utility_fn(model: Module) -> float:
        idx = splits["utility_eval"]
        return _evaluate_utility(model, utility_metric, splits["x"][idx], splits["y"][idx], device)

    def attack_fn(model: Module, config: dict) -> AttackScores:
        rng = np.random.default_rng(ref_seed)
        x, y = splits["x"], splits["y"]
        members, nonmembers = splits["audit_members"], splits["audit_nonmembers"]
        n_ref_train = len(splits["target_train"])  # guaranteed <= ref_pool by the check above

        # References are trained one at a time and freed after scoring (GPU memory).
        ref_phi_m = np.zeros(len(members))
        ref_phi_n = np.zeros(len(nonmembers))
        for _ in range(n_refs):
            sub = rng.choice(splits["ref_pool"], size=n_ref_train, replace=False)
            ref = train_with_dpsgd(recipe, config, sub, device)
            ref_phi_m += confidence_signal(ref, x[members], y[members], device, recipe.output_kind) / n_refs
            ref_phi_n += confidence_signal(ref, x[nonmembers], y[nonmembers], device, recipe.output_kind) / n_refs
            ref = train_with_dpsgd(recipe, config, sub, device, **train_kwargs)
            del ref
            _free_cuda(device)

        return AttackScores(
            member_scores=confidence_signal(model, x[members], y[members], device, recipe.output_kind) - ref_phi_m,
            nonmember_scores=(
                confidence_signal(model, x[nonmembers], y[nonmembers], device, recipe.output_kind) - ref_phi_n
            ),
        )

    return train_fn, utility_fn, attack_fn
