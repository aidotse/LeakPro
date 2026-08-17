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

import numpy as np
import torch
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
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


def train_with_dpsgd(  # noqa: PLR0913
    recipe: PETRecipe,
    config: dict,
    train_indices: np.ndarray,
    device: str,
    delta: float = 1e-5,
    max_physical_batch: int = 256,
) -> Module:
    """Train one model under the sampled config; the single shared Opacus path.

    ``noise_multiplier > 0`` wraps the loop in a PrivacyEngine with the physical
    batch capped at ``max_physical_batch`` (per-example gradients cost
    batch x params memory; the sampled batch size and the accounting are
    unchanged). ``noise_multiplier == 0`` runs the plain loop — the non-private
    anchor, ε = ∞.

    The formal (ε, δ) is stored on the returned model as ``campaign_extras``,
    which ``Campaign`` records next to the empirical attack result.
    """
    loader = recipe.make_loader(train_indices, config)
    model = recipe.make_model(config).to(device)
    optimizer = recipe.make_optimizer(model.parameters(), config)
    private = config.get("noise_multiplier", 0) > 0

    def run_epochs(epoch_loader) -> None:  # noqa: ANN001
        for _ in range(recipe.epochs):
            for xb, yb in epoch_loader:
                optimizer.zero_grad()
                loss = recipe.criterion(model(xb.to(device)), yb.to(device))
                loss.backward()
                optimizer.step()

    model.train()
    if private:
        engine = PrivacyEngine(accountant="rdp")
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
    model.campaign_extras = {"epsilon": epsilon, "delta": delta}
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
            pred = model(x[i:i + 1024].to(device)).argmax(dim=1).cpu()
            correct += int((pred == y[i:i + 1024]).sum())
        return correct / len(x)
    if metric == "auc":
        from sklearn.metrics import roc_auc_score
        scores = [model(x[i:i + 4096].to(device)).cpu().reshape(-1) for i in range(0, len(x), 4096)]
        return float(roc_auc_score(y.numpy().ravel(), torch.cat(scores).numpy()))
    raise ValueError(f"Unknown utility_metric '{metric}'; use 'accuracy', 'auc' or a callable.")


def build_campaign_fns(  # noqa: PLR0913
    recipe: PETRecipe,
    splits: dict,
    n_refs: int,
    device: str,
    utility_metric: str | Callable = "accuracy",
    ref_seed: int = 1,
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

    """
    missing = [k for k in REQUIRED_SPLIT_KEYS if k not in splits]
    if missing:
        raise KeyError(f"splits is missing {missing}; required keys are {REQUIRED_SPLIT_KEYS}.")
    if len(splits["ref_pool"]) == 0:
        raise ValueError("Reference pool is empty; the matched attack needs disjoint training data.")

    def train_fn(config: dict) -> Module:
        _free_cuda(device)
        return train_with_dpsgd(recipe, config, splits["target_train"], device)

    def utility_fn(model: Module) -> float:
        idx = splits["utility_eval"]
        return _evaluate_utility(model, utility_metric, splits["x"][idx], splits["y"][idx], device)

    def attack_fn(model: Module, config: dict) -> AttackScores:
        rng = np.random.default_rng(ref_seed)
        x, y = splits["x"], splits["y"]
        members, nonmembers = splits["audit_members"], splits["audit_nonmembers"]
        n_ref_train = min(len(splits["target_train"]), len(splits["ref_pool"]))

        # References are trained one at a time and freed after scoring (GPU memory).
        ref_phi_m = np.zeros(len(members))
        ref_phi_n = np.zeros(len(nonmembers))
        for _ in range(n_refs):
            sub = rng.choice(splits["ref_pool"], size=n_ref_train, replace=False)
            ref = train_with_dpsgd(recipe, config, sub, device)
            ref_phi_m += confidence_signal(ref, x[members], y[members], device, recipe.output_kind) / n_refs
            ref_phi_n += confidence_signal(ref, x[nonmembers], y[nonmembers], device, recipe.output_kind) / n_refs
            del ref
            _free_cuda(device)

        return AttackScores(
            member_scores=confidence_signal(model, x[members], y[members], device, recipe.output_kind) - ref_phi_m,
            nonmember_scores=(
                confidence_signal(model, x[nonmembers], y[nonmembers], device, recipe.output_kind) - ref_phi_n
            ),
        )

    return train_fn, utility_fn, attack_fn
