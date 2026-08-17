#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Build a ``PETRecipe`` and its data splits from an architecture file.

The campaign's contract is a set of factories plus five disjoint index sets
(``leakpro.optimization.training``). Writing those by hand is fine in a script;
a caller that only has "a module defining an nn.Module, a tensor dataset, and
the settings the model was trained with" needs an adapter. That is this module.

It deliberately does *not* go through ``AbstractInputHandler``: a user handler
exists to supply a custom ``train()``, which is exactly what the shared DP-SGD
loop replaces. Anything whose training loop cannot be expressed as
(model, optimizer, loader, criterion, epochs) should build the campaign
callables directly instead.
"""

import importlib.util
import sys
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from leakpro.optimization.training import PETRecipe
from leakpro.utils.logger import logger

OPTIMIZERS: dict[str, Callable] = {"adam": optim.Adam, "sgd": optim.SGD, "adamw": optim.AdamW}


def load_module(module_path: str | Path):  # noqa: ANN201
    """Import a .py file under a unique name, so repeated loads never collide."""
    module_path = Path(module_path)
    if not module_path.exists():
        raise FileNotFoundError(f"Architecture module not found: {module_path}")
    name = f"leakpro_arch_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load a module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def find_model_class(module, model_class: str | None = None) -> type:  # noqa: ANN001
    """Pick the nn.Module subclass to train, by name or as the only candidate."""
    candidates = {
        name: obj for name, obj in vars(module).items()
        if isinstance(obj, type) and issubclass(obj, nn.Module) and obj.__module__ == module.__name__
    }
    if model_class:
        if model_class not in candidates:
            raise KeyError(f"'{model_class}' is not an nn.Module defined in {module.__name__}; found {sorted(candidates)}.")
        return candidates[model_class]
    if len(candidates) != 1:
        raise ValueError(
            f"Cannot choose a model class automatically: {sorted(candidates)}. Pass model_class explicitly."
        )
    return next(iter(candidates.values()))


def detect_binary(
    module_path: str | Path,
    x: torch.Tensor,
    model_class: str | None = None,
    init_params: dict | None = None,
) -> bool:
    """Report whether this architecture emits a single logit per sample.

    The loss, the label dtype and the way the attack reads confidences all
    follow from this, and it cannot be inferred from the class count alone: a
    two-class problem may be modelled with one sigmoid output or two softmax
    ones. So instantiate the model once and look at what it produces.
    """
    cls = find_model_class(load_module(module_path), model_class)
    probe = cls(**dict(init_params or {}))
    probe.eval()
    with torch.no_grad():
        out = probe(x[:2].float())
    return out.ndim == 1 or out.shape[-1] == 1


def recipe_from_module(  # noqa: PLR0913
    module_path: str | Path,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    model_class: str | None = None,
    init_params: dict | None = None,
    optimizer_name: str = "adam",
    binary: bool = False,
) -> PETRecipe:
    """Assemble a recipe from an architecture file and an in-memory tensor dataset.

    Args:
        module_path: a .py file defining the architecture.
        x: full feature tensor; loaders index into it.
        y: full label tensor.
        epochs: training epochs, held fixed across the campaign.
        model_class: which nn.Module to use; inferred when the file defines one.
        init_params: keyword arguments for the model constructor.
        optimizer_name: "adam", "adamw" or "sgd"; the learning rate is a searched knob.
        binary: single-logit head trained with BCEWithLogitsLoss rather than
            multiclass cross-entropy. Sets ``output_kind`` to match, which is
            what the attack reads confidences with.

    """
    if optimizer_name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer '{optimizer_name}'; use one of {sorted(OPTIMIZERS)}.")

    module = load_module(module_path)
    cls = find_model_class(module, model_class)
    kwargs = dict(init_params or {})
    optimizer_cls = OPTIMIZERS[optimizer_name]

    # A fresh model per config: the campaign must never resume a trained one.
    def make_model(_config: dict) -> nn.Module:
        return cls(**kwargs)

    def make_optimizer(params, config: dict):  # noqa: ANN001, ANN202
        return optimizer_cls(params, lr=config["learning_rate"])

    def make_loader(indices: np.ndarray, config: dict) -> DataLoader:
        return DataLoader(
            TensorDataset(x[indices], y[indices]),
            batch_size=int(config["batch_size"]),
            shuffle=True,
        )

    logger.info(f"Recipe built from {Path(module_path).name}: {cls.__name__}({kwargs}), {optimizer_name}, {epochs} epochs.")
    return PETRecipe(
        make_model=make_model,
        make_optimizer=make_optimizer,
        make_loader=make_loader,
        criterion=nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss(),
        epochs=epochs,
        output_kind="binary_logits" if binary else "logits",
    )


def carve_splits(  # noqa: PLR0913
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int = 0,
    target_fraction: float = 0.4,
    ref_fraction: float = 0.4,
    audit_size: int = 2000,
) -> dict:
    """Carve the five disjoint roles a matched-reference campaign needs.

    The reference pool is the expensive requirement: it must be disjoint from
    the target's training set, so it comes out of the same dataset and the
    target trains on less data than it otherwise would. Callers should surface
    that, because it means the frontier describes a smaller-data model than a
    plain audit of the user's own model does.

    Members are drawn from the target's training set; nonmembers and the
    utility split come from what is left, and never overlap each other.
    """
    if not 0 < target_fraction < 1 or not 0 < ref_fraction < 1:
        raise ValueError("target_fraction and ref_fraction must each be in (0, 1).")
    if target_fraction + ref_fraction >= 1:
        raise ValueError(
            f"target_fraction + ref_fraction = {target_fraction + ref_fraction:.2f} leaves nothing "
            "for the audit and utility splits; they must sum to less than 1."
        )

    n = len(x)
    order = np.random.default_rng(seed).permutation(n)
    n_target = int(n * target_fraction)
    n_ref = int(n * ref_fraction)

    target_train = order[:n_target]
    ref_pool = order[n_target:n_target + n_ref]
    rest = order[n_target + n_ref:]

    # Nonmembers and the utility split share `rest`, so the audit set cannot
    # grow past half of it without the two overlapping.
    n_audit = min(audit_size, len(target_train), len(rest) // 2)

    # Every role has to be non-empty for the campaign to mean anything: no
    # reference pool means no matched attack, no nonmembers means no attack
    # metric, no utility split means no quality axis.
    empty = [
        name for name, size in (
            ("target_train", len(target_train)),
            ("ref_pool", len(ref_pool)),
            ("audit set", n_audit),
            ("utility_eval", len(rest) - n_audit),
        ) if size == 0
    ]
    if empty:
        raise ValueError(
            f"Dataset of {n} samples is too small: {', '.join(empty)} would be empty at "
            f"target_fraction={target_fraction}, ref_fraction={ref_fraction}."
        )
    if n_audit < audit_size:
        logger.warning(f"Audit set capped at {n_audit} (requested {audit_size}) by the available disjoint data.")

    return {
        "x": x,
        "y": y,
        "target_train": target_train,
        "ref_pool": ref_pool,
        "audit_members": target_train[:n_audit],
        "audit_nonmembers": rest[:n_audit],
        "utility_eval": rest[n_audit:],
    }
