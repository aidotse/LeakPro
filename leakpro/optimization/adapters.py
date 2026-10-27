#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Build a ``PETRecipe`` from an architecture file and a dataset.

:func:`leakpro.optimization.train_with_dpsgd` wants a
:class:`~leakpro.optimization.training.PETRecipe`: factories for the model, the
optimizer and the training loader, plus the loss and the epoch count. Writing
those by hand is fine in a script; a caller that only has "a .py file defining an
``nn.Module``, a dataset, and the settings the model was trained with" needs an
adapter. That is this module.

The loader factory wraps the dataset in :class:`torch.utils.data.Subset`, so the
model trains on exactly what ``dataset.__getitem__`` yields. Note that
:func:`leakpro.optimization.objective.make_dpsgd_objective` does not use that
factory: it trains on rows of the materialised audit population, because the
audit labels population rows as members and a loader over any other dataset
would train one set of records while another is labelled as seen. Pass that
population here when the recipe feeds the objective; the factory matters for
:func:`leakpro.optimization.train_with_dpsgd`.
"""

import inspect
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from leakpro.input_handler.user_imports import (
    get_class_from_module,
    get_optimizer_mapping,
    import_module_from_file,
)
from leakpro.optimization.training import PETRecipe
from leakpro.utils.logger import logger


def model_classes_defined_in(module) -> list[type]:  # noqa: ANN001
    """The ``nn.Module`` subclasses *defined* in ``module``, in definition order.

    Imported classes are excluded on purpose: an ``from torchvision.models.resnet
    import BasicBlock`` at the top of an arch file is not the user's model, and
    picking it was issue #484.
    """
    return [obj for obj in vars(module).values()
            if isinstance(obj, type) and issubclass(obj, nn.Module) and obj.__module__ == module.__name__]


def find_model_class(module, model_class: str | None = None, init_params: dict | None = None) -> type:  # noqa: ANN001
    """The one rule for which class in an architecture file is the model.

    In order: a name, when the caller has one (``user_imports.get_class_from_module``);
    otherwise, among the classes defined in the file, the first whose constructor
    accepts every recorded ``init_params`` key (so a helper block defined before
    the network does not shadow it); otherwise the only candidate; otherwise the
    first one defined, with a warning naming the choice. Shared by the webapp's
    training step, audit worker and PET runner, so the model that was trained,
    the one that is audited and the one that is retrained are the same class.
    """
    if model_class:
        return get_class_from_module(module, model_class)
    candidates = model_classes_defined_in(module)
    if not candidates:
        raise ValueError(f"{module.__name__} defines no nn.Module subclass.")
    if init_params:
        keys = set(init_params)
        for cls in candidates:
            if keys <= (set(inspect.signature(cls.__init__).parameters) - {"self"}):
                return cls
    if len(candidates) > 1:
        logger.warning(f"{module.__name__} defines {[c.__name__ for c in candidates]}; "
                       f"using the first, {candidates[0].__name__}. Name model_class to choose.")
    return candidates[0]


def detect_binary(
    module_path: str | Path,
    sample: torch.Tensor,
    model_class: str | None = None,
    init_params: dict | None = None,
) -> bool:
    """Report whether this architecture emits a single logit per sample.

    The loss, the label dtype and the way the attack reads confidences all
    follow from this, and it cannot be inferred from the class count alone: a
    two-class problem may be modelled with one sigmoid output or two softmax
    ones. So instantiate the model once and look at what it produces.

    Args:
        module_path: a .py file defining the architecture.
        sample: one or more inputs as the model expects them (batch axis first).
        model_class: which nn.Module to use; inferred when the file defines one.
        init_params: keyword arguments for the model constructor.

    """
    cls = find_model_class(import_module_from_file(str(module_path)), model_class, init_params)
    probe = cls(**dict(init_params or {}))
    probe.eval()
    with torch.no_grad():
        out = probe(sample[:2].float())
    return out.ndim == 1 or out.shape[-1] == 1


def recipe_from_module(  # noqa: PLR0913
    module_path: str | Path,
    dataset: Dataset,
    epochs: int,
    model_class: str | None = None,
    init_params: dict | None = None,
    optimizer_name: str = "adam",
    binary: bool = False,
) -> PETRecipe:
    """Assemble a recipe from an architecture file and a dataset.

    Args:
        module_path: a .py file defining the architecture.
        dataset: what ``make_loader`` indexes. For the DP-SGD objective this is
            the materialised population (its ``train_indices`` are positions in
            it); the objective itself trains on the population directly.
        epochs: training epochs, held fixed across the search.
        model_class: which nn.Module to use; inferred when the file defines one.
        init_params: keyword arguments for the model constructor. Pass the same
            ones LeakPro stores in the target's metadata, so the RMIA shadow
            models (rebuilt from that metadata) share the architecture.
        optimizer_name: any ``torch.optim`` optimizer name; the learning rate is
            a searched knob.
        binary: single-logit head trained with ``BCEWithLogitsLoss`` rather than
            multiclass cross-entropy.

    """
    # torch.optim is the source of truth for what optimizers exist, via the
    # repo's mapping — a hardcoded dict here would silently exclude the rest.
    optimizers = get_optimizer_mapping()
    if optimizer_name not in optimizers:
        raise ValueError(f"Unknown optimizer '{optimizer_name}'; use one of {sorted(optimizers)}.")

    module = import_module_from_file(str(module_path))
    cls = find_model_class(module, model_class, init_params)
    kwargs = dict(init_params or {})
    optimizer_cls = optimizers[optimizer_name]

    # A fresh model per config: the search must never resume a trained one.
    def make_model(_config: dict) -> nn.Module:
        return cls(**kwargs)

    def make_optimizer(params, config: dict):  # noqa: ANN001, ANN202
        return optimizer_cls(params, lr=config["learning_rate"])

    def make_loader(indices: np.ndarray, config: dict) -> DataLoader:
        return DataLoader(
            Subset(dataset, [int(i) for i in indices]),
            batch_size=int(config["batch_size"]),
            shuffle=True,
        )

    logger.info(f"Recipe built from {Path(module_path).name}: {cls.__name__}({kwargs}), {optimizer_name}, {epochs} epochs.")
    # A single-logit head still emits raw logits (BCEWithLogitsLoss applies the
    # sigmoid itself); "binary_probs" is for models whose output already IS a
    # probability. The criterion, not output_kind, is what marks the head binary.
    return PETRecipe(
        make_model=make_model,
        make_optimizer=make_optimizer,
        make_loader=make_loader,
        criterion=nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss(),
        epochs=epochs,
        output_kind="logits",
    )
