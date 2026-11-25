#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Module containing helper functions for handling input files and modules."""

import importlib.util
import inspect
import os
import sys

from torch import nn, optim

from leakpro.utils.import_helper import Callable, ModuleType
from leakpro.utils.logger import logger


def import_module_from_file(filepath: str) -> ModuleType:
    """Import a module from a given file path."""
    filepath = os.path.abspath(filepath)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found")

    module_name = os.path.splitext(os.path.basename(filepath))[0]

    existing = sys.modules.get(module_name)
    if existing is not None and os.path.abspath(getattr(existing, "__file__", "")) == filepath:
        return existing

    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {filepath}")

    module = importlib.util.module_from_spec(spec)

    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is not None:
            sys.modules[module_name] = previous
        else:
            sys.modules.pop(module_name, None)
        raise

    return module



def get_class_from_module(module:ModuleType, class_name:str) -> Callable:
    """Get the specified class from a module."""
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if name == class_name:
            return obj
    raise ValueError(f"Class {class_name} not found in module {module.__name__}")

def load_model_blueprint(module_path: str, model_class: str, role: str = "target") -> Callable:
    """The model class ``model_class`` from the file ``module_path`` (``target.module_path`` / ``model_class``).

    Shared by the input handlers, so every attack type resolves a model class the same way.
    """
    if model_class is None:
        raise ValueError(f"{role} model_class not found in configs.")
    if module_path is None:
        raise ValueError(f"{role} module_path not found in configs.")
    try:
        blueprint = get_class_from_module(import_module_from_file(module_path), model_class)
    except Exception as e:
        raise ValueError(f"Failed to create the {role} model blueprint from {model_class} in {module_path}") from e
    logger.info(f"{role.capitalize()} model blueprint created from {model_class} in {module_path}.")
    return blueprint


def get_optimizer_mapping() -> dict:
    """Return a dictionary mapping optimizer names to optimizer classes."""
    optimizer_mapping = {
            attr.lower(): getattr(optim, attr)
            for attr in dir(optim)
            if isinstance(getattr(optim, attr), type) and issubclass(getattr(optim, attr), optim.Optimizer)
        }
    return optimizer_mapping  # noqa: RET504

def get_criterion_mapping() -> dict:
    """Return a dictionary mapping loss names to loss classes."""
    loss_mapping = {}

    for attr in dir(nn):
        # Get the attribute
        attribute = getattr(nn, attr, None)
        # Ensure it's a class and a subclass of _Loss
        if isinstance(attribute, type) and issubclass(attribute, nn.modules.loss._Loss):
            loss_mapping[attr.lower()] = attribute
    return loss_mapping
