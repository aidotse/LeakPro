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
