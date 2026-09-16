#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions for loading and storing."""

import hashlib
import json
from typing import Callable

import numpy as np
import torch
from torch.nn import Module


def hash_config(config: dict) -> str:
    """Generate a SHA-256 hash of a dictionary."""

    # Convert the config to a canonical JSON string (sorted keys ensures consistency)
    config_str = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(config_str.encode("utf-8")).hexdigest()

def hash_model(model: Module) -> str:
    """Generate a SHA-256 hash of a PyTorch model's state dictionary.

    This function takes into account the model weights by iterating
    over the state dictionary (which includes both parameters and buffers)
    and updating the hash with both the key names and the corresponding tensor values.
    """
    hasher = hashlib.sha256()
    state_dict = model.state_dict()

    # Sort keys to ensure consistent ordering across models
    for key in sorted(state_dict.keys()):
        hasher.update(key.encode("utf-8"))
        # Hash the raw bytes of the tensor. Flatten to 1-D and reinterpret as uint8 so the
        # path also works for dtypes numpy cannot represent (bfloat16). For dtypes numpy does
        # support this yields exactly the same bytes as ``.numpy().tobytes()`` did, so the hash
        # of every existing checkpoint is unchanged. ``reshape(-1)`` is required: ``view`` with
        # a different element size refuses 0-dim tensors such as BatchNorm's ``num_batches_tracked``.
        tensor = state_dict[key].detach().cpu().contiguous().reshape(-1)
        tensor_bytes = tensor.view(torch.uint8).numpy().tobytes()
        hasher.update(tensor_bytes)

    return hasher.hexdigest()

def fingerprint_model(model: Module, sample_elements: int = 1024) -> str:
    """Generate a cheap SHA-256 fingerprint of a model without reading every weight.

    Hashes, for every state-dict entry in sorted order, its name, shape and dtype plus the first and
    last ``sample_elements`` values, and the model's ``pretrained_name_or_path`` attribute when it has
    one. Cost is O(number of tensors), not O(number of parameters), so it is usable as an attack id
    component for multi-billion-parameter models where :func:`hash_model` would read tens of GB.
    Fine-tuning perturbs every weight, so the sampled values still separate checkpoints of one
    architecture; this is *not* a collision-resistant hash of the full weights and must not be used
    where :func:`hash_model` is (shadow-model cache validity).
    """
    hasher = hashlib.sha256()
    hasher.update(str(getattr(model, "pretrained_name_or_path", "")).encode("utf-8"))
    state_dict = model.state_dict()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        hasher.update(f"{key}|{tuple(tensor.shape)}|{tensor.dtype}".encode("utf-8"))
        flat = tensor.detach().cpu().contiguous().reshape(-1)
        if flat.numel() > 2 * sample_elements:
            flat = torch.cat([flat[:sample_elements], flat[-sample_elements:]])
        hasher.update(flat.view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def hash_indices(train_indices: np.ndarray, test_indices: np.ndarray) -> str:
    """Generate a SHA-256 hash of the train/test index split.

    Order-independent: the same set of indices produces the same hash regardless
    of ordering, so splits can be compared across runs that may shuffle differently.
    """
    hasher = hashlib.sha256()
    hasher.update(np.sort(np.asarray(train_indices)).tobytes())
    hasher.update(b"|")
    hasher.update(np.sort(np.asarray(test_indices)).tobytes())
    return hasher.hexdigest()

def hash_attack(config:dict, target_model:Module, model_hasher: Callable[[Module], str] = hash_model) -> str:
    """Generate a SHA-256 hash of the attack configuration and target model.

    This function combines the hash of the attack configuration and the hash of the model
    to create a unique identifier for the attack. ``model_hasher`` defaults to the full-weight
    :func:`hash_model`; attacks on very large models may pass :func:`fingerprint_model`.
    """
    config_hash = hash_config(config)
    model_hash = model_hasher(target_model)
    combined_hash = hashlib.sha256((config_hash + model_hash).encode("utf-8")).hexdigest()
    return combined_hash  # noqa: RET504
