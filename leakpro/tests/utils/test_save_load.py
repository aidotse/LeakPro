#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for leakpro.utils.save_load.

Tests cover:
- hash_model(): byte-identical to the legacy ``.numpy().tobytes()`` path for float32 and
  int64 tensors (so no existing cached hash changes), including the 0-dim
  ``num_batches_tracked`` buffer that BatchNorm layers carry
- hash_model(): works for bfloat16 state dicts, which numpy cannot represent
- hash_model(): distinguishes models with different weights
- hash_config() / hash_indices() / hash_attack(): determinism and order handling
"""

import hashlib

import numpy as np
import pytest
import torch
from torch import nn

from leakpro.utils.save_load import hash_attack, hash_config, hash_indices, hash_model


def _legacy_hash_model(model: nn.Module) -> str:
    """Reference implementation: the exact byte path hash_model used before bf16 support."""
    hasher = hashlib.sha256()
    state_dict = model.state_dict()
    for key in sorted(state_dict.keys()):
        hasher.update(key.encode("utf-8"))
        hasher.update(state_dict[key].detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


@pytest.fixture
def linear_model() -> nn.Module:
    """A float32 model with weights and biases only."""
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))


@pytest.fixture
def batchnorm_model() -> nn.Module:
    """A model whose state dict carries a 0-dim int64 buffer (BatchNorm's num_batches_tracked)."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.BatchNorm2d(2))
    assert any(v.dim() == 0 for v in model.state_dict().values()), "fixture must contain a 0-dim buffer"
    return model


def test_hash_model_matches_legacy_bytes_for_linear(linear_model: nn.Module) -> None:
    """float32-only state dicts hash exactly as before."""
    assert hash_model(linear_model) == _legacy_hash_model(linear_model)


def test_hash_model_matches_legacy_bytes_for_batchnorm(batchnorm_model: nn.Module) -> None:
    """Mixed float32 / int64 / 0-dim state dicts hash exactly as before."""
    assert hash_model(batchnorm_model) == _legacy_hash_model(batchnorm_model)


def test_hash_model_handles_non_contiguous_tensors() -> None:
    """A transposed (non-contiguous) parameter hashes the same bytes as the legacy path."""
    model = nn.Linear(4, 3)
    with torch.no_grad():
        model.weight = nn.Parameter(torch.arange(12.0).reshape(4, 3).t())
    assert not model.weight.is_contiguous()
    assert hash_model(model) == _legacy_hash_model(model)


def test_hash_model_supports_bfloat16() -> None:
    """bfloat16 state dicts hash without error and deterministically."""
    torch.manual_seed(0)
    model = nn.Linear(4, 3).to(torch.bfloat16)
    with pytest.raises(TypeError):
        _legacy_hash_model(model)  # numpy has no bfloat16 — this is the case being fixed
    first = hash_model(model)
    assert len(first) == 64
    assert hash_model(model) == first


def test_hash_model_distinguishes_weights(linear_model: nn.Module) -> None:
    """Changing one weight changes the hash."""
    before = hash_model(linear_model)
    with torch.no_grad():
        linear_model[0].weight[0, 0] += 1.0
    assert hash_model(linear_model) != before


def test_hash_config_is_key_order_independent() -> None:
    """hash_config canonicalises key order."""
    assert hash_config({"a": 1, "b": [1, 2]}) == hash_config({"b": [1, 2], "a": 1})
    assert hash_config({"a": 1}) != hash_config({"a": 2})


def test_hash_indices_is_order_independent() -> None:
    """Same split in a different order hashes the same; a different split differs."""
    train, test = np.array([3, 1, 2]), np.array([5, 4])
    assert hash_indices(train, test) == hash_indices(train[::-1], test[::-1])
    assert hash_indices(train, test) != hash_indices(test, train)


def test_hash_attack_depends_on_config_and_model(linear_model: nn.Module) -> None:
    """hash_attack changes when either the config or the model changes."""
    base = hash_attack({"x": 1}, linear_model)
    assert hash_attack({"x": 2}, linear_model) != base
    with torch.no_grad():
        linear_model[0].bias += 1.0
    assert hash_attack({"x": 1}, linear_model) != base
