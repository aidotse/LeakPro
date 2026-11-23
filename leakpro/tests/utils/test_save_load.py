#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for leakpro.utils.save_load.

Tests cover:
- hash_model(): matches legacy NumPy hashes for linear and BatchNorm models,
  transposed parameters, and scalar and transposed buffers across dtypes
- hash_model(): hashes BF16 weights and scalar buffers deterministically and detects changes to either
- hash_model(): distinguishes models with different weights
- hash_config() / hash_indices() / hash_attack(): determinism and order handling
- unwrapped_state_dict(): strips only leading wrapper prefixes, keeps inner names
"""

import hashlib

import numpy as np
import pytest
import torch
from torch import nn

from leakpro.utils.save_load import (
    fingerprint_model,
    hash_attack,
    hash_config,
    hash_indices,
    hash_model,
    unwrapped_state_dict,
)


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


def test_hash_model_preserves_numpy_supported_digests(
    linear_model: nn.Module, batchnorm_model: nn.Module,
) -> None:
    transposed_model = nn.Linear(4, 3)
    transposed_model.weight = nn.Parameter(torch.arange(12.0).reshape(4, 3).t())
    assert not transposed_model.weight.is_contiguous()
    for model in (linear_model, batchnorm_model, transposed_model):
        assert hash_model(model) == _legacy_hash_model(model), repr(model)

    dtypes = (torch.float16, torch.float32, torch.float64, torch.int64, torch.int32, torch.uint8, torch.bool)
    for dtype in dtypes:
        transposed = torch.arange(6).reshape(2, 3).T.to(dtype)
        assert not transposed.is_contiguous()
        for values in (torch.tensor(1, dtype=dtype), transposed):
            model = nn.Module()
            model.register_buffer("values", values)
            assert hash_model(model) == _legacy_hash_model(model), (dtype, values.shape)


def test_hash_model_distinguishes_weights(linear_model: nn.Module) -> None:
    """Changing one weight changes the hash."""
    before = hash_model(linear_model)
    with torch.no_grad():
        linear_model[0].weight[0, 0] += 1.0
    assert hash_model(linear_model) != before


def test_fingerprint_model_is_cheap_deterministic_and_discriminating(linear_model: nn.Module) -> None:
    """Fingerprint changes with weights, shapes and the pretrained name, but reads only a sample of each tensor."""
    base = fingerprint_model(linear_model)
    assert len(base) == 64
    assert fingerprint_model(linear_model) == base
    with torch.no_grad():
        linear_model[0].weight[0, 0] += 1.0                    # inside the sampled prefix
    assert fingerprint_model(linear_model) != base
    linear_model.pretrained_name_or_path = "gpt2"
    named = fingerprint_model(linear_model)
    linear_model.pretrained_name_or_path = "gpt2-medium"
    assert fingerprint_model(linear_model) != named

    big = nn.Linear(4096, 4096)                                # 16M params: sampled, not read in full
    with torch.no_grad():
        before = fingerprint_model(big)
        big.weight[2048, 2048] += 1.0                          # middle of the tensor, outside the sample
    assert fingerprint_model(big) == before                    # documents the deliberate trade-off
    assert fingerprint_model(nn.Linear(4, 3).to(torch.bfloat16))  # bf16 works


def test_hash_attack_accepts_alternative_model_hasher(linear_model: nn.Module) -> None:
    """hash_attack composes the config hash with whichever model hasher is passed."""
    full = hash_attack({"x": 1}, linear_model)
    cheap = hash_attack({"x": 1}, linear_model, model_hasher=fingerprint_model)
    assert full != cheap
    assert cheap == hash_attack({"x": 1}, linear_model, model_hasher=fingerprint_model)


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


class _Encoder(nn.Module):
    """Submodule names that end in "module": a substring replace mangled them."""

    def __init__(self) -> None:
        super().__init__()
        self.submodule = nn.Linear(4, 4)
        self.attn_module = nn.Linear(4, 2)


class _Wrapper(nn.Module):
    """Stands in for a wrapper that registers the model under a given attribute name."""

    def __init__(self, inner: nn.Module, name: str) -> None:
        super().__init__()
        setattr(self, name, inner)


def test_inner_names_ending_in_module_are_kept() -> None:
    """A submodule whose name ends in "module" keeps its key; only leading wrappers are stripped."""
    # "encoder.submodule.weight".replace("module.", "") gave "encoder.subweight".
    model = nn.Sequential()
    model.add_module("encoder", _Encoder())
    keys = set(unwrapped_state_dict(model))
    assert keys == set(model.state_dict())
    assert "encoder.submodule.weight" in keys
    assert "encoder.attn_module.bias" in keys


def test_wrapper_prefixes_are_stripped_and_weights_load() -> None:
    """Opacus, DataParallel and nested wrappers unwrap to keys that load strictly."""
    inner = _Encoder()
    for wrapped in (_Wrapper(inner, "_module"),                        # Opacus GradSampleModule
                    _Wrapper(inner, "module"),                         # DataParallel / DDP
                    _Wrapper(_Wrapper(inner, "module"), "_module")):   # nested
        state_dict = unwrapped_state_dict(wrapped)
        assert set(state_dict) == set(inner.state_dict())
        fresh = _Encoder()
        fresh.load_state_dict(state_dict)  # strict: fails on any mangled or leftover key
        assert torch.equal(fresh.submodule.weight, inner.submodule.weight)


def test_bfloat16_weights_and_buffers_are_hashed_without_numpy_dtype_support() -> None:
    model = nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    model.register_buffer("scale", torch.tensor(1.0, dtype=torch.bfloat16))
    with pytest.raises(TypeError):
        _legacy_hash_model(model)
    first = hash_model(model)
    assert len(first) == 64
    assert hash_model(model) == first

    attack_id = hash_attack({"attack": "test"}, model)
    with torch.no_grad():
        model.weight.add_(1.0)
    after_weight_change = hash_model(model)
    assert after_weight_change != first
    assert hash_attack({"attack": "test"}, model) != attack_id
    model.scale.add_(1.0)
    assert hash_model(model) != after_weight_change
