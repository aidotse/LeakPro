"""Byte hashing compatibility across supported tensor types and layouts."""

import hashlib

import pytest
import torch
from torch import nn

from leakpro.utils.save_load import hash_attack, hash_model


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64, torch.int64, torch.int32, torch.uint8, torch.bool])
@pytest.mark.parametrize("scalar", [False, True])
def test_hash_model_preserves_numpy_supported_digests(dtype, scalar) -> None:
    model = nn.Module()
    values = torch.tensor(1, dtype=dtype) if scalar else torch.arange(6).reshape(2, 3).T.to(dtype)
    model.register_buffer("values", values)
    expected = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        expected.update(key.encode("utf-8"))
        expected.update(value.detach().cpu().numpy().tobytes())
    assert hash_model(model) == expected.hexdigest()


def test_bfloat16_weights_and_buffers_are_hashed_without_numpy_dtype_support() -> None:
    model = nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    model.register_buffer("scale", torch.tensor(1.0, dtype=torch.bfloat16))
    first = hash_model(model)
    attack_id = hash_attack({"attack": "test"}, model)
    with torch.no_grad():
        model.weight.add_(1.0)
    after_weight_change = hash_model(model)
    assert after_weight_change != first
    assert hash_attack({"attack": "test"}, model) != attack_id
    model.scale.add_(1.0)
    assert hash_model(model) != after_weight_change
