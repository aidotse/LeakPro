#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for leakpro.utils.save_load."""

import torch
from torch import nn

from leakpro.utils.save_load import unwrapped_state_dict


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


def test_inner_names_ending_in_module_are_kept():
    # "encoder.submodule.weight".replace("module.", "") gave "encoder.subweight".
    model = nn.Sequential()
    model.add_module("encoder", _Encoder())
    keys = set(unwrapped_state_dict(model))
    assert keys == set(model.state_dict())
    assert "encoder.submodule.weight" in keys
    assert "encoder.attn_module.bias" in keys


def test_wrapper_prefixes_are_stripped_and_weights_load():
    inner = _Encoder()
    for wrapped in (_Wrapper(inner, "_module"),                        # Opacus GradSampleModule
                    _Wrapper(inner, "module"),                         # DataParallel / DDP
                    _Wrapper(_Wrapper(inner, "module"), "_module")):   # nested
        state_dict = unwrapped_state_dict(wrapped)
        assert set(state_dict) == set(inner.state_dict())
        fresh = _Encoder()
        fresh.load_state_dict(state_dict)  # strict: fails on any mangled or leftover key
        assert torch.equal(fresh.submodule.weight, inner.submodule.weight)
