#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""A seed axis works with a GAN representation, whose generator builds one decode graph shared by every seed slice."""
from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from leakpro.attacks.gia_attacks.modular.components.representations import GANRepresentation
from leakpro.tests.gia.unit.test_multi_seed_shared_decode import _run


class _Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 3 * 8 * 8)

    def forward(self, z: Tensor) -> Tensor:
        return torch.tanh(self.lin(z)).view(-1, 3, 8, 8)


@pytest.mark.parametrize("rep_fn", [lambda: GANRepresentation(_Generator(), 4)], ids=["gan"])
def test_two_seeds_run_through_a_shared_decode_graph(rep_fn) -> None:  # noqa: ANN001
    state, _ = _run(rep_fn())
    assert state.input.shape == (2, 3, 8, 8)
    assert torch.isfinite(state.input).all()
