#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""BatchSampler contract: one epoch yields every index exactly once, in batches of at most the batch size."""
from __future__ import annotations

import pytest
import torch

from leakpro.attacks.gia_attacks.modular.core.interfaces import BatchSampler
from leakpro.tests.gia import helpers
from leakpro.tests.gia.contract.registry import build, cases

pytestmark = pytest.mark.parametrize(("cls", "factory"), cases(BatchSampler))


@pytest.mark.parametrize(("num_samples", "batch_size"), [(7, 3), (4, 4), (1, 2)])
def test_an_epoch_yields_every_index_once(cls, factory, num_samples, batch_size) -> None:  # noqa: ANN001
    sampler = build(cls, factory).component
    batches = list(sampler.yield_batches(num_samples, batch_size, epoch=0, device=helpers.DEVICE))
    assert all(b.dim() == 1 and 0 < b.numel() <= batch_size for b in batches)
    assert sorted(torch.cat(batches).tolist()) == list(range(num_samples))
