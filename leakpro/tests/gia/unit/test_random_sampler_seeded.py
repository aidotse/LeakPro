#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""RandomSampler's order is a function of (seed, epoch): victim and attacker see the same minibatches.

The same FedAvg protocol drives the victim (simulate_client) and every attacker iteration, so a shuffle
drawn from the global RNG gave each call a different order and an objective that kept moving.
"""
from __future__ import annotations

import torch
from torch import nn

from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.components.samplers import RandomSampler
from leakpro.attacks.gia_attacks.modular.core.protocol import FedAvg
from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo

CPU = torch.device("cpu")


def _order(sampler: RandomSampler, epoch: int, n: int = 12, bs: int = 4) -> list[list[int]]:
    return [b.tolist() for b in sampler.yield_batches(n, bs, epoch=epoch, device=CPU)]


def test_same_seed_and_epoch_give_the_same_order_regardless_of_global_rng() -> None:
    torch.manual_seed(0)
    first = _order(RandomSampler(seed=5), epoch=1)
    torch.manual_seed(1)
    assert _order(RandomSampler(seed=5), epoch=1) == first


def test_epochs_and_seeds_give_different_orders() -> None:
    sampler = RandomSampler(seed=5)
    assert _order(sampler, 0, n=20) != _order(sampler, 1, n=20)
    assert _order(RandomSampler(seed=5), 0, n=20) != _order(RandomSampler(seed=6), 0, n=20)


def test_sampling_leaves_the_global_rng_untouched() -> None:
    torch.manual_seed(0)
    expected = torch.rand(3)
    torch.manual_seed(0)
    _order(RandomSampler(), 0)
    torch.testing.assert_close(torch.rand(3), expected)


def test_fedavg_observation_is_identical_across_calls() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 4))
    x, y = torch.randn(8, 3, 2, 2), torch.arange(8) % 4
    protocol = FedAvg(local_epochs=2, batch_size=3, lr=0.1, sampler=RandomSampler(seed=3))
    a = protocol.observe(model, CrossEntropy(), x, TargetInfo(value=y), create_graph=False, has_epoch_prefix=False)
    b = protocol.observe(model, CrossEntropy(), x, TargetInfo(value=y), create_graph=False, has_epoch_prefix=False)
    for ga, gb in zip(a.gradients, b.gradients):
        torch.testing.assert_close(ga, gb, rtol=0, atol=0)
