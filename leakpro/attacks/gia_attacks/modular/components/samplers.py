#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Minibatch samplers that yield index tensors for each epoch."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor

from leakpro.attacks.gia_attacks.modular.core.interfaces import BatchSampler


class SequentialSampler(BatchSampler):
    """Yields minibatches in sequential order (0, 1, ..., N-1)."""

    def yield_batches(
        self, num_samples: int, batch_size: int, epoch: int, device: torch.device  # noqa: ARG002 -- interface signature
    ) -> Iterator[Tensor]:
        """Yields consecutive index slices in dataset order."""
        indices = torch.arange(num_samples, device=device)
        for start in range(0, num_samples, batch_size):
            yield indices[start : start + batch_size]


@dataclass(frozen=True)
class RandomSampler(BatchSampler):
    """Yields minibatches in a shuffled order that is a fixed function of ``(seed, epoch)``.

    Each epoch draws its permutation from a private ``torch.Generator`` seeded with ``seed`` and the epoch
    index, so the global RNG is neither read nor advanced. The same protocol instance drives the victim
    client and every attacker iteration, so both train on the same minibatches: this models an attacker
    who knows the client's shuffling seed. An attacker who does not know it is modelled by giving the
    attack's protocol a different seed from the one the client was simulated with.

    Attributes:
        seed: Base seed of the per-epoch permutations.

    """

    seed: int = 0

    def yield_batches(
        self, num_samples: int, batch_size: int, epoch: int, device: torch.device
    ) -> Iterator[Tensor]:
        """Yields index slices of the permutation for ``epoch``."""
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + epoch)
        indices = torch.randperm(num_samples, generator=generator).to(device)
        for start in range(0, num_samples, batch_size):
            yield indices[start : start + batch_size]
