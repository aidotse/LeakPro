#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for SequentialSampler and RandomSampler.

Each sampler is exercised directly — no driver, no run_attack.
"""

import torch


from leakpro.attacks.gia_attacks.modular.components.samplers import RandomSampler, SequentialSampler

_CPU = torch.device("cpu")


def _collect(sampler, num_samples: int, batch_size: int) -> list[list[int]]:
    """Run one epoch and return index batches as plain Python lists."""
    return [b.tolist() for b in sampler.yield_batches(num_samples, batch_size, epoch=0, device=_CPU)]


# ── SequentialSampler ──────────────────────────────────────────────────────────

class TestSequentialSampler:
    def setup_method(self):
        self.s = SequentialSampler()

    # --- exact division ---

    def test_exact_division_chunk_count(self):
        """8 samples, batch 4 → exactly 2 batches."""
        batches = _collect(self.s, 8, 4)
        assert len(batches) == 2

    def test_exact_division_each_batch_size(self):
        """Both batches must have 4 elements."""
        batches = _collect(self.s, 8, 4)
        assert all(len(b) == 4 for b in batches)

    def test_exact_division_covers_all_indices(self):
        """All indices 0-7 appear exactly once."""
        all_idx = sorted(idx for b in _collect(self.s, 8, 4) for idx in b)
        assert all_idx == list(range(8))

    # --- remainder ---

    def test_remainder_chunk_count(self):
        """10 samples, batch 3 → 4 batches (3+3+3+1)."""
        batches = _collect(self.s, 10, 3)
        assert len(batches) == 4

    def test_remainder_batch_sizes(self):
        """First 3 batches have 3 elements; last has 1."""
        batches = _collect(self.s, 10, 3)
        assert [len(b) for b in batches] == [3, 3, 3, 1]

    def test_remainder_covers_all_indices(self):
        """All indices 0-9 appear exactly once."""
        all_idx = sorted(idx for b in _collect(self.s, 10, 3) for idx in b)
        assert all_idx == list(range(10))

    # --- sequential ordering ---

    def test_yields_indices_in_order(self):
        """Indices must be 0, 1, 2, ... N-1 in order."""
        all_idx = [idx for b in _collect(self.s, 7, 3) for idx in b]
        assert all_idx == list(range(7))

    def test_epoch_argument_does_not_affect_order(self):
        """Different epoch values must still produce the same sequential order."""
        epoch0 = [i for b in self.s.yield_batches(6, 2, epoch=0, device=_CPU) for i in b.tolist()]
        epoch3 = [i for b in self.s.yield_batches(6, 2, epoch=3, device=_CPU) for i in b.tolist()]
        assert epoch0 == epoch3 == list(range(6))

    def test_single_batch_equals_full_range(self):
        """batch_size >= num_samples → one batch containing all indices."""
        batches = _collect(self.s, 5, 10)
        assert len(batches) == 1
        assert sorted(batches[0]) == list(range(5))


# ── RandomSampler ──────────────────────────────────────────────────────────────

class TestRandomSampler:
    def setup_method(self):
        self.s = RandomSampler()

    # --- exact division ---

    def test_exact_division_chunk_count(self):
        """8 samples, batch 4 → exactly 2 batches."""
        batches = _collect(self.s, 8, 4)
        assert len(batches) == 2

    def test_exact_division_each_batch_size(self):
        batches = _collect(self.s, 8, 4)
        assert all(len(b) == 4 for b in batches)

    # --- remainder ---

    def test_remainder_chunk_count(self):
        """10 samples, batch 3 → 4 batches."""
        batches = _collect(self.s, 10, 3)
        assert len(batches) == 4

    def test_remainder_batch_sizes(self):
        batches = _collect(self.s, 10, 3)
        assert [len(b) for b in batches] == [3, 3, 3, 1]

    # --- coverage (no duplicates, all indices present) ---

    def test_every_index_appears_exactly_once(self):
        """After one epoch every index 0..N-1 appears exactly once."""
        all_idx = sorted(idx for b in _collect(self.s, 12, 4) for idx in b)
        assert all_idx == list(range(12))

    def test_no_duplicate_indices_within_epoch(self):
        all_idx = [idx for b in _collect(self.s, 9, 3) for idx in b]
        assert len(all_idx) == len(set(all_idx))

    # --- randomness ---

    def test_different_epochs_produce_different_orders(self):
        """Two successive epochs should (almost always) differ in permutation order."""
        torch.manual_seed(0)
        epoch0 = [i for b in self.s.yield_batches(20, 5, epoch=0, device=_CPU) for i in b.tolist()]
        epoch1 = [i for b in self.s.yield_batches(20, 5, epoch=1, device=_CPU) for i in b.tolist()]
        # With 20 elements the chance of an identical permutation is ~1/20! ≈ 0
        assert epoch0 != epoch1

    def test_single_batch_covers_all_indices(self):
        """batch_size >= num_samples → one batch with all indices (shuffled)."""
        batches = _collect(self.s, 5, 10)
        assert len(batches) == 1
        assert sorted(batches[0]) == list(range(5))
