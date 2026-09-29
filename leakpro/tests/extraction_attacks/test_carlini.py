"""End-to-end orchestration tests for both Carlini attack modes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
import torch

from leakpro.attacks.extraction_attacks.adapters import CallableDiffusionAdapter
from leakpro.attacks.extraction_attacks.carlini import AttackCarliniExtraction


def test_conditional_black_box_retains_repeatability_clique() -> None:
    def sample(batch_size: int, conditions: Sequence[Any] | None, seed: int) -> torch.Tensor:
        del seed
        assert conditions == ["memorized"] * batch_size
        images = torch.zeros((batch_size, 1, 4, 4))
        images[10:] = 1.0
        return images

    adapter = CallableDiffusionAdapter(image_shape=(1, 4, 4), sample_fn=sample)
    attack = AttackCarliniExtraction(
        adapter,
        {
            "authorized_audit": True,
            "mode": "conditional_black_box",
            "num_generations_per_condition": 22,
            "generation_batch_size": 22,
            "tile_grid": (2, 2),
            "tiled_l2_threshold": 0.01,
            "min_clique_size": 10,
        },
        audit_hash="conditional-test",
        conditions=["memorized"],
        reference_images=torch.ones((1, 1, 4, 4)),
    )
    attack.prepare_attack()
    result = attack.run_attack()
    assert result.images.shape == (1, 1, 4, 4)
    assert result.candidates[0].metadata["largest_clique_size"] == 12
    torch.testing.assert_close(result.images, torch.ones_like(result.images))
    assert result.candidates[0].verified is True
    assert result.metrics["candidate_count"] == 1


def test_unconditional_reference_mode_deduplicates_reference_matches() -> None:
    references = torch.stack([torch.full((1, 2, 2), value) for value in (0.0, 0.2, 0.4, 0.6)])

    def sample(batch_size: int, conditions: Sequence[Any] | None, seed: int) -> torch.Tensor:
        assert conditions is None and batch_size == 2
        exact = references[0 if seed % 2 == 0 else 1]
        return torch.stack([exact, torch.full((1, 2, 2), 0.9)])

    adapter = CallableDiffusionAdapter(image_shape=(1, 2, 2), sample_fn=sample)
    attack = AttackCarliniExtraction(
        adapter,
        {
            "authorized_audit": True,
            "mode": "unconditional_reference_audit",
            "num_unconditional_generations": 6,
            "generation_batch_size": 2,
            "reference_neighbors": 3,
            "reference_alpha": 0.5,
            "ratio_threshold": 1.0,
            "tile_grid": (1, 1),
        },
        audit_hash="unconditional-test",
        reference_images=references,
    )
    attack.prepare_attack()
    result = attack.run_attack()
    assert result.images.shape[0] == 2
    assert {record.nearest_reference_index for record in result.candidates} == {0, 1}
    assert all(record.score == 0.0 for record in result.candidates)


def test_unconditional_candidate_neighborhood_changes_acceptance() -> None:
    """The generated-image denominator accepts this outlier; the old one rejected it."""
    adapter = CallableDiffusionAdapter((1, 1, 1), lambda count, conditions, seed: torch.full((count, 1, 1, 1), 0.14))
    attack = AttackCarliniExtraction(
        adapter,
        {
            "authorized_audit": True,
            "mode": "unconditional_reference_audit",
            "num_unconditional_generations": 1,
            "reference_neighbors": 2,
        },
        audit_hash="candidate-neighborhood",
        reference_images=torch.tensor([0.0, 0.1, 1.0]).reshape(3, 1, 1, 1),
    )
    attack.prepare_attack()
    result = attack.run_attack()
    assert len(result.candidates) == 1
    assert result.candidates[0].nearest_reference_index == 1
    assert result.candidates[0].score == pytest.approx(8 / 9)
