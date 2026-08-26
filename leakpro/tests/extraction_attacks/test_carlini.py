"""End-to-end orchestration tests for both Carlini attack modes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
import torch

from leakpro.attacks.extraction_attacks.adapters import CallableDiffusionAdapter
from leakpro.attacks.extraction_attacks.abstract_extraction import AttackState
from leakpro.attacks.extraction_attacks.carlini import AttackCarliniExtraction


def test_conditional_black_box_retains_repeatability_clique() -> None:
    def sample(batch_size: int, conditions: Sequence[Any] | None, seed: int) -> torch.Tensor:
        del seed
        assert conditions == ["memorized"] * batch_size
        images = torch.zeros((batch_size, 1, 4, 4))
        images[-2] = 0.5
        images[-1] = 1.0
        return images

    adapter = CallableDiffusionAdapter(image_shape=(1, 4, 4), sample_fn=sample)
    attack = AttackCarliniExtraction(
        adapter,
        {
            "authorized_audit": True,
            "mode": "conditional_black_box",
            "num_generations_per_condition": 8,
            "generation_batch_size": 8,
            "tile_grid": (2, 2),
            "tiled_l2_threshold": 0.01,
            "min_clique_size": 6,
        },
        audit_fingerprint="conditional-test",
        conditions=["memorized"],
        reference_images=torch.zeros((1, 1, 4, 4)),
    )
    attack.prepare_attack()
    result = attack.run_attack()
    assert result.images.shape == (1, 1, 4, 4)
    assert result.candidates[0].metadata["largest_clique_size"] == 6
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
        audit_fingerprint="unconditional-test",
        reference_images=references,
    )
    attack.prepare_attack()
    result = attack.run_attack()
    assert result.images.shape[0] == 2
    assert {record.nearest_reference_index for record in result.candidates} == {0, 1}
    assert all(record.score == 0.0 for record in result.candidates)


def test_tile_grid_divisibility_applies_only_to_conditional_mode() -> None:
    sample_calls = 0
    references = torch.stack((torch.zeros((1, 3, 3)), torch.ones((1, 3, 3))))

    def sample(batch_size: int, conditions: Sequence[Any] | None, seed: int) -> torch.Tensor:
        nonlocal sample_calls
        del conditions, seed
        sample_calls += 1
        return torch.zeros((batch_size, 1, 3, 3))

    adapter = CallableDiffusionAdapter(image_shape=(1, 3, 3), sample_fn=sample)
    unconditional = AttackCarliniExtraction(
        adapter,
        {
            "authorized_audit": True,
            "mode": "unconditional_reference_audit",
            "num_unconditional_generations": 1,
            "generation_batch_size": 1,
            "reference_neighbors": 2,
            "ratio_threshold": 1.0,
        },
        audit_fingerprint="unconditional-non-tiled-shape-test",
        reference_images=references,
    )

    unconditional.prepare_attack()
    result = unconditional.run_attack()

    assert result.images.shape == (1, 1, 3, 3)
    assert sample_calls == 1

    conditional = AttackCarliniExtraction(
        adapter,
        {
            "authorized_audit": True,
            "mode": "conditional_black_box",
            "num_generations_per_condition": 2,
            "generation_batch_size": 2,
            "min_clique_size": 2,
        },
        audit_fingerprint="conditional-non-tiled-shape-test",
        conditions=["condition"],
    )

    with pytest.raises(ValueError, match="divisible by tile_grid"):
        conditional.prepare_attack()
    assert sample_calls == 1


def test_attack_fails_closed_without_authorization() -> None:
    adapter = CallableDiffusionAdapter(
        image_shape=(1, 2, 2),
        sample_fn=lambda batch_size, conditions, seed: torch.zeros((batch_size, 1, 2, 2)),
    )
    attack = AttackCarliniExtraction(adapter, {}, audit_fingerprint="authorization-test", conditions=["x"])
    with pytest.raises(PermissionError, match="authorized_audit"):
        attack.prepare_attack()


def test_carlini_provenance_names_the_canonical_paper() -> None:
    adapter = CallableDiffusionAdapter(
        image_shape=(1, 2, 2),
        sample_fn=lambda batch_size, conditions, seed: torch.zeros((batch_size, 1, 2, 2)),
    )
    attack = AttackCarliniExtraction(
        adapter,
        {"authorized_audit": True},
        audit_fingerprint="citation-test",
        conditions=["x"],
    )

    description = attack.description()

    assert "Nicholas Carlini" in description["reference"]
    assert "Extracting Training Data from Diffusion Models" in description["reference"]
    assert "medoid" in description["scope"]
    assert "implementation-defined" in description["scope"]


def test_conditional_mode_rejects_none_before_sampling() -> None:
    sample_calls = 0

    def sample(batch_size: int, conditions: Sequence[Any] | None, seed: int) -> torch.Tensor:
        nonlocal sample_calls
        del conditions, seed
        sample_calls += 1
        return torch.zeros((batch_size, 1, 2, 2))

    attack = AttackCarliniExtraction(
        CallableDiffusionAdapter(image_shape=(1, 2, 2), sample_fn=sample),
        {"authorized_audit": True},
        audit_fingerprint="none-condition-test",
        conditions=[None],
    )

    with pytest.raises(ValueError, match="must not contain None"):
        attack.prepare_attack()
    assert attack.state is AttackState.FAILED

    with pytest.raises(RuntimeError, match="one-shot"):
        attack.prepare_attack()

    assert sample_calls == 0


def test_failed_carlini_run_cannot_reuse_partial_state() -> None:
    sample_calls = 0

    def sample(batch_size: int, conditions: Sequence[Any] | None, seed: int) -> torch.Tensor:
        nonlocal sample_calls
        del conditions, seed
        sample_calls += 1
        if sample_calls == 2:
            raise RuntimeError("injected sampler failure")
        return torch.zeros((batch_size, 1, 4, 4))

    attack = AttackCarliniExtraction(
        CallableDiffusionAdapter(image_shape=(1, 4, 4), sample_fn=sample),
        {
            "authorized_audit": True,
            "num_generations_per_condition": 2,
            "generation_batch_size": 2,
            "tile_grid": (2, 2),
            "min_clique_size": 2,
        },
        audit_fingerprint="failed-run-lifecycle-test",
        conditions=["first", "second"],
    )
    attack.prepare_attack()

    with pytest.raises(RuntimeError, match="injected sampler failure"):
        attack.run_attack()
    assert sample_calls == 2

    with pytest.raises(RuntimeError, match="run only once"):
        attack.run_attack()
    assert sample_calls == 2
