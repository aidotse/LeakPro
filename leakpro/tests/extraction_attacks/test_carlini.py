"""End-to-end orchestration tests for both Carlini attack modes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
import torch

from leakpro.tests.extraction_attacks.image_sampler import image_sampler
from leakpro.attacks.extraction_attacks.carlini import AttackCarliniExtraction


def test_conditional_black_box_retains_repeatability_clique() -> None:
    def sample(batch_size: int, conditions: Sequence[Any] | None, seed: int) -> torch.Tensor:
        del seed
        assert conditions == ["memorized"] * batch_size
        images = torch.zeros((batch_size, 1, 4, 4))
        images[10:] = 1.0
        return images

    adapter = image_sampler(image_shape=(1, 4, 4), sample_fn=sample, model=torch.nn.Identity())
    attack = AttackCarliniExtraction(
        adapter,
        {
            "mode": "conditional_black_box",
            "num_generations_per_condition": 22,
            "generation_batch_size": 22,
            "tile_grid": (2, 2),
            "tiled_l2_threshold": 0.01,
            "min_clique_size": 10,
        },
        target_name="conditional-test",
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

    adapter = image_sampler(image_shape=(1, 2, 2), sample_fn=sample, model=torch.nn.Identity())
    attack = AttackCarliniExtraction(
        adapter,
        {
            "mode": "unconditional_reference_audit",
            "num_unconditional_generations": 6,
            "generation_batch_size": 2,
            "reference_neighbors": 3,
            "reference_alpha": 0.5,
            "ratio_threshold": 1.0,
            "tile_grid": (1, 1),
        },
        target_name="unconditional-test",
        reference_images=references,
    )
    attack.prepare_attack()
    result = attack.run_attack()
    assert result.images.shape[0] == 2
    assert {record.nearest_reference_index for record in result.candidates} == {0, 1}
    assert all(record.score == 0.0 for record in result.candidates)


def test_unconditional_candidate_neighborhood_changes_acceptance() -> None:
    """The generated-image denominator accepts this outlier; the old one rejected it."""
    adapter = image_sampler((1, 1, 1), lambda count, conditions, seed: torch.full((count, 1, 1, 1), 0.14), model=torch.nn.Identity())
    attack = AttackCarliniExtraction(
        adapter,
        {
            "mode": "unconditional_reference_audit",
            "num_unconditional_generations": 1,
            "reference_neighbors": 2,
        },
        target_name="candidate-neighborhood",
        reference_images=torch.tensor([0.0, 0.1, 1.0]).reshape(3, 1, 1, 1),
    )
    attack.prepare_attack()
    result = attack.run_attack()
    assert len(result.candidates) == 1
    assert result.candidates[0].nearest_reference_index == 1
    assert result.candidates[0].score == pytest.approx(8 / 9)


@pytest.mark.parametrize("condition", [torch.tensor([1]), np.array([1]), [1], {"label": 1}, True, 1.5])
def test_unsupported_conditions_fail_before_sampling(condition) -> None:
    def sample(*args, **kwargs):
        raise AssertionError("Unsupported conditions must be rejected before generation")

    adapter = image_sampler((1, 4, 4), sample, model=torch.nn.Identity())
    with pytest.raises(TypeError, match="string prompts and integer labels"):
        AttackCarliniExtraction(adapter, {}, conditions=[condition])


def test_prompt_and_label_hashes_use_shared_json_utilities() -> None:
    from leakpro.attacks.extraction_attacks.utils_generative import json_safe
    from leakpro.utils.save_load import hash_attack, hash_config

    model = torch.nn.Linear(1, 1)
    adapter = image_sampler((1, 4, 4), lambda *args, **kwargs: None, model=model)
    conditions = ["1", 1, np.int64(2)]
    attack = AttackCarliniExtraction(adapter, {}, conditions=conditions)
    identity = {"config": attack.config.model_dump(mode="json", exclude={"overwrite_results"}),
                "conditions": [hash_config({"condition": json_safe(value)}) for value in conditions]}
    assert attack.result_id.endswith(hash_attack(identity, model)[:16])
    assert identity["conditions"][0] != identity["conditions"][1]


def test_missing_weights_cannot_fall_back_to_a_target_name() -> None:
    adapter = image_sampler((1, 4, 4), lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="adapter.model"):
        AttackCarliniExtraction(adapter, {}, target_name="not-a-model-hash")
