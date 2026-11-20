"""Report identity for saved model state and effective sampler settings."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
from torch import nn

from leakpro import LeakPro
from leakpro.attacks.extraction_attacks.adapters import DiffusionAdapter
from leakpro.attacks.extraction_attacks.carlini import AttackCarliniExtraction
from leakpro.attacks.extraction_attacks.side import AttackSIDEExtraction
from leakpro.attacks.utils.diff_mi.script_util import create_gaussian_diffusion
from leakpro.tests.extraction_attacks.test_leakpro_integration import SIDEProvider, _write_config
from leakpro.utils.save_load import hash_config, hash_model


def _adapter(**kwargs):
    return DiffusionAdapter(
        model=nn.Linear(1, 1),
        diffusion=create_gaussian_diffusion(diffusion_steps=8, noise_schedule="cosine"),
        image_shape=(1, 4, 4),
        **kwargs,
    )


def test_built_in_sampler_modes_and_model_state_change_ids() -> None:
    adapter = _adapter(sampling_steps=4)
    first = AttackCarliniExtraction(adapter, {})
    side = AttackSIDEExtraction(adapter, nn.Flatten(), {})
    adapter.clip_guided = True
    assert AttackCarliniExtraction(adapter, {}).result_id == first.result_id
    assert AttackSIDEExtraction(adapter, nn.Flatten(), {}).result_id != side.result_id

    assert first.identity_metadata["generation"]["target"]["state_hash"] == hash_model(adapter.model)
    assert first.result_id.endswith(hash_config(first.identity_metadata)[:16])
    with torch.no_grad():
        adapter.model.weight.add_(1)
    assert AttackCarliniExtraction(adapter, {}).result_id != first.result_id
    assert AttackCarliniExtraction(_adapter(sampling_steps=6), {}).identity_metadata["generation"]["sampler"] != (
        first.identity_metadata["generation"]["sampler"]
    )


def test_side_hashes_module_state_and_names_plain_scorers() -> None:
    adapter = _adapter()
    features = nn.Linear(4, 2, dtype=torch.bfloat16)
    transform = nn.BatchNorm2d(1)
    settings = {"similarity_bands": {"match": {"lower": 0.5, "upper": 1.0}}}

    def score(candidates, references):
        return torch.ones((len(candidates), len(references)))

    def other_score(candidates, references):
        return 0.5 * score(candidates, references)

    def audit(scorer=score, config=settings):
        return AttackSIDEExtraction(adapter, features, config, feature_transform=transform, reference_score_fn=scorer)

    first = audit()
    assert first.identity_metadata["reference_score"]["name"].endswith(score.__qualname__)
    assert audit(other_score).result_id != first.result_id
    assert audit(score, {}).result_id == audit(other_score, {}).result_id
    transform.running_mean.add_(1)
    changed_transform = audit()
    assert changed_transform.result_id != first.result_id
    with torch.no_grad():
        features.weight.add_(1)
    assert audit().result_id != changed_transform.result_id

    scorer = nn.Linear(1, 1)
    module_score_id = audit(scorer).result_id
    with torch.no_grad():
        scorer.weight.add_(1)
    assert audit(scorer).result_id != module_score_id


@pytest.mark.parametrize("legacy", [False, True])
def test_side_includes_ordered_optional_settings_for_both_modes(legacy: bool) -> None:
    adapter = SIDEProvider().get_diffusion_adapter()
    synthetic = {"preview": 2, "standard": 6}

    def settings(*, guided):
        sampler = {"standard": 6} if guided else synthetic
        return {"target": {}, "sampler": sampler} if legacy else sampler

    adapter.get_extraction_metadata = settings
    before = AttackSIDEExtraction(adapter, nn.Flatten(), {})
    synthetic = dict(reversed(list(synthetic.items())))
    after = AttackSIDEExtraction(adapter, nn.Flatten(), {})
    assert before.result_id != after.result_id
    assert after.identity_metadata["generation"]["declared_synthetic_settings"]["sampler"]["items"] == [
        [name, steps] for name, steps in synthetic.items()
    ]

    def guided_only(*, guided):
        if not guided:
            raise NotImplementedError
        return settings(guided=guided)

    adapter.get_extraction_metadata = guided_only
    assert "declared_synthetic_settings" not in AttackSIDEExtraction(adapter, nn.Flatten(), {}).identity_metadata["generation"]


def test_diffusion_sampler_keeps_nested_kwargs_order() -> None:
    priority = {"preview": 2, "standard": 6}
    adapter = _adapter(base_model_kwargs={"priority": priority})
    before = AttackCarliniExtraction(adapter, {})
    adapter.base_model_kwargs["priority"] = dict(reversed(list(priority.items())))
    assert AttackCarliniExtraction(adapter, {}).result_id != before.result_id


def test_bound_encoder_does_not_hash_unused_provider_fields() -> None:
    class Provider:
        def encode(self, conditions):
            return {}

    provider = Provider()
    adapter = _adapter(condition_encoder=provider.encode)
    first = AttackCarliniExtraction(adapter, {}, conditions=["prompt"])
    provider.diagnostic = "unused"
    assert AttackCarliniExtraction(adapter, {}, conditions=["prompt"]).result_id == first.result_id
    assert first.identity_metadata["generation"]["condition_encoder"]["name"].endswith(Provider.encode.__qualname__)

    adapter.condition_encoder = lambda conditions: {}
    assert AttackCarliniExtraction(adapter, {}, conditions=["prompt"]).result_id != first.result_id
    assert "condition_encoder" not in AttackCarliniExtraction(
        adapter, {"mode": "unconditional_reference_audit"},
    ).identity_metadata["generation"]


def test_missing_side_scorer_fails_before_sampling(tmp_path) -> None:
    class Provider(SIDEProvider):
        def get_diffusion_adapter(self):
            adapter = super().get_diffusion_adapter()
            adapter.sample = Mock(side_effect=AssertionError("unexpected sample"))
            return adapter

    config = _write_config(tmp_path, "side", {"similarity_bands": {"match": {"lower": 0.5, "upper": 1.0}}})
    with pytest.raises(ValueError, match="requires a reference_score_fn"):
        LeakPro(Provider, config).run_audit()
