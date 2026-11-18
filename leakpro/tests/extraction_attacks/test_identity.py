"""Public result identity checks for effective generation and feature behavior."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn

from leakpro import LeakPro
from leakpro.attacks.extraction_attacks.adapters import DiffusionAdapter
from leakpro.attacks.extraction_attacks.carlini import AttackCarliniExtraction
from leakpro.attacks.extraction_attacks.side import AttackSIDEExtraction
from leakpro.attacks.extraction_attacks.utils_generative import adapter_metadata
from leakpro.attacks.extraction_attacks.utils_generative.image_metrics import SimilarityBand
from leakpro.attacks.utils.diff_mi.gaussian_diffusion import ModelMeanType
from leakpro.attacks.utils.diff_mi.respace import SpacedDiffusion
from leakpro.attacks.utils.diff_mi.script_util import create_gaussian_diffusion
from leakpro.tests.extraction_attacks.test_leakpro_integration import SIDEProvider, _write_config
from leakpro.utils.save_load import hash_attack, hash_model


class MetadataDenoiser(nn.Module):
    """Use kwargs and fixed weights so sampler changes remain observable."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, images, timesteps, offset=0.0):
        return torch.cat((images * self.scale + offset, torch.zeros_like(images)), dim=1)


def _diffusion_adapter(**settings):
    schedule_scale = settings.pop("schedule_scale", 1.0)
    mean_type = settings.pop("mean_type", ModelMeanType.EPSILON)
    rescale = settings.pop("rescale_timesteps", True)
    diffusion = create_gaussian_diffusion(
        diffusion_steps=8, learn_sigma=True, noise_schedule="cosine", rescale_timesteps=rescale,
    )
    diffusion = SpacedDiffusion(
        use_timesteps=range(8), betas=diffusion.betas * schedule_scale,
        model_mean_type=mean_type, model_var_type=diffusion.model_var_type,
        loss_type=diffusion.loss_type, rescale_timesteps=rescale, w=diffusion.w,
    )
    return DiffusionAdapter(
        model=MetadataDenoiser(), diffusion=diffusion, image_shape=(1, 4, 4),
        target_config={"architecture": "metadata_denoiser_v1", "scale_initialization": 0.1}, **settings,
    )


def _spatial_images():
    row = torch.tensor([-1, -1, 1, 1]).view(1, 4, 1)
    column = row.transpose(1, 2)
    return torch.stack([
        0.5 + 0.25 * (vertical * row + horizontal * column)
        for vertical, horizontal in ((-1, -1), (-1, 1), (1, -1), (1, 1)) for _ in range(2)
    ])


@pytest.mark.parametrize("attack_name", ["side", "carlini_diffusion"])
def test_public_sampler_variants_have_executed_metadata_and_separate_artifacts(tmp_path: Path, attack_name: str) -> None:
    variants = [
        {}, {"sampling_steps": 4}, {"sampling_steps": 6},
        {"sampling_steps": 4, "clip_guided": True}, {"sampling_steps": 4, "clip_denoised": False},
        {"sampling_steps": 4, "base_model_kwargs": {"offset": 0.05}},
        {"sampling_steps": 4, "schedule_scale": 0.8},
        {"sampling_steps": 4, "mean_type": ModelMeanType.START_X},
        {"sampling_steps": 4, "rescale_timesteps": False},
    ]
    results = []
    for variant in variants:
        class SamplerProvider(SIDEProvider):
            def get_diffusion_adapter(self):
                return _diffusion_adapter(**variant)

        settings = {"overwrite_results": True, "image_range": "minus_one_one"}
        if attack_name == "side":
            settings.update(compute_device="cpu", synthetic_samples=8, synthetic_batch_size=8,
                            clusters=2, min_cluster_size=1, cohesion_threshold=-1.0,
                            classifier_epochs=1, classifier_batch_size=4, num_generations=4,
                            guidance_scale=0.001)
        else:
            settings.update(mode="unconditional_reference_audit", num_unconditional_generations=4,
                            reference_neighbors=2, ratio_threshold=10.0)
        audit = LeakPro(SamplerProvider, _write_config(tmp_path, attack_name, settings))
        attack = audit.attack_scheduler.attacks[0]
        result = audit.run_audit()[0]
        identity = result.provenance["identity"]
        assert result.id.endswith(hash_attack(identity, attack.adapter.model)[:16])
        expected = adapter_metadata(attack.adapter, None, guided=attack_name == "side")
        assert identity["generation"] == expected
        saved = json.loads((tmp_path / "output" / "data_objects" / f"{result.id}.json").read_text())
        assert saved["provenance"]["identity"] == identity
        with np.load(tmp_path / "output" / "results" / result.id / "candidates.npz") as archive:
            np.testing.assert_array_equal(archive["images"], result.images.numpy())
        results.append(result)
    expected_count = len(variants) if attack_name == "side" else len(variants) - 1
    assert len({result.id for result in results}) == expected_count
    assert len(list((tmp_path / "output" / "results").glob("*/candidates.npz"))) == expected_count
    assert not torch.equal(results[0].images, results[1].images)
    if attack_name == "side":
        assert not torch.equal(results[1].images, results[3].images)
    else:
        # Guided-x0 clipping is inactive in Carlini, so it is canonicalized off.
        assert results[1].id == results[3].id


def test_side_similarity_report_id_depends_on_the_active_scorer(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    scorer = Mock(side_effect=AssertionError("Identity must be determined before scoring"))

    class ScorerProvider(SIDEProvider):
        def get_extraction_reference_score(self):
            return scorer

    def side(settings):
        config_path = _write_config(tmp_path, "side", settings)
        return LeakPro(ScorerProvider, config_path).attack_scheduler.attacks[0]

    bands = {"match": {"lower": 0.5, "upper": 1.0}}
    first = side({"similarity_bands": bands, "reference_score_id": "scorer-v1"})
    second = side({"similarity_bands": bands, "reference_score_id": "scorer-v2"})
    with pytest.raises(AttributeError, match="reference_score_fn"):
        first.reference_score_fn = Mock()
    assert first.reference_score_fn is scorer
    assert first.identity_metadata["config"]["reference_score_id"] == "scorer-v1"
    assert first.result_id != second.result_id
    missing_id = LeakPro(ScorerProvider, _write_config(tmp_path, "side", {"similarity_bands": bands}))
    assert not missing_id.attack_scheduler.attacks
    assert "reference_score_id" in caplog.text

    baseline = side({})
    for scorer_id in ("unused", "   "):
        unused = side({"reference_score_id": scorer_id})
        assert "reference_score_id" not in unused.identity_metadata["config"]
        assert baseline.result_id == unused.result_id
    scorer.assert_not_called()


@pytest.mark.parametrize(("setting", "new_value"), [
    ("reference_score_id", "scorer-v2"),
    ("l2_bands", {"near": SimilarityBand(lower=0.0, upper=0.75)}),
])
def test_side_rejects_changed_report_settings_on_an_existing_audit(setting: str, new_value: object) -> None:
    provider = SIDEProvider()
    adapter = provider.get_diffusion_adapter()
    adapter.sample = Mock(side_effect=AssertionError("Changed report settings must be rejected before sampling"))
    config = AttackSIDEExtraction.AttackConfig(
        similarity_bands={"match": {"lower": 0.5, "upper": 1.0}}, reference_score_id="scorer-v1",
        l2_bands={"near": {"lower": 0.0, "upper": 0.25}},
    )
    attack = AttackSIDEExtraction(
        adapter, provider.get_side_feature_extractor(), config,
        feature_metadata=provider.get_side_feature_metadata(),
        reference_images=provider.get_extraction_reference_images(), reference_score_fn=Mock(),
    )
    setattr(config, setting, new_value)
    attack.identity_metadata["config"][setting] = config.model_dump(mode="json")[setting]
    with pytest.raises(ValueError, match="Start a new audit"):
        attack.prepare_attack()
    adapter.sample.assert_not_called()


def test_side_report_uses_the_assigned_scoring_bands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider = SIDEProvider()
    band = SimilarityBand(lower=0.5, upper=1.0)
    near = SimilarityBand(lower=0.0, upper=0.25)
    changed = SimilarityBand(lower=0.9, upper=1.0)
    config = AttackSIDEExtraction.AttackConfig(
        similarity_bands={"match": band}, reference_score_id="scorer-v1", l2_bands={"near": near},
        num_generations=1, distance_block_size=1, compute_device="cpu",
    )
    scores = iter(((0.7, changed), (0.0, band)))

    def sample(*args, **kwargs):
        config.l2_bands["near"] = changed
        config.image_range = "minus_one_one"
        return torch.zeros(1, 1, 4, 4)

    def scorer(candidates, references):
        config.l2_bands["near"] = near
        config.image_range = "zero_one"
        value, band_for_call = next(scores)
        config.similarity_bands["match"] = band_for_call
        return torch.full((len(candidates), len(references)), value)

    attack = AttackSIDEExtraction(
        provider.get_diffusion_adapter(), provider.get_side_feature_extractor(), config,
        feature_metadata=provider.get_side_feature_metadata(), reference_score_fn=scorer,
    )
    attack.identity_metadata["config"]["reference_score_id"] = "edited metadata"
    attack.classifier = nn.Identity()
    attack.cluster_centroids = torch.zeros(2, 2)
    attack._references_zero_one = torch.zeros(2, 1, 4, 4)
    monkeypatch.setattr(attack, "_sample_guided_batch", sample)
    result = attack.run_attack()
    assert not torch.count_nonzero(result.images)
    result.save(output_dir=tmp_path)
    saved = json.loads((tmp_path / "data_objects" / f"{attack.result_id}.json").read_text())
    assert saved["provenance"]["identity"]["config"]["reference_score_id"] == "scorer-v1"
    assert saved["metrics"]["l2_band_scores"]["near"]["unique_references"] == 2
    assert saved["metrics"]["similarity_band_scores"]["match"]["unique_references"] == 1
    assert saved["provenance"]["identity"]["config"]["similarity_bands"]["match"] == band.model_dump()


@pytest.mark.parametrize("scorer_config", [{}, {"reference_score_id": "   "}])
def test_side_similarity_bands_require_a_scorer_id(scorer_config: dict) -> None:
    with pytest.raises(ValueError, match="reference_score_id"):
        AttackSIDEExtraction.AttackConfig(
            similarity_bands={"match": {"lower": 0.5, "upper": 1.0}}, **scorer_config,
        )

    config = AttackSIDEExtraction.AttackConfig(**scorer_config)
    config.similarity_bands["match"] = SimilarityBand(lower=0.5, upper=1.0)
    provider = SIDEProvider()
    with pytest.raises(ValueError, match="reference_score_id"):
        AttackSIDEExtraction(
            provider.get_diffusion_adapter(), provider.get_side_feature_extractor(), config,
            feature_metadata=provider.get_side_feature_metadata(),
        )


def test_unconditional_carlini_ignores_provider_conditions(tmp_path: Path) -> None:
    class ConditionProvider(SIDEProvider):
        get_extraction_conditions = Mock(side_effect=AssertionError("Unconditional audits do not use conditions"))

        def get_diffusion_adapter(self):
            adapter = super().get_diffusion_adapter()
            metadata = adapter.get_extraction_metadata

            def sampler_metadata(*, guided=False, **kwargs):
                payload = metadata(guided=guided)
                payload["sampler"]["conditional"] = kwargs.get("conditional", True)
                return payload

            adapter.get_extraction_metadata = sampler_metadata
            return adapter

    settings = {"mode": "unconditional_reference_audit"}
    config_path = _write_config(tmp_path, "carlini_diffusion", settings)
    actual = LeakPro(ConditionProvider, config_path).attack_scheduler.attacks[0]
    a = AttackCarliniExtraction(actual.adapter, settings, conditions=["unused-a"])
    b = AttackCarliniExtraction(actual.adapter, settings, conditions=["unused-b"])
    assert a.result_id == b.result_id == actual.result_id
    assert "conditions" not in actual.identity_metadata
    assert actual.identity_metadata["generation"]["sampler"]["conditional"] is False
    ConditionProvider.get_extraction_conditions.assert_not_called()


def test_unconditional_carlini_passes_mode_to_provider_metadata(tmp_path: Path) -> None:
    class Provider(SIDEProvider):
        def get_extraction_metadata(self, *, conditional: bool):
            return {"target": {"kind": "stand-in"}, "sampler": {"conditional": conditional}}

    settings = {"mode": "unconditional_reference_audit"}
    audit = LeakPro(Provider, _write_config(tmp_path, "carlini_diffusion", settings))
    assert audit.attack_scheduler.attacks[0].identity_metadata["generation"]["sampler"] == {"conditional": False}


def test_default_resolved_json_metadata_and_equivalent_reconstruction_preserve_identity() -> None:
    first = _diffusion_adapter(sampling_steps=4)
    equivalent = _diffusion_adapter(sampling_steps=4, clip_denoised=True, clip_guided=False, base_model_kwargs={})
    a = AttackCarliniExtraction(first, {}, conditions=["prompt"])
    b = AttackCarliniExtraction(equivalent, {}, conditions=["prompt"])
    assert a.result_id == b.result_id
    declared = first.get_extraction_metadata()
    declared = json.loads(json.dumps(declared, sort_keys=True))
    c = AttackCarliniExtraction(first, {}, conditions=["prompt"], generation_metadata=declared)
    assert c.result_id == a.result_id and c.identity_metadata == a.identity_metadata
    first.base_model_kwargs["offset"] = 1.0
    # The hash inputs are copied, so future mutations cannot rewrite saved declarations.
    assert not a.identity_metadata["generation"]["sampler"]["model_kwargs"]


def test_target_non_state_configuration_and_layout_are_part_of_identity() -> None:
    first = _diffusion_adapter(sampling_steps=4)
    a = AttackCarliniExtraction(first, {}, conditions=["prompt"])
    second = _diffusion_adapter(sampling_steps=4)
    second.target_config = {**first.target_config, "forward_revision": 2}
    b = AttackCarliniExtraction(second, {}, conditions=["prompt"])
    assert hash_model(first.model) == hash_model(second.model)
    assert a.result_id != b.result_id


def test_public_same_state_feature_configs_change_clustering_and_preserve_both_results(tmp_path: Path) -> None:
    """Exercise same-type, empty-state pooling modules through prepare/run/save."""
    images = _spatial_images()
    outputs, labels, losses, feature_states = [], [], [], []
    for output_size in ((1, 2), (2, 1)):
        class FeatureProvider(SIDEProvider):
            def get_diffusion_adapter(self):
                adapter = super().get_diffusion_adapter()
                adapter.sample = lambda batch_size, **kwargs: images[:batch_size].clone()
                return adapter

            def get_side_feature_extractor(self):
                return nn.Sequential(nn.AdaptiveAvgPool2d(output_size), nn.Flatten(1))

            def get_side_feature_metadata(self):
                return {"extractor": {"layers": [{"adaptive_avg_pool": list(output_size)}, {"flatten_start": 1}]}}

        settings = {"overwrite_results": True, "compute_device": "cpu", "synthetic_samples": 8,
                    "synthetic_batch_size": 8, "clusters": 2, "min_cluster_size": 1,
                    "cohesion_threshold": -1.0, "classifier_epochs": 1,
                    "classifier_batch_size": 4, "num_generations": 4}
        audit = LeakPro(FeatureProvider, _write_config(tmp_path, "side", settings))
        attack = audit.attack_scheduler.attacks[0]
        feature_states.append(hash_model(attack.feature_extractor))
        outputs.append(audit.run_audit()[0])
        labels.append(attack.synthetic_labels.clone())
        losses.append(attack.training_history)
    assert feature_states[0] == feature_states[1]
    assert outputs[0].id != outputs[1].id
    # Compare partitions independently of KMeans' arbitrary numeric label names.
    assert not torch.equal(labels[0][:, None] == labels[0], labels[1][:, None] == labels[1])
    assert losses[0] != losses[1]
    assert not torch.equal(outputs[0].images, outputs[1].images)
    assert len(list((tmp_path / "output" / "results").glob("*/candidates.npz"))) == 2
    for result in outputs:
        saved = json.loads((tmp_path / "output" / "data_objects" / f"{result.id}.json").read_text())
        assert saved["provenance"]["identity"]["features"] == result.provenance["identity"]["features"]


def test_transform_closures_buffers_and_equivalent_defaults_are_identified() -> None:
    def closure(shift):
        def transform(images):
            return images.roll(shifts=shift, dims=-1)
        return transform

    class BufferedTransform(nn.Module):
        def __init__(self, shift):
            super().__init__()
            self.register_buffer("shift", torch.tensor(shift))

        def forward(self, images):
            return images + self.shift

    adapter = _diffusion_adapter(sampling_steps=4)
    feature = nn.Sequential(nn.AdaptiveAvgPool2d((1, 2)), nn.Flatten(1))
    descriptor = {"extractor": {"layers": [{"adaptive_avg_pool": [1, 2]}, {"flatten_start": 1}]}}
    def attack(transform=None, metadata=None):
        return AttackSIDEExtraction(adapter, copy.deepcopy(feature), {}, feature_transform=transform,
                                    feature_metadata=metadata or descriptor)

    assert attack().result_id == attack(nn.Identity(), json.loads(json.dumps(descriptor))).result_id
    transforms = [closure(1), closure(2)]
    assert transforms[0].__qualname__ == transforms[1].__qualname__
    assert attack(transforms[0], {**descriptor, "transform": {"revision": 1, "roll": 1}}).result_id != attack(
        transforms[1], {**descriptor, "transform": {"revision": 1, "roll": 2}},
    ).result_id
    buffered = {**descriptor, "transform": {"revision": 1, "operation": "add_buffer"}}
    assert attack(BufferedTransform(0.1), buffered).result_id != attack(BufferedTransform(0.2), buffered).result_id
    with pytest.raises(ValueError, match="transform configuration"):
        attack(closure(1))
    for register in ("register_forward_hook", "register_forward_pre_hook"):
        hooked_identity = nn.Identity()
        getattr(hooked_identity, register)(lambda *args: None)
        with pytest.raises(ValueError, match="non-empty configuration"):
            attack(hooked_identity)


def test_missing_or_non_json_descriptors_fail_before_sampling() -> None:
    adapter = _diffusion_adapter()
    adapter.sample = lambda *args, **kwargs: pytest.fail("Metadata must be validated before generation")
    with pytest.raises(ValueError, match="feature_metadata"):
        AttackSIDEExtraction(adapter, nn.Identity(), {})
    adapter.target_config = None
    with pytest.raises(ValueError, match="non-empty configuration"):
        AttackCarliniExtraction(adapter, {})
    with pytest.raises(TypeError, match="Unsupported result metadata type"):
        AttackCarliniExtraction(adapter, {}, generation_metadata={"target": {"callable": lambda x: x}, "sampler": {"kind": "x"}})


def test_custom_provider_metadata_is_forwarded_by_the_declared_contract(tmp_path: Path) -> None:
    class DeclaredProvider(SIDEProvider):
        def get_diffusion_adapter(self):
            adapter = super().get_diffusion_adapter()
            del adapter.get_extraction_metadata
            return adapter

        def get_extraction_metadata(self):
            return {"target": {"architecture": "identity"}, "sampler": {"kind": "declared_toy", "revision": 2}}

    path = _write_config(tmp_path, "carlini_diffusion", {
        "mode": "unconditional_reference_audit", "num_unconditional_generations": 4, "reference_neighbors": 2,
    })
    result = LeakPro(DeclaredProvider, path).run_audit()[0]
    assert result.provenance["identity"]["generation"]["sampler"] == {"kind": "declared_toy", "revision": 2}


@pytest.mark.parametrize("variadic", [False, True])
def test_legacy_positional_metadata_getters_still_run_public_audit(tmp_path: Path, variadic: bool) -> None:
    class Provider(SIDEProvider):
        def get_diffusion_adapter(self):
            adapter = super().get_diffusion_adapter()

            def metadata(conditional="legacy-default", /, *, guided=False):
                return {"target": {"architecture": "identity"},
                        "sampler": {"kind": "legacy_toy", "setting": conditional, "guided": guided}}

            def variadic_metadata(*conditional, guided=False):
                assert not conditional
                return metadata(guided=guided)

            adapter.get_extraction_metadata = variadic_metadata if variadic else metadata
            return adapter

    path = _write_config(tmp_path, "carlini_diffusion", {
        "mode": "unconditional_reference_audit", "num_unconditional_generations": 4, "reference_neighbors": 2,
    })
    result = LeakPro(Provider, path).run_audit()[0]
    assert result.provenance["identity"]["generation"]["sampler"]["setting"] == "legacy-default"


def test_condition_encoder_configuration_and_search_policy_affect_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    from leakpro.attacks.extraction_attacks import carlini

    def encoder(scale):
        return lambda conditions: {"offset": scale * len(conditions)}

    adapter = _diffusion_adapter(sampling_steps=4)
    adapter.condition_encoder = encoder(0.01)
    with pytest.raises(ValueError, match="condition_config"):
        AttackCarliniExtraction(adapter, {}, conditions=["prompt"])
    adapter.condition_config = {"revision": 1, "offset_per_condition": 0.01}
    a = AttackCarliniExtraction(adapter, {}, conditions=["prompt"])
    adapter.condition_encoder = encoder(0.02)
    adapter.condition_config = {"revision": 1, "offset_per_condition": 0.02}
    b = AttackCarliniExtraction(adapter, {}, conditions=["prompt"])
    assert a.result_id != b.result_id
    monkeypatch.setattr(carlini, "CLIQUE_SEARCH_POLICY", "another_selection_policy")
    c = AttackCarliniExtraction(adapter, {}, conditions=["prompt"])
    assert b.result_id != c.result_id


@pytest.mark.parametrize("attack_name", ["side", "carlini_diffusion"])
def test_inactive_condition_encoders_do_not_block_or_split_public_results(tmp_path: Path, attack_name: str) -> None:
    def unused_encoder(conditions):
        pytest.fail("This sampling mode must not execute the condition encoder")

    settings = {"overwrite_results": True, "image_range": "minus_one_one"}
    if attack_name == "side":
        settings.update(compute_device="cpu", synthetic_samples=8, synthetic_batch_size=8,
                        clusters=2, min_cluster_size=1, cohesion_threshold=-1.0,
                        classifier_epochs=1, classifier_batch_size=4, num_generations=4,
                        guidance_scale=0.001)
    else:
        settings.update(mode="unconditional_reference_audit", num_unconditional_generations=4,
                        reference_neighbors=2)
    results = []
    for encoder, config in ((None, None), (unused_encoder, None),
                            (unused_encoder, {"revision": 1}), (unused_encoder, {"revision": 2})):
        class Provider(SIDEProvider):
            def get_diffusion_adapter(self):
                return _diffusion_adapter(sampling_steps=4, condition_encoder=encoder, condition_config=config)

        audit = LeakPro(Provider, _write_config(tmp_path, attack_name, settings))
        completed = audit.run_audit()
        assert len(completed) == 1
        result = completed[0]
        assert result.provenance["identity"]["generation"]["sampler"]["condition_encoder"] is None
        results.append(result)
    assert len({result.id for result in results}) == 1
    assert all(torch.equal(results[0].images, result.images) for result in results[1:])


def test_custom_diffusion_subclasses_need_their_declared_behavior() -> None:
    class CustomDiffusion(SpacedDiffusion):
        pass

    adapter = _diffusion_adapter()
    adapter.diffusion.__class__ = CustomDiffusion
    with pytest.raises(ValueError, match="diffusion_config"):
        AttackCarliniExtraction(adapter, {})
    adapter.diffusion_config = {"implementation_revision": 1}
    first = AttackCarliniExtraction(adapter, {})
    adapter.diffusion_config = {"implementation_revision": 2}
    second = AttackCarliniExtraction(adapter, {})
    assert first.result_id != second.result_id


def test_declared_identity_hooks_keep_distinct_preprocessing_results(tmp_path: Path) -> None:
    images = _spatial_images()
    results, labels = [], []
    for turns in (0, 1):
        class HookProvider(SIDEProvider):
            def get_diffusion_adapter(self):
                adapter = super().get_diffusion_adapter()
                adapter.sample = lambda count, **kwargs: images[:count].clone()
                return adapter

            def get_side_feature_extractor(self):
                return nn.Sequential(nn.AdaptiveAvgPool2d((1, 2)), nn.Flatten(1))

            def get_side_feature_transform(self):
                transform = nn.Identity()
                transform.register_forward_hook(lambda module, args, output: output.rot90(turns, (-2, -1)))
                return transform

            def get_side_feature_metadata(self):
                return {"extractor": {"pool": [1, 2], "flatten_start": 1},
                        "transform": {"hook_revision": 1, "quarter_turns": turns, "dims": [-2, -1]}}

        path = _write_config(tmp_path, "side", {
            "overwrite_results": True, "compute_device": "cpu", "synthetic_samples": 8,
            "synthetic_batch_size": 8, "clusters": 2, "min_cluster_size": 1,
            "cohesion_threshold": -1.0, "classifier_epochs": 1,
            "classifier_batch_size": 4, "num_generations": 4,
        })
        audit = LeakPro(HookProvider, path)
        results.append(audit.run_audit()[0])
        labels.append(audit.attack_scheduler.attacks[0].synthetic_labels.clone())
    assert results[0].id != results[1].id
    assert not torch.equal(labels[0][:, None] == labels[0], labels[1][:, None] == labels[1])
    assert not torch.equal(results[0].images, results[1].images)
    assert len(list((tmp_path / "output" / "results").glob("*/candidates.npz"))) == 2
    for turns, result in enumerate(results):
        transform = result.provenance["identity"]["features"]["transform"]
        assert transform["config"]["quarter_turns"] == turns
        saved = json.loads((tmp_path / "output" / "data_objects" / f"{result.id}.json").read_text())
        assert saved["provenance"]["identity"]["features"]["transform"] == transform
        with np.load(tmp_path / "output" / "results" / result.id / "candidates.npz") as archive:
            np.testing.assert_array_equal(archive["images"], result.images.numpy())


def test_bfloat16_feature_buffer_remains_supported_through_public_audit(tmp_path: Path) -> None:
    class BF16Feature(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.tensor(1.0, dtype=torch.bfloat16))

        def forward(self, images):
            mean = images.float().mean(dim=(1, 2, 3)) * self.scale.float()
            return torch.stack((mean, 1 - mean), dim=1)

    class BF16Provider(SIDEProvider):
        def get_side_feature_extractor(self):
            return BF16Feature()

        def get_side_feature_metadata(self):
            return {"extractor": {"architecture": "buffer_scaled_mean_v1"}}

    audit = LeakPro(BF16Provider, _write_config(tmp_path, "side", {
        "compute_device": "cpu", "synthetic_samples": 8, "synthetic_batch_size": 8,
        "clusters": 2, "min_cluster_size": 1, "cohesion_threshold": -1.0,
        "classifier_epochs": 1, "classifier_batch_size": 4, "num_generations": 4,
    }))
    result = audit.run_audit()[0]
    assert result.images.shape == (4, 1, 4, 4)
    extractor = result.provenance["identity"]["features"]["extractor"]
    assert extractor["state_layout"]["scale"] == {"shape": [], "dtype": "torch.bfloat16"}
    assert (tmp_path / "output" / "results" / result.id / "candidates.npz").exists()
