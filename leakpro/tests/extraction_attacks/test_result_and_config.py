"""Result persistence and configuration boundary tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from leakpro.attacks.extraction_attacks.configs import CarliniConfig, ExtractionConfig, SIDEConfig, SimilarityBand
from leakpro.attacks.extraction_attacks.utils_generative import condition_hash
from leakpro.reporting.extraction_result import CandidateRecord, ExtractionResult


def _make_result(*, overwrite: bool = False, pixel: float = 0.0) -> ExtractionResult:
    return ExtractionResult(
        name="test",
        result_id="safe-id",
        config={},
        images=torch.full((1, 1, 2, 2), pixel),
        candidates=[CandidateRecord(image_index=0, source="unit")],
        metrics={"pixel": pixel},
        provenance={"paper": "test"},
        overwrite=overwrite,
    )


def test_result_save_round_trip(tmp_path: pytest.TempPathFactory) -> None:
    config = CarliniConfig(authorized_audit=True, num_generations_per_condition=2, min_clique_size=2)
    result = ExtractionResult(
        name="test",
        result_id="safe-id",
        config=config,
        images=torch.zeros((1, 1, 2, 2)),
        candidates=[CandidateRecord(image_index=0, source="unit")],
        metrics={"count": np.int64(1)},
        provenance={"paper": "test"},
    )
    result.save(output_dir=tmp_path)
    metadata_path = tmp_path / "results" / "safe-id" / "result.json"
    array_path = tmp_path / "results" / "safe-id" / "candidates.npz"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["candidate_count"] == 1
    assert metadata["candidates"][0]["image_index"] == 0
    assert result.images is result.samples
    with np.load(array_path) as archive:
        assert archive["images"].shape == (1, 1, 2, 2)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float64])
def test_generic_samples_round_trip(tmp_path: Path, dtype: torch.dtype) -> None:
    samples = torch.tensor([[16777217, 2], [3, 4]], dtype=dtype)
    result = ExtractionResult(
        name="test",
        result_id="generic",
        config={},
        samples=samples,
        candidates=[CandidateRecord(sample_index=index, source="unit") for index in range(2)],
        metrics={"count": np.int64(2)},
        provenance={},
    )
    result.save(output_dir=tmp_path)
    with np.load(tmp_path / "results" / "generic" / "candidates.npz") as archive:
        assert archive.files == ["samples"]
        assert archive["samples"].dtype == samples.numpy().dtype
        np.testing.assert_array_equal(archive["samples"], samples.numpy())
    metadata = json.loads((tmp_path / "results" / "generic" / "result.json").read_text())
    assert metadata["metrics"] == {"count": 2}
    assert [candidate["sample_index"] for candidate in metadata["candidates"]] == [0, 1]
    assert result.candidates[1].sample_index == result.candidates[1].image_index == 1
    with pytest.raises(ValueError, match="BCHW"):
        _ = result.images


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "exactly one"),
        ({"images": torch.zeros((0, 1, 2, 2)), "samples": torch.zeros((0, 2))}, "exactly one"),
        ({"samples": torch.tensor(1)}, "batch dimension"),
        ({"samples": torch.zeros((1, 2))}, "metadata count"),
        ({"samples": torch.tensor([[float("nan")]])}, "NaN or infinity"),
        ({"images": torch.zeros((0, 2))}, "BCHW"),
    ],
)
def test_result_validates_sample_payload(payload: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ExtractionResult(
            name="test", result_id="generic", config={}, candidates=[], metrics={}, provenance={}, **payload
        )


@pytest.mark.parametrize("failure", ["json", "npz"])
def test_failed_first_save_leaves_no_published_or_temporary_artifacts(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    result = _make_result()
    if failure == "json":
        def fail_json(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise OSError("disk full")

        monkeypatch.setattr(json, "dumps", fail_json)
    else:
        def fail_npz(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise OSError("disk full")

        monkeypatch.setattr(np, "savez_compressed", fail_npz)

    with pytest.raises(OSError, match="disk full"):
        result.save(output_dir=tmp_path)

    assert not (tmp_path / "results" / result.id).exists()
    assert not (tmp_path / "data_objects" / f"{result.id}.json").exists()
    assert list((tmp_path / "results").glob(f".{result.id}-*")) == []
    assert list((tmp_path / "data_objects").glob(f".{result.id}-*")) == []
    assert list(tmp_path.glob(f".{result.id}-*")) == []


def test_failed_overwrite_preserves_previous_complete_bundle(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = _make_result(pixel=0.0)
    initial.save(output_dir=tmp_path)
    result_path = tmp_path / "results" / initial.id / "result.json"
    candidates_path = tmp_path / "results" / initial.id / "candidates.npz"
    data_path = tmp_path / "data_objects" / f"{initial.id}.json"
    before = (result_path.read_bytes(), candidates_path.read_bytes(), data_path.read_bytes())

    def fail_npz(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", fail_npz)
    with pytest.raises(OSError, match="disk full"):
        _make_result(overwrite=True, pixel=1.0).save(output_dir=tmp_path)

    after = (result_path.read_bytes(), candidates_path.read_bytes(), data_path.read_bytes())
    assert after == before


@pytest.mark.parametrize("overwrite", [False, True])
def test_stale_claim_does_not_block_save(tmp_path: Path, overwrite: bool) -> None:
    """An abandoned claim from an older run does not prevent saving."""
    if overwrite:
        _make_result().save(output_dir=tmp_path)
    (tmp_path / ".extraction_result_claims" / "safe-id").mkdir(parents=True)
    _make_result(overwrite=overwrite, pixel=1.0).save(output_dir=tmp_path)
    with np.load(tmp_path / "results" / "safe-id" / "candidates.npz") as archive:
        assert np.all(archive["images"] == 1.0)
    result = json.loads((tmp_path / "results" / "safe-id" / "result.json").read_text())
    assert result == json.loads((tmp_path / "data_objects" / "safe-id.json").read_text())
    assert result["metrics"]["pixel"] == 1.0


def test_save_requires_explicit_overwrite(tmp_path: Path) -> None:
    """Saving the same result ID without overwrite leaves the original unchanged."""
    _make_result().save(output_dir=tmp_path)
    with pytest.raises(FileExistsError, match="overwrite_results=true"):
        _make_result(pixel=1.0).save(output_dir=tmp_path)
    with np.load(tmp_path / "results" / "safe-id" / "candidates.npz") as archive:
        assert np.all(archive["images"] == 0.0)


@pytest.mark.parametrize("overwrite", [False, True])
def test_failed_publish_can_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overwrite: bool,
) -> None:
    """Failed publication removes incomplete output so a later save can succeed."""
    if overwrite:
        _make_result().save(output_dir=tmp_path)
    real_replace = os.replace
    data_path = tmp_path / "data_objects" / "safe-id.json"

    def fail_metadata_publish(source: Path, destination: Path) -> None:
        if destination == data_path:
            raise OSError("publish interrupted")
        real_replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail_metadata_publish)
        with pytest.raises(OSError, match="publish interrupted"):
            _make_result(overwrite=overwrite, pixel=1.0).save(output_dir=tmp_path)
    assert not (tmp_path / "results" / "safe-id").exists()
    assert not data_path.exists()
    assert list(tmp_path.glob(".safe-id-*")) == []
    _make_result(pixel=1.0).save(output_dir=tmp_path)
    assert json.loads(data_path.read_text())["metrics"]["pixel"] == 1.0


def test_result_rejects_path_traversal_id() -> None:
    with pytest.raises(ValueError, match="unsafe"):
        ExtractionResult(
            name="test",
            result_id="../escape",
            config={},
            images=torch.empty((0, 1, 2, 2)),
            candidates=[],
            metrics={},
            provenance={},
        )

    with pytest.raises(ValueError, match="NaN or infinity"):
        ExtractionResult(
            name="test",
            result_id="safe-id",
            config={},
            images=torch.full((1, 1, 2, 2), torch.nan),
            candidates=[CandidateRecord(image_index=0, source="unit")],
            metrics={},
            provenance={},
        )


def test_configs_reject_incoherent_counts() -> None:
    with pytest.raises(ValidationError, match="min_clique_size"):
        CarliniConfig(num_generations_per_condition=4, min_clique_size=5)
    with pytest.raises(ValidationError, match="clusters"):
        SIDEConfig(synthetic_samples=4, clusters=5)
    with pytest.raises(ValidationError, match="at least two clusters"):
        SIDEConfig(synthetic_samples=3, clusters=2, min_cluster_size=2)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_configs_reject_nonfinite_attack_parameters(value: float) -> None:
    with pytest.raises(ValidationError, match="finite number"):
        SimilarityBand(lower=value, upper=1.0)
    with pytest.raises(ValidationError, match="finite number"):
        CarliniConfig(tiled_l2_threshold=value)
    with pytest.raises(ValidationError, match="finite number"):
        SIDEConfig(classifier_learning_rate=value)


def test_condition_hash_rejects_process_dependent_objects() -> None:
    class UnsupportedCondition:
        pass

    with pytest.raises(TypeError, match="Unsupported extraction condition type"):
        condition_hash(UnsupportedCondition())
    with pytest.raises(TypeError, match="Unsupported extraction condition type"):
        condition_hash(UnsupportedCondition())


def test_condition_hash_is_stable_for_canonical_scientific_values() -> None:
    first = {
        "label": "class-a",
        "embedding": torch.tensor(2.0),
        "array": np.array([1.0, 2.0], dtype=np.float32),
    }
    second = {
        "array": np.array([1.0, 2.0], dtype=np.float32),
        "embedding": torch.tensor(2.0),
        "label": "class-a",
    }

    assert condition_hash(first) == condition_hash(second)


def test_result_rejects_invalid_sample_index() -> None:
    with pytest.raises(ValidationError):
        CandidateRecord(sample_index=-1, source="unit")
    with pytest.raises(ValueError, match="outside the sample batch"):
        ExtractionResult(
            name="test", result_id="generic", config={}, samples=torch.zeros((1, 2)),
            candidates=[CandidateRecord(sample_index=1, source="unit")], metrics={}, provenance={},
        )


def test_generic_config_keeps_image_settings_in_subclasses() -> None:
    """Shared validation is inherited without imposing image settings."""
    config = ExtractionConfig(random_seed=7, authorized_audit=True)
    assert config.model_dump() == {
        "random_seed": 7, "authorized_audit": True, "overwrite_results": False,
    }
    with pytest.raises(ValidationError):
        ExtractionConfig(image_range="zero_one")
    for config_type in (CarliniConfig, SIDEConfig):
        specialized = config_type(**config.model_dump(), image_range="minus_one_one")
        assert isinstance(specialized, ExtractionConfig)
        assert specialized.random_seed == 7
        assert specialized.image_range == "minus_one_one"
        with pytest.raises(ValidationError):
            config_type(image_range="invalid")
