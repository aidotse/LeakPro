"""Result persistence and configuration boundary tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from leakpro.attacks.extraction_attacks.configs import CarliniConfig, SIDEConfig, SimilarityBand
from leakpro.attacks.extraction_attacks.utils import condition_fingerprint
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
    assert json.loads(metadata_path.read_text())["candidate_count"] == 1
    with np.load(array_path) as archive:
        assert archive["images"].shape == (1, 1, 2, 2)


@pytest.mark.parametrize("failure", ["json", "npz"])
def test_failed_first_save_leaves_no_published_or_temporary_artifacts(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    result = _make_result()
    if failure == "json":
        def fail_json(stream: object, payload: dict[str, object]) -> None:
            del stream, payload
            raise OSError("disk full")

        monkeypatch.setattr(ExtractionResult, "_dump_json", staticmethod(fail_json))
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


def test_failed_publish_rolls_back_previous_complete_bundle(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = _make_result(pixel=0.0)
    initial.save(output_dir=tmp_path)
    result_path = tmp_path / "results" / initial.id / "result.json"
    candidates_path = tmp_path / "results" / initial.id / "candidates.npz"
    data_path = tmp_path / "data_objects" / f"{initial.id}.json"
    before = (result_path.read_bytes(), candidates_path.read_bytes(), data_path.read_bytes())
    real_replace = os.replace

    def fail_staged_data_publish(source: object, destination: object) -> None:
        source_path = Path(source)  # type: ignore[arg-type]
        destination_path = Path(destination)  # type: ignore[arg-type]
        if destination_path == data_path and "-stage-" in source_path.name:
            raise OSError("publish interrupted")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_staged_data_publish)
    with pytest.raises(OSError, match="publish interrupted"):
        _make_result(overwrite=True, pixel=1.0).save(output_dir=tmp_path)

    after = (result_path.read_bytes(), candidates_path.read_bytes(), data_path.read_bytes())
    assert after == before
    assert list((tmp_path / "results").glob(f".{initial.id}-*")) == []
    assert list((tmp_path / "data_objects").glob(f".{initial.id}-*")) == []


def test_keyboard_interrupt_during_publish_rolls_back_previous_bundle(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = _make_result(pixel=0.0)
    initial.save(output_dir=tmp_path)
    result_path = tmp_path / "results" / initial.id / "result.json"
    candidates_path = tmp_path / "results" / initial.id / "candidates.npz"
    data_path = tmp_path / "data_objects" / f"{initial.id}.json"
    before = (result_path.read_bytes(), candidates_path.read_bytes(), data_path.read_bytes())
    real_replace = os.replace

    def interrupt_staged_data_publish(source: object, destination: object) -> None:
        source_path = Path(source)  # type: ignore[arg-type]
        destination_path = Path(destination)  # type: ignore[arg-type]
        if destination_path == data_path and "-stage-" in source_path.name:
            raise KeyboardInterrupt
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", interrupt_staged_data_publish)
    with pytest.raises(KeyboardInterrupt):
        _make_result(overwrite=True, pixel=1.0).save(output_dir=tmp_path)

    after = (result_path.read_bytes(), candidates_path.read_bytes(), data_path.read_bytes())
    assert after == before
    assert list((tmp_path / "results").glob(f".{initial.id}-*")) == []
    assert list((tmp_path / "data_objects").glob(f".{initial.id}-*")) == []


def test_interrupt_after_result_rename_rolls_back_previous_bundle(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = _make_result(pixel=0.0)
    initial.save(output_dir=tmp_path)
    result_dir = tmp_path / "results" / initial.id
    result_path = result_dir / "result.json"
    candidates_path = result_dir / "candidates.npz"
    data_path = tmp_path / "data_objects" / f"{initial.id}.json"
    before = (result_path.read_bytes(), candidates_path.read_bytes(), data_path.read_bytes())
    real_replace = os.replace
    interrupted = False

    def rename_then_interrupt(source: object, destination: object) -> None:
        nonlocal interrupted
        source_path = Path(source)  # type: ignore[arg-type]
        destination_path = Path(destination)  # type: ignore[arg-type]
        real_replace(source, destination)
        if not interrupted and destination_path == result_dir and "-stage-" in source_path.name:
            interrupted = True
            raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", rename_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        _make_result(overwrite=True, pixel=1.0).save(output_dir=tmp_path)

    after = (result_path.read_bytes(), candidates_path.read_bytes(), data_path.read_bytes())
    assert after == before
    assert list((tmp_path / "results").glob(f".{initial.id}-*")) == []
    assert list((tmp_path / "data_objects").glob(f".{initial.id}-*")) == []


def test_competing_bundle_appearing_after_staging_is_not_overwritten(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _make_result(pixel=1.0)
    result_dir = tmp_path / "results" / result.id
    data_path = tmp_path / "data_objects" / f"{result.id}.json"
    original_stage = result._stage_bundle  # noqa: SLF001 - deterministic publication-race injection
    competitor_result = b"competitor-result"
    competitor_data = b"competitor-data"

    def stage_then_inject_competitor(
        results_dir: Path,
        data_dir: Path,
        metadata: dict[str, object],
    ) -> object:
        staged = original_stage(results_dir, data_dir, metadata)
        result_dir.mkdir()
        (result_dir / "result.json").write_bytes(competitor_result)
        data_path.write_bytes(competitor_data)
        return staged

    monkeypatch.setattr(result, "_stage_bundle", stage_then_inject_competitor)
    with pytest.raises(FileExistsError, match="appeared while saving"):
        result.save(output_dir=tmp_path)

    assert (result_dir / "result.json").read_bytes() == competitor_result
    assert data_path.read_bytes() == competitor_data
    assert list((tmp_path / "results").glob(f".{result.id}-*")) == []
    assert list((tmp_path / "data_objects").glob(f".{result.id}-*")) == []


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


def test_condition_fingerprint_rejects_process_dependent_objects() -> None:
    class UnsupportedCondition:
        pass

    with pytest.raises(TypeError, match="Unsupported extraction condition type"):
        condition_fingerprint(UnsupportedCondition())
    with pytest.raises(TypeError, match="Unsupported extraction condition type"):
        condition_fingerprint(UnsupportedCondition())


def test_condition_fingerprint_is_stable_for_canonical_scientific_values() -> None:
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

    assert condition_fingerprint(first) == condition_fingerprint(second)
