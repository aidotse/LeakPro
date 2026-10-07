"""Result persistence tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from leakpro.reporting.extraction_result import CandidateRecord, ExtractionResult


def _make_result(*, overwrite: bool = False, pixel: float = 0.0) -> ExtractionResult:
    return ExtractionResult(
        name="test",
        result_id="safe-id",
        config={},
        images=torch.full((1, 1, 2, 2), pixel),
        candidates=[CandidateRecord(image_index=0, source="unit")],
        metrics={"pixel": np.int64(pixel)},
        provenance={"paper": "test"},
        overwrite=overwrite,
    )


@pytest.mark.parametrize(("failure", "overwrite"), [("json", False), ("npz", False), ("npz", True)])
def test_failed_staged_save_preserves_complete_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    overwrite: bool,
) -> None:
    result = _make_result(overwrite=overwrite, pixel=1.0)
    result_dir = tmp_path / "results" / result.id
    previous = None
    if overwrite:
        _make_result().save(output_dir=tmp_path)
        previous = (
            (result_dir / "result.json").read_bytes(),
            (result_dir / "candidates.npz").read_bytes(),
        )
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

    if overwrite:
        assert previous == (
            (result_dir / "result.json").read_bytes(),
            (result_dir / "candidates.npz").read_bytes(),
        )
    else:
        assert not result_dir.exists()
    assert list((tmp_path / "results").glob(f".{result.id}-*")) == []


@pytest.mark.parametrize("overwrite", [False, True])
def test_failed_publish_preserves_previous_bundle_and_can_be_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overwrite: bool,
) -> None:
    """A failed overwrite keeps the old output and a later save can succeed."""
    result_path = tmp_path / "results" / "safe-id"
    previous = None
    if overwrite:
        _make_result().save(output_dir=tmp_path)
        previous = (
            (result_path / "result.json").read_bytes(),
            (result_path / "candidates.npz").read_bytes(),
        )
    real_replace = os.replace

    def fail_result_publish(source: Path, destination: Path) -> None:
        if destination == result_path:
            raise OSError("publish interrupted")
        real_replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail_result_publish)
        with pytest.raises(OSError, match="publish interrupted"):
            _make_result(overwrite=overwrite, pixel=1.0).save(output_dir=tmp_path)
    if overwrite:
        assert previous == (
            (result_path / "result.json").read_bytes(),
            (result_path / "candidates.npz").read_bytes(),
        )
    else:
        assert not result_path.exists()
    assert list((tmp_path / "results").glob(".safe-id-*")) == []
    _make_result(overwrite=overwrite, pixel=1.0).save(output_dir=tmp_path)
    assert json.loads((result_path / "result.json").read_text())["metrics"]["pixel"] == 1.0
    assert list((tmp_path / "results").glob(".safe-id-*")) == []


@pytest.mark.parametrize("overwrite", [False, True])
def test_competing_publish_does_not_delete_either_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overwrite: bool,
) -> None:
    result_dir = tmp_path / "results" / "safe-id"
    if overwrite:
        _make_result().save(output_dir=tmp_path)
    real_replace = os.replace

    def publish_competing_result(source: Path, destination: Path) -> None:
        if destination == result_dir:
            result_dir.mkdir()
            (result_dir / "result.json").write_text('{"writer": "other"}', encoding="utf-8")
            np.savez_compressed(result_dir / "candidates.npz", images=np.full((1, 1, 2, 2), 2.0))
        real_replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", publish_competing_result)
        with pytest.raises(RuntimeError if overwrite else FileExistsError, match="concurrently"):
            _make_result(overwrite=overwrite, pixel=1.0).save(output_dir=tmp_path)

    assert json.loads((result_dir / "result.json").read_text()) == {"writer": "other"}
    with np.load(result_dir / "candidates.npz") as archive:
        assert np.all(archive["images"] == 2.0)
    recovery_dirs = list((tmp_path / "results").glob(".safe-id-*-previous"))
    assert len(recovery_dirs) == int(overwrite)
    if overwrite:
        assert json.loads((recovery_dirs[0] / "result.json").read_text())["metrics"]["pixel"] == 0.0


@pytest.mark.parametrize("result_id", ["../escape", ".", ".."])
def test_result_rejects_path_traversal_id(tmp_path: Path, result_id: str) -> None:
    existing = tmp_path / "results" / "existing" / "keep.txt"
    existing.parent.mkdir(parents=True)
    existing.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe"):
        ExtractionResult(
            name="test",
            result_id=result_id,
            config={},
            images=torch.empty((0, 1, 2, 2)),
            candidates=[],
            metrics={},
            provenance={},
            overwrite=True,
        ).save(output_dir=tmp_path)
    assert existing.read_text(encoding="utf-8") == "keep"
