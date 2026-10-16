"""Extraction image serialization and write-error checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from leakpro.reporting.extraction_result import CandidateRecord, ExtractionResult


def _make_result(pixel: float = 0.0) -> ExtractionResult:
    return ExtractionResult(
        name="test",
        result_id="safe-id",
        config={},
        images=torch.full((1, 1, 2, 2), pixel),
        candidates=[CandidateRecord(image_index=0, source="unit")],
        metrics={"pixel": np.int64(pixel)},
        provenance={"paper": "test"},
    )


def test_save_uses_standard_directories_and_replaces_files(tmp_path: Path) -> None:
    _make_result().save(output_dir=tmp_path)
    newer = _make_result(1.0)
    newer.save(output_dir=tmp_path)
    metadata = json.loads((tmp_path / "data_objects" / "safe-id.json").read_text())
    displayed = json.loads((tmp_path / "results" / "safe-id" / "result.txt").read_text())
    assert metadata == displayed
    assert metadata["metrics"]["pixel"] == 1
    assert metadata["candidates"][0]["image_index"] == 0
    assert "execution_trace" not in metadata
    with np.load(tmp_path / "results" / "safe-id" / "candidates.npz") as arrays:
        assert arrays.files == ["images"]
        np.testing.assert_array_equal(arrays["images"], newer.images.numpy())


@pytest.mark.parametrize("failure", ["json", "npz"])
def test_save_propagates_write_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    def fail_write(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("disk full")

    if failure == "json":
        monkeypatch.setattr(json, "dumps", fail_write)
    else:
        monkeypatch.setattr(np, "savez_compressed", fail_write)
    with pytest.raises(OSError, match="disk full"):
        _make_result().save(output_dir=tmp_path)


@pytest.mark.parametrize("result_id", ["../escape", ".", ".."])
def test_result_rejects_path_traversal_id(tmp_path: Path, result_id: str) -> None:
    existing = tmp_path / "results" / "existing" / "keep.txt"
    existing.parent.mkdir(parents=True)
    existing.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe"):
        ExtractionResult(
            name="test", result_id=result_id, config={}, images=torch.empty((0, 1, 2, 2)),
            candidates=[], metrics={}, provenance={},
        ).save(output_dir=tmp_path)
    assert existing.read_text(encoding="utf-8") == "keep"
