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


@pytest.mark.parametrize("filename", ["data_objects/safe-id.json", "results/safe-id/candidates.npz"])
def test_save_propagates_write_errors(tmp_path: Path, filename: str) -> None:
    destination = tmp_path / filename
    destination.mkdir(parents=True)
    with pytest.raises((IsADirectoryError, PermissionError)) as error:
        _make_result().save(output_dir=tmp_path)
    assert Path(error.value.filename) == destination


@pytest.mark.parametrize("result_id", ["../escape", ".", ".."])
def test_result_rejects_path_traversal_id(result_id: str) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        ExtractionResult(
            name="test", result_id=result_id, config={}, images=torch.empty((0, 1, 2, 2)),
            candidates=[], metrics={}, provenance={},
        )
