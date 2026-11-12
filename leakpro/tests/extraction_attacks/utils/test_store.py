"""Tests for the stage cache."""

from pathlib import Path

import pytest

from leakpro.attacks.utils.extraction.records import Candidate
from leakpro.attacks.utils.extraction.store import ArtifactStore, stage_key


def _cand(i: int) -> Candidate:
    return Candidate(id=f"c{i}", phase="initial", strategy="s", prompt_ids=[1], gen_ids=[i],
                     prompt_text="", continuation=f"text {i}", full_text=f"text {i}")


def test_stage_key_depends_on_every_part() -> None:
    assert stage_key("a", {"x": 1}) == stage_key("a", {"x": 1})
    assert stage_key("a", {"x": 1}) != stage_key("a", {"x": 2})
    assert stage_key("a", {"x": 1, "y": 2}) == stage_key("a", {"y": 2, "x": 1})


def test_stage_is_done_only_after_write(tmp_path: Path) -> None:
    stage = ArtifactStore(str(tmp_path)).stage("gen", stage_key("k"))
    assert not stage.is_done()
    with pytest.raises(RuntimeError):
        stage.read()
    stage.write([_cand(0), _cand(1)])
    assert stage.is_done()
    assert stage.read() == [_cand(0), _cand(1)]


def test_batches_resume_in_order(tmp_path: Path) -> None:
    stage = ArtifactStore(str(tmp_path)).stage("gen", stage_key("k"))
    stage.write_batch(2, [_cand(4), _cand(5)])
    stage.write_batch(0, [_cand(0)])
    assert stage.completed_batches() == {0, 2}
    assert [c.id for c in stage.read_batches()] == ["c0", "c4", "c5"]


def test_different_keys_use_different_directories(tmp_path: Path) -> None:
    store = ArtifactStore(str(tmp_path))
    assert store.stage("gen", stage_key(1)).path != store.stage("gen", stage_key(2)).path
