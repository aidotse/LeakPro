"""Tests for the in-memory verifier."""

import json
from pathlib import Path

from leakpro.attacks.utils.extraction.records import Windowing
from leakpro.attacks.utils.extraction.verification import InMemoryVerifier

WORDS = [f"t{i}" for i in range(80)]


def _verifier() -> InMemoryVerifier:
    doc = "Intro words here. " + " ".join(WORDS[10:60]) + " outro"
    return InMemoryVerifier([("other", "nothing relevant at all"), ("doc-a", doc)])


def test_verbatim_window_is_found_with_doc_and_word_offset() -> None:
    hit = _verifier().lookup(" ".join(WORDS[10:40]))
    assert hit == ("doc-a", 3)  # three intro words precede it


def test_matching_is_case_and_whitespace_insensitive() -> None:
    text = "  ".join(w.upper() for w in WORDS[20:50])
    assert _verifier().verify(text, Windowing())


def test_windows_never_match_inside_a_word() -> None:
    v = InMemoryVerifier([("d", "xt1 " + " ".join(WORDS[2:40]))])
    assert v.lookup(" ".join(WORDS[1:31])) is None  # "t1" must not match the tail of "xt1"


def test_first_mode_stops_and_all_mode_returns_every_verified_window() -> None:
    text = " ".join(WORDS[0:70])  # words 10..59 are in the corpus
    v, w = _verifier(), Windowing()
    first = v.verify(text, w, "first")
    every = v.verify(text, w, "all")
    assert [m.cand_start for m in first] == [10]
    assert [m.cand_start for m in every] == [10, 15, 20, 25, 30]
    assert all(m.cand_end - m.cand_start == 30 for m in every)


def test_unverified_text_returns_no_match() -> None:
    assert _verifier().verify(" ".join(WORDS[60:80] + WORDS[0:20]), Windowing(), "all") == []


def test_corpus_id_tracks_contents(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text(json.dumps({"doc_id": "a", "text": "x y z"}) + "\n" + json.dumps({"text": "u v"}) + "\n")
    v1 = InMemoryVerifier.from_jsonl(str(path))
    assert v1.num_documents == 2
    assert v1.corpus_id == InMemoryVerifier([("a", "X  y z"), ("1", "u v")]).corpus_id
    assert v1.corpus_id != InMemoryVerifier([("a", "x y z")]).corpus_id
