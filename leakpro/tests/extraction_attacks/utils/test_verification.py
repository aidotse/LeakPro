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


def test_windows_match_as_substrings_even_inside_a_word() -> None:
    # As the reference pipeline: `window in document` on the normalized texts, so "t1 t2 ..." also
    # matches inside "xt1 t2 ..."; the offset is the word in which the match starts.
    v = InMemoryVerifier([("d", "intro xt1 " + " ".join(WORDS[2:40]))])
    assert v.lookup(" ".join(WORDS[1:31])) == ("d", 1)


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


def test_first_mode_finds_a_match_beyond_the_first_lookup_batch() -> None:
    # 60 junk words put the first verified window at index 12, past the first batch of 8 windows
    text = " ".join([f"junk{i}" for i in range(60)] + WORDS[10:60])
    v = _verifier()
    (match,) = v.verify(text, Windowing(), "first")
    assert match.cand_start == 60 and v.first_mode_batch == 8
    assert v.verify(text, Windowing(), "all")[0].cand_start == 60
