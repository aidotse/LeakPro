"""Phase 2 novelty (window strings, as in the thesis) and unique-passage counts."""

from typing import List

import pytest

from leakpro.attacks.utils.extraction.coverage import (
    KnownWindows,
    merged_spans,
    novel_candidates,
    text_grams,
    unique_passages,
)
from leakpro.attacks.utils.extraction.records import Candidate, Match, Windowing
from leakpro.attacks.utils.extraction.verification import InMemoryVerifier

WORDS = [f"t{i}" for i in range(140)]
DOC = " ".join(WORDS)
WINDOWING = Windowing(size=30, stride=5)


def _cand(cid: str, text: str, verifier: InMemoryVerifier, mode: str = "all") -> Candidate:
    return Candidate(id=cid, phase="expansion", strategy="expansion", prompt_ids=[], gen_ids=[], prompt_text="",
                     continuation=text, full_text=text, attacker_matches=verifier.verify(text, WINDOWING, mode))


def _words(a: int, b: int) -> str:
    return " ".join(WORDS[a:b])


@pytest.fixture
def verifier() -> InMemoryVerifier:
    return InMemoryVerifier([("doc", DOC)])


@pytest.mark.parametrize(("start", "novel"), [(5, False), (1, True), (3, True)])
def test_window_strings_decide_novelty(verifier: InMemoryVerifier, start: int, novel: bool) -> None:
    # S0 = words 0..99, verified on every window; S1 = S0[start:90]. As in the thesis, novelty compares
    # window strings, so start=1 and start=3 are novel: their windows fall on a different grid.
    s1 = _cand("s1", _words(start, 90), verifier)
    assert len(s1.attacker_matches) == 12
    assert bool(novel_candidates([s1], KnownWindows([_cand("s0", _words(0, 100), verifier)]))) is novel
    assert s1.meta["novel"] is novel


def test_text_beyond_s0_is_novel(verifier: InMemoryVerifier) -> None:
    known = KnownWindows([_cand("s0", _words(0, 100), verifier)])
    s1 = [_cand("inside", _words(5, 90), verifier), _cand("beyond", _words(75, 130), verifier),
          _cand("unverified", "nothing from the corpus " * 10, verifier)]
    assert [c.id for c in novel_candidates(s1, known)] == ["beyond"]
    assert [c.meta["novel"] for c in s1] == [False, True, False]


def test_merged_spans_join_overlapping_windows_only() -> None:
    matches = [Match("d", 0, 10, 40, ""), Match("d", 0, 0, 30, ""), Match("d", 0, 40, 70, ""), Match("d", 0, 45, 75, "")]
    assert merged_spans(matches) == [(0, 40), (40, 75)]


def test_text_grams_are_independent_of_the_window_grid(verifier: InMemoryVerifier) -> None:
    grams = text_grams(_cand("c", _words(0, 40), verifier), "full_text", 30)
    assert grams == {_words(i, i + 30) for i in range(11)}


def test_unique_passages(verifier: InMemoryVerifier) -> None:
    # One passage sampled many times with different starts (first-match verification: one window
    # each, on different grids) counts once; a passage elsewhere in the document counts separately.
    same: List[Candidate] = [_cand(f"s{k}", _words(k, k + 40), verifier, mode="first") for k in (0, 1, 3, 7, 12)]
    other = _cand("o", _words(100, 135), verifier, mode="first")
    unverified = _cand("u", "nothing from the corpus " * 10, verifier, mode="first")
    assert unique_passages(same, "full_text", 30) == 1
    assert unique_passages([*same, other, unverified], "full_text", 30) == 2
    assert unique_passages([], "full_text", 30) == 0
    # the same text resolved to two copies of a document is one passage
    a = Candidate(id="a", phase="p", strategy="s", prompt_ids=[], gen_ids=[], prompt_text="",
                  continuation=_words(0, 30), full_text=_words(0, 30),
                  attacker_matches=[Match("copy-1", 0, 0, 30, _words(0, 30))])
    b = Candidate(id="b", phase="p", strategy="s", prompt_ids=[], gen_ids=[], prompt_text="",
                  continuation=_words(0, 30), full_text=_words(0, 30),
                  attacker_matches=[Match("copy-2", 0, 0, 30, _words(0, 30))])
    assert unique_passages([a, b], "full_text", 30) == 1
