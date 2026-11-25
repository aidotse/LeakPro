"""Phase 2 novelty and unique-passage counts decided on covered content, not on window strings."""

from typing import List

import pytest

from leakpro.attacks.utils.extraction.coverage import (
    KnownContent,
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


@pytest.mark.parametrize(("start", "window_novel"), [(5, False), (1, True), (3, True)])
def test_a_shifted_substring_of_s0(verifier: InMemoryVerifier, start: int, window_novel: bool) -> None:
    # S0 = words 0..99, verified on every window; S1 = S0[start:90]. Window strings (the thesis test)
    # flag start=1 and start=3 as novel, because their windows fall on a different grid; covered
    # content does not.
    s0 = [_cand("s0", _words(0, 100), verifier)]
    s1 = _cand("s1", _words(start, 90), verifier)
    assert len(s1.attacker_matches) == 12
    assert bool(novel_candidates([s1], KnownWindows(s0))) is window_novel
    assert novel_candidates([s1], KnownContent(s0, "full_text", WINDOWING.size)) == []
    assert s1.meta["novel"] is False


def test_text_beyond_s0_is_novel(verifier: InMemoryVerifier) -> None:
    known = KnownContent([_cand("s0", _words(0, 100), verifier)], "full_text", WINDOWING.size)
    s1 = [_cand("inside", _words(3, 90), verifier), _cand("beyond", _words(73, 130), verifier),
          _cand("unverified", "nothing from the corpus " * 10, verifier)]
    assert [c.id for c in novel_candidates(s1, known)] == ["beyond"]
    assert [c.meta["novel"] for c in s1] == [False, True, False]


def test_known_text_counts_even_when_the_corpus_resolves_it_to_another_copy() -> None:
    # The corpus holds the passage twice; a shifted window resolved to the second copy covers new
    # corpus words but no new text, so it is not novel.
    s0 = Candidate(id="s0", phase="initial", strategy="b", prompt_ids=[], gen_ids=[], prompt_text="",
                   continuation=_words(0, 40), full_text=_words(0, 40),
                   attacker_matches=[Match("copy-1", 0, 0, 30, _words(0, 30)), Match("copy-1", 5, 5, 35, _words(5, 35))])
    known = KnownContent([s0], "continuation", 30)
    assert known.is_known(Match("copy-2", 2, 0, 30, _words(2, 32)))  # inside S0's verified span 0..35
    assert not known.is_known(Match("copy-2", 10, 0, 30, _words(10, 40)))  # words 35..39 were never verified
    assert known.is_known(Match("copy-1", 3, 0, 30, "a different string over known corpus words"))


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
