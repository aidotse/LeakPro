"""Tests for normalize(), Windowing and Candidate round trips."""

import re

import pytest

from leakpro.attacks.utils.extraction.records import Candidate, Match, Windowing, normalize


def _reference_windows(text: str) -> list:
    """Reference windowing: 30 words (at least 10 for short texts), stride 5, lowercased."""
    tokens = text.split()
    window_size = 30
    if len(tokens) < window_size:
        window_size = max(10, len(tokens))
    out = []
    for i in range(0, len(tokens) - (window_size - 1), 5):
        out.append(re.sub(r"\s+", " ", " ".join(tokens[i:i + window_size]).lower()).strip())
    return out


def test_normalize_lowercases_and_collapses_whitespace() -> None:
    assert normalize("  The\tQuick \n\n Fox  ") == "the quick fox"


@pytest.mark.parametrize("n_words", [0, 5, 9, 10, 12, 29, 30, 31, 34, 35, 47, 100])
def test_default_windowing_matches_reference_windowing(n_words: int) -> None:
    text = " ".join(f"Word{i}" for i in range(n_words))
    ours = [w for _, _, w in Windowing().windows(text)]
    assert ours == _reference_windows(text)


@pytest.mark.parametrize("n_words", [0, 10, 29])
def test_min_span_30_never_checks_fewer_than_30_words(n_words: int) -> None:
    text = " ".join(f"w{i}" for i in range(n_words))
    assert Windowing(min_span=30).windows(text) == []


def test_window_offsets_are_word_positions() -> None:
    text = " ".join(f"w{i}" for i in range(40))
    windows = Windowing().windows(text)
    assert [(s, e) for s, e, _ in windows] == [(0, 30), (5, 35), (10, 40)]
    assert windows[1][2].split()[0] == "w5"


def test_windowing_rejects_min_span_above_size() -> None:
    with pytest.raises(ValueError, match="min_span"):
        Windowing(size=10, min_span=20)


def test_candidate_round_trip() -> None:
    cand = Candidate(id="c1", phase="initial", strategy="baseline", prompt_ids=[1], gen_ids=[5, 6],
                     prompt_text="", continuation="ab", full_text="ab", scores={"zlib": 1.5},
                     selected_by=["zlib"], attacker_matches=[Match("d", 3, 0, 30, "w")], meta={"batch": 0})
    assert Candidate.from_dict(cand.to_dict()) == cand
