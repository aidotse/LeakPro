#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Phase 2 novelty (as in the thesis pipeline) and unique-passage counts.

* :class:`KnownWindows` and :func:`novel_candidates`: a Phase 2 sequence is novel when one of its
  verified window strings was not verified for S0 (exhaustive verification of S0).
* :func:`unique_passages`: verified sequences whose matches cover a common corpus word
  (``doc_id`` + word offset) or share an ``n``-word text count as one passage. It is reported next
  to the sequence counts and changes none of them.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from leakpro.attacks.utils.extraction.records import Candidate, Match, normalize
from leakpro.utils.import_helper import Self


def merged_spans(matches: Iterable[Match]) -> List[Tuple[int, int]]:
    """Word spans ``(start, end)`` of the candidate covered by ``matches``, overlapping windows merged."""
    spans: List[Tuple[int, int]] = []
    for m in sorted(matches, key=lambda m: (m.cand_start, m.cand_end)):
        if spans and m.cand_start < spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], m.cand_end))
        else:
            spans.append((m.cand_start, m.cand_end))
    return spans


def covered_texts(cand: Candidate, text_of: str, field: str = "attacker_matches") -> List[str]:
    """The normalized text of every merged verified span of ``cand``."""
    words = normalize(getattr(cand, text_of)).split()
    return [" ".join(words[a:b]) for a, b in merged_spans(getattr(cand, field))]


def text_grams(cand: Candidate, text_of: str, n: int, field: str = "attacker_matches") -> Set[str]:
    """Every ``n``-word sequence (stride 1) inside the verified spans of ``cand``.

    A span shorter than ``n`` words (possible with ``Windowing.min_span < size``) is one gram.
    """
    grams: Set[str] = set()
    for span in covered_texts(cand, text_of, field):
        words = span.split()
        width = min(n, len(words))
        grams.update(" ".join(words[i:i + width]) for i in range(len(words) - width + 1))
    return grams


def corpus_words(matches: Iterable[Match]) -> Set[Tuple[str, int]]:
    """``(doc_id, word)`` for every corpus word covered by ``matches``."""
    return {(m.doc_id, w) for m in matches for w in range(m.corpus_offset, m.corpus_offset + m.cand_end - m.cand_start)}


class KnownWindows:
    """Window strings verified for a set of candidates (Phase 1's S0, verified on every window).

    A Phase 2 window is known when the same normalized string was verified for S0. The same text
    shifted by a number of words that is not a multiple of the stride is therefore *not* known.
    """

    def __init__(self: Self, candidates: Sequence[Candidate]) -> None:
        self.windows = {m.window for c in candidates for m in c.attacker_matches}

    def is_known(self: Self, match: Match) -> bool:
        """True when ``match``'s window string was verified for S0."""
        return match.window in self.windows


def novel_candidates(
    s1: Sequence[Candidate], known: KnownWindows, novel_key: Optional[str] = "novel",
) -> List[Candidate]:
    """S1 candidates with at least one verified window that ``known`` does not cover; marks ``meta[novel_key]``."""
    out = []
    for cand in s1:
        novel = any(not known.is_known(m) for m in cand.attacker_matches)
        if novel_key:
            cand.meta[novel_key] = novel
        if novel:
            out.append(cand)
    return out


def unique_passages(candidates: Sequence[Candidate], text_of: str, n: int, field: str = "attacker_matches") -> int:
    """Number of distinct memorized passages among the verified ``candidates``.

    Two verified candidates belong to the same passage when their matches cover a common corpus word
    or their verified spans share an ``n``-word sequence; passages are the connected components of
    that relation. Candidates without a match on ``field`` are ignored.
    """
    verified = [c for c in candidates if getattr(c, field)]
    parent = list(range(len(verified)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner: Dict[tuple, int] = {}
    for i, cand in enumerate(verified):
        matches = getattr(cand, field)
        keys = [("w", *w) for w in corpus_words(matches)] + [("t", g) for g in text_grams(cand, text_of, n, field)]
        for key in keys:
            j = owner.setdefault(key, i)
            if j != i:
                parent[find(i)] = find(j)
    return len({find(i) for i in range(len(verified))})
