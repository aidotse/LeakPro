#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Records passed between the stages of a memorization extraction attack.

Every stage (generation, scoring, selection, verification) consumes and produces
:class:`Candidate` records, so a phase is just a chain of stages and every intermediate
result can be written to disk as JSON lines and read back unchanged.

Text conventions, stated once and relied on everywhere:

* :func:`normalize` lowercases and collapses every whitespace run to a single space. It is the
  only normalization used for verification and deduplication.
* Verification windows are counted in **words** (whitespace-split), not model tokens, so they
  are independent of the tokenizer of the audited model.
"""

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

from leakpro.utils.import_helper import Self

_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase ``text`` and collapse every whitespace run to a single space."""
    return _WHITESPACE.sub(" ", text.lower()).strip()


@dataclass(frozen=True)
class Windowing:
    """Sliding word windows used to verify a text against a corpus.

    Args:
    ----
        size: Window length in words.
        stride: Step between window starts, in words.
        min_span: Shortest window ever queried. A text shorter than ``size`` words is checked with
            one window of ``max(min_span, n_words)`` words: with the default 10, texts of 10 to
            ``size - 1`` words get one window over the whole text and shorter texts none. Set it to
            ``size`` to skip texts shorter than ``size`` words.

    """

    size: int = 30
    stride: int = 5
    min_span: int = 10

    def __post_init__(self: Self) -> None:
        """Reject window settings that cannot produce a meaningful window."""
        if self.size < 1 or self.stride < 1 or self.min_span < 1:
            raise ValueError("size, stride and min_span must all be >= 1")
        if self.min_span > self.size:
            raise ValueError(f"min_span ({self.min_span}) must be <= size ({self.size})")

    def windows(self: Self, text: str) -> List[Tuple[int, int, str]]:
        """Return ``(start_word, end_word, normalized_window)`` for every window of ``text``."""
        words = text.split()
        n_words = len(words)
        width = self.size if n_words >= self.size else max(self.min_span, n_words)
        return [
            (start, start + width, normalize(" ".join(words[start:start + width])))
            for start in range(0, n_words - width + 1, self.stride)
        ]


@dataclass
class Match:
    """One verified window: where it sits in the candidate and in the corpus.

    Args:
    ----
        doc_id: Identifier of the corpus document that contains the window.
        corpus_offset: Word offset of the window inside that (normalized) document.
        cand_start: First word of the window inside the verified text.
        cand_end: One past the last word of the window inside the verified text.
        window: The normalized window string.

    """

    doc_id: str
    corpus_offset: int
    cand_start: int
    cand_end: int
    window: str


@dataclass
class Candidate:
    """One generated sequence and everything later stages learn about it.

    Args:
    ----
        id: Stable identifier, unique within an attack run.
        phase: Pipeline phase that produced the candidate.
        strategy: Prompting strategy that produced the candidate.
        prompt_ids: Token ids of the prompt.
        gen_ids: Token ids generated after the prompt.
        prompt_text: Decoded prompt (empty for BOS-only prompting).
        continuation: Decoded generated text. Phase 1 scores and verifies this.
        full_text: Prompt and continuation joined by a space.
        scores: Filtering scores by name (``ppl``, ``zlib``, ``lowercase``, ...).
        selected_by: Filters that selected the candidate (``zlib``, ``lowercase``).
        attacker_matches: Windows confirmed by the attacker-side verifier.
        meta: Free-form extra information (batch index, prefix source, ...).

    """

    id: str
    phase: str
    strategy: str
    prompt_ids: List[int]
    gen_ids: List[int]
    prompt_text: str
    continuation: str
    full_text: str
    scores: Dict[str, float] = field(default_factory=dict)
    selected_by: List[str] = field(default_factory=list)
    attacker_matches: List[Match] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self: Self) -> Dict[str, Any]:
        """JSON-serialisable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Candidate":  # noqa: ANN102
        """Inverse of :meth:`to_dict`."""
        data = dict(data)
        data["attacker_matches"] = [Match(**m) for m in data.get("attacker_matches", [])]
        data.pop("oracle_matches", None)  # written by earlier versions
        return cls(**data)
