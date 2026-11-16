#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Verify candidates word for word against a reference corpus.

A candidate is verified when at least one of its windows (see
:class:`~leakpro.attacks.utils.extraction.records.Windowing`) occurs verbatim, after
:func:`~leakpro.attacks.utils.extraction.records.normalize`, in a corpus document. Matching is
word-aligned: a window never matches across the middle of a word.

The same verifier class serves two roles that must stay separate:

* the **attacker verifier** checks candidates against a corpus the attacker owns, and its result
  feeds later attack stages;
* the **audit oracle** checks the attack's outputs against the real training data, and its result
  is only reported.
"""

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from leakpro.attacks.utils.extraction.progress import Progress
from leakpro.attacks.utils.extraction.records import Match, Windowing, normalize
from leakpro.utils.import_helper import Self

VerifyMode = Literal["first", "all"]


class Verifier(ABC):
    """Look windows up in a corpus."""

    @property
    @abstractmethod
    def corpus_id(self: Self) -> str:
        """Identifier of the corpus contents; part of every cache key that depends on verification."""

    @abstractmethod
    def lookup(self: Self, window: str) -> Optional[Tuple[str, int]]:
        """Return ``(doc_id, word_offset)`` of one corpus occurrence of ``window``, or None."""

    def lookup_many(self: Self, windows: Sequence[str]) -> List[Optional[Tuple[str, int]]]:
        """:meth:`lookup` for several windows; backends override it to batch their queries."""
        return [self.lookup(w) for w in windows]

    def verify(self: Self, text: str, windowing: Windowing, mode: VerifyMode = "first") -> List[Match]:
        """Return the verified windows of ``text``.

        Args:
        ----
            text: Text to verify.
            windowing: How ``text`` is cut into windows.
            mode: ``first`` stops at the first verified window; ``all`` checks every window.

        Returns:
        -------
            The verified windows in text order; empty if none.

        """
        windows = windowing.windows(text)
        if mode == "all":
            hits = self.lookup_many([w for _, _, w in windows])
            return [
                Match(doc_id=hit[0], corpus_offset=hit[1], cand_start=start, cand_end=end, window=window)
                for (start, end, window), hit in zip(windows, hits) if hit is not None
            ]
        for start, end, window in windows:
            hit = self.lookup(window)
            if hit is not None:
                return [Match(doc_id=hit[0], corpus_offset=hit[1], cand_start=start, cand_end=end, window=window)]
        return []


class InMemoryVerifier(Verifier):
    """Verifier over documents held in memory. Meant for tests and corpora of modest size.

    Args:
    ----
        documents: ``(doc_id, text)`` pairs.

    """

    def __init__(self: Self, documents: Iterable[Tuple[str, str]]) -> None:
        self._docs: List[Tuple[str, str]] = []
        digest = hashlib.sha256()
        for doc_id, text in documents:
            norm = normalize(text)
            self._docs.append((str(doc_id), f" {norm} "))
            digest.update(f"{doc_id}\x00{norm}\x01".encode("utf-8"))
        self._corpus_id = f"in_memory:{digest.hexdigest()}"
        self._cache: Dict[str, Optional[Tuple[str, int]]] = {}

    @classmethod
    def from_jsonl(cls, path: str, text_field: str = "text", id_field: str = "doc_id") -> "InMemoryVerifier":  # noqa: ANN102
        """Load documents from a JSON-lines file; a missing ``id_field`` falls back to the line number."""
        def _docs() -> Iterable[Tuple[str, str]]:
            with Path(path).open(encoding="utf-8") as f:
                for line_no, line in enumerate(f):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    yield str(record.get(id_field, line_no)), record[text_field]
        return cls(_docs())

    @property
    def corpus_id(self: Self) -> str:
        """Hash of every document id and normalized text."""
        return self._corpus_id

    @property
    def num_documents(self: Self) -> int:
        """Number of documents in the corpus."""
        return len(self._docs)

    def lookup(self: Self, window: str) -> Optional[Tuple[str, int]]:
        """First document containing ``window`` as whole words, with the word offset of the match."""
        if window not in self._cache:
            needle = f" {window} "
            hit = None
            for doc_id, padded in self._docs:
                offset = word_offset(padded, needle)
                if offset is not None:
                    hit = (doc_id, offset)
                    break
            self._cache[window] = hit
        return self._cache[window]


def word_offset(padded: str, needle: str) -> Optional[int]:
    """Word offset of `` needle `` inside `` padded `` (both space-padded, normalized), or None."""
    pos = padded.find(needle)
    return None if pos == -1 else padded[:pos + 1].count(" ") - 1


def verify_candidates(
    candidates: Sequence[object],
    verifier: Verifier,
    windowing: Windowing,
    mode: VerifyMode,
    text_of: str,
    field: str,
    progress: Optional[Progress] = None,
) -> None:
    """Verify ``getattr(c, text_of)`` for every candidate and store the matches on ``c.<field>``."""
    for cand in candidates:
        setattr(cand, field, verifier.verify(getattr(cand, text_of), windowing, mode))
        if progress is not None:
            progress.update()
