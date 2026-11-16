#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Membership scores used to rank generated candidates before verification.

Following Carlini et al., *Extracting Training Data from Large Language Models* (USENIX 2021),
two scores rank candidates without a reference model:

* ``zlib``: zlib-compressed length of the text divided by ``log(perplexity)``. High = the model
  finds the text likelier than its compressibility suggests.
* ``lowercase``: ``log(perplexity of the lowercased text) / log(perplexity)``. High = the
  likelihood depends on exact casing.

Perplexity is the exponentiated mean next-token negative log-likelihood (NLL), computed in float32
by :class:`~leakpro.signals.token_evidence.CausalLMModel` over the text encoded *with* special
tokens and truncated to ``max_length`` tokens. Both scores only need ``log(perplexity)``, i.e. the
mean NLL, so they are computed from it directly: exponentiating first overflows for texts the model
finds very unlikely, and ``inf / inf`` would make the lowercase ratio NaN.

Selection (:func:`select_top_union`) keeps the top ``top_percent`` of candidates under each
score, merges both sets, and labels candidates picked by both as ``both``.
"""

import math
import zlib
from typing import Dict, List, Optional, Sequence

import numpy as np
from torch.utils.data import DataLoader

from leakpro.attacks.utils.extraction.generation import TokenizerLike, pad_id_of
from leakpro.attacks.utils.extraction.progress import Progress
from leakpro.attacks.utils.extraction.records import Candidate
from leakpro.signals.token_evidence import CausalLMCollate, CausalLMModel

_MIN_LOG_PPL = math.log(1.0000001)  # clamps ppl so that log(ppl) > 0
SCORE_NAMES = ("zlib", "lowercase")


def zlib_entropy(text: str) -> int:
    """zlib-compressed length of ``text`` in bytes."""
    return len(zlib.compress(text.encode("utf-8")))


def mean_nll(
    texts: Sequence[str],
    model: CausalLMModel,
    tok: TokenizerLike,
    batch_size: int = 4,
    max_length: int = 512,
    progress: Optional[Progress] = None,
) -> np.ndarray:
    """Per-text mean next-token negative log-likelihood (``log(perplexity)``) under ``model``.

    Args:
    ----
        texts: Texts to score.
        model: The scoring model.
        tok: Its tokenizer.
        batch_size: Texts per forward pass.
        max_length: Texts are truncated to this many tokens (special tokens included).
        progress: Counts one unit per forward pass.

    Returns:
    -------
        ``(len(texts),)`` float64 mean NLLs.

    """
    encoded = []
    for text in texts:
        ids = tok.encode(text, add_special_tokens=True)[:max_length]
        if len(ids) < 2:
            ids = ids + [pad_id_of(tok)] * (2 - len(ids))
        encoded.append((ids, ids))
    loader = DataLoader(encoded, batch_size=batch_size, shuffle=False, collate_fn=CausalLMCollate(pad_id_of(tok)))
    evidence = model.evidence_from_loader(progress.wrap(loader) if progress is not None else loader)
    nll = -(evidence.logprob * evidence.mask).sum(axis=1) / np.maximum(evidence.lengths, 1)
    return nll.astype(np.float64)


def membership_scores(nll: float, nll_lower: float, text: str) -> Dict[str, float]:
    """The ``zlib`` and ``lowercase`` scores of one text from the mean NLL of it and of its lowercased form."""
    log_p = max(float(nll), _MIN_LOG_PPL)
    log_p_lower = max(float(nll_lower), _MIN_LOG_PPL)
    return {
        "ppl": math.exp(min(log_p, 700.0)),
        "zlib": float(zlib_entropy(text) / log_p),
        "lowercase": float(log_p_lower / log_p),
    }


def score_candidates(
    candidates: Sequence[Candidate],
    model: CausalLMModel,
    tok: TokenizerLike,
    text_of: str = "continuation",
    batch_size: int = 4,
    max_length: int = 512,
    progress_label: Optional[str] = None,
) -> None:
    """Fill ``candidate.scores`` in place (``ppl``, ``zlib``, ``lowercase``).

    Args:
    ----
        candidates: Candidates to score.
        model: Scoring model.
        tok: Its tokenizer.
        text_of: Candidate field that is scored (``continuation`` or ``full_text``).
        batch_size: Texts per forward pass.
        max_length: Truncation in tokens.
        progress_label: Log progress under this name (two passes: as written and lowercased).

    """
    if not candidates:
        return
    texts = [getattr(c, text_of) for c in candidates]
    progress = None
    if progress_label is not None:
        progress = Progress(progress_label, 2 * math.ceil(len(texts) / batch_size), "batch")
    nll = mean_nll(texts, model, tok, batch_size, max_length, progress)
    nll_lower = mean_nll([t.lower() for t in texts], model, tok, batch_size, max_length, progress)
    for cand, text, n, n_lower in zip(candidates, texts, nll, nll_lower):
        cand.scores = membership_scores(n, n_lower, text)


def drop_short_and_duplicates(candidates: Sequence[Candidate], text_of: str, min_words: int) -> List[Candidate]:
    """Drop empty texts, texts under ``min_words`` words, and exact duplicates (first occurrence kept)."""
    seen = set()
    kept = []
    for cand in candidates:
        text = getattr(cand, text_of)
        if not text or len(text.split()) < min_words or text in seen:
            continue
        seen.add(text)
        kept.append(cand)
    return kept


def select_top_union(
    candidates: Sequence[Candidate],
    top_percent: float,
    metrics: Sequence[str] = SCORE_NAMES,
) -> List[Candidate]:
    """Keep the top ``top_percent`` under each metric and merge the sets.

    ``top_n = max(1, int(len(candidates) * top_percent))`` per metric, highest first, ties kept in
    input order. ``selected_by`` lists the metrics that picked a candidate. The result is ordered by
    the first metric, highest first.

    Args:
    ----
        candidates: Scored, deduplicated candidates (one selection pool).
        top_percent: Fraction kept per metric, in (0, 1].
        metrics: Score names to select on.

    Returns:
    -------
        The selected candidates, each with ``selected_by`` set.

    """
    if not candidates:
        return []
    top_n = max(1, int(len(candidates) * top_percent))
    picked: Dict[str, List[str]] = {}
    for metric in metrics:
        values = np.array([c.scores[metric] for c in candidates], dtype=np.float64)
        order = np.argsort(-values, kind="stable")[:top_n]
        for idx in order:
            picked.setdefault(candidates[idx].id, []).append(metric)
    selected = [c for c in candidates if c.id in picked]
    for cand in selected:
        cand.selected_by = picked[cand.id]
    first = metrics[0]
    return sorted(selected, key=lambda c: -c.scores[first])
