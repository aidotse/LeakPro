#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Phase 2 (expansion): re-prompt the model with anchors cut from verified Phase 1 sequences.

1. :func:`build_anchors` takes every S0 sequence and cuts an anchor that starts at its first
   verified window and runs for up to ``anchor_tokens`` tokens. Identical anchors are kept once.
2. :class:`AnchorGenerator` prompts with a random-length prefix of each anchor (one length per batch,
   from ``prefix_lengths``) and samples ``gens_per_anchor`` continuations with a temperature that
   decays from ``t_start``. Every generation is ``max_tokens`` long in total (prefix included).
3. After filtering and verification (on prefix + continuation, see :func:`join_prompt`),
   :func:`~leakpro.attacks.utils.extraction.coverage.novel_candidates` keeps the sequences with a
   verified window that S0 does not already have (``novelty``: window strings or covered content).
"""

import math
from typing import Dict, List, Literal, Sequence

import numpy as np
from torch import nn

from leakpro.attacks.utils.extraction.generation import SamplingSettings, TokenizerLike, generate_tokens, pad_id_of
from leakpro.attacks.utils.extraction.records import Candidate
from leakpro.utils.import_helper import Self

EXPANSION_STRATEGY = "expansion"


def build_anchors(
    s0: Sequence[Candidate],
    tok: TokenizerLike,
    text_of: str,
    anchor_tokens: int = 50,
    min_anchor_tokens: int = 10,
) -> List[Dict[str, object]]:
    """Anchors from verified Phase 1 candidates, in S0 order, without duplicates.

    The anchor starts at the first word of the candidate's first verified window (in the original,
    un-normalized text) and keeps up to ``anchor_tokens`` tokens of what follows. Anchors shorter than
    ``min_anchor_tokens`` tokens are dropped.
    """
    anchors: Dict[tuple, Dict[str, object]] = {}
    for cand in s0:
        if not cand.attacker_matches:
            continue
        first = min(cand.attacker_matches, key=lambda m: m.cand_start)
        words = getattr(cand, text_of).split()
        ids = tok.encode(" ".join(words[first.cand_start:]), add_special_tokens=False)[:anchor_tokens]
        if len(ids) < min_anchor_tokens or tuple(ids) in anchors:
            continue
        anchors[tuple(ids)] = {"tokens": list(ids), "parent_id": cand.id, "parent_strategy": cand.strategy}
    return list(anchors.values())


class AnchorGenerator:
    """Generate the Phase 2 candidates, batch by batch (same interface as ``StrategyGenerator``).

    Batch ``b`` belongs to anchor ``b // batches_per_anchor``; every batch picks its own prefix
    length, so the ``gens_per_anchor`` generations of one anchor mix several lengths.

    Args:
    ----
        anchors: Output of :func:`build_anchors`.
        cfg: The expansion config (``ExpansionConfig``).
        tok: Tokenizer.
        phase: Phase name stored on every candidate.

    """

    def __init__(self: Self, anchors: Sequence[Dict[str, object]], cfg: object, tok: TokenizerLike,
                 phase: str = "expansion") -> None:
        self.anchors = list(anchors)
        self.cfg = cfg
        self.tok = tok
        self.phase = phase
        self.name = EXPANSION_STRATEGY
        self.batches_per_anchor = math.ceil(cfg.gens_per_anchor / cfg.batch_size)
        self.total = len(self.anchors) * cfg.gens_per_anchor

    def num_batches(self: Self) -> int:
        """Number of generation batches."""
        return len(self.anchors) * self.batches_per_anchor

    def prefix_length(self: Self, batch_index: int, seed: int) -> int:
        """Prefix length (in anchor tokens) used by one batch."""
        anchor = self.anchors[batch_index // self.batches_per_anchor]["tokens"]
        valid = [n for n in self.cfg.prefix_lengths if n <= len(anchor)] or [len(anchor)]
        rng = np.random.default_rng(np.random.SeedSequence([seed, 2, batch_index]))
        return int(valid[rng.integers(len(valid))])

    def generate_batch(
        self: Self,
        batch_index: int,
        model: nn.Module,
        seed: int,
        device: object,
        backend: Literal["auto", "hf", "torch"] = "auto",
    ) -> List[Candidate]:
        """Generate one batch; empty when the prefix leaves no room under ``max_tokens``."""
        cfg, tok = self.cfg, self.tok
        anchor_index, rep = divmod(batch_index, self.batches_per_anchor)
        anchor = self.anchors[anchor_index]
        rows = min(cfg.batch_size, cfg.gens_per_anchor - rep * cfg.batch_size)
        length = self.prefix_length(batch_index, seed)
        prefix_text = tok.decode(anchor["tokens"][:length], skip_special_tokens=True)
        prompt = tok.encode(prefix_text, add_special_tokens=True)
        gen_len = cfg.max_tokens - len(prompt)
        if gen_len <= 0:
            return []
        settings = SamplingSettings(
            max_new_tokens=gen_len, min_new_tokens=gen_len, do_sample=True, temperature=cfg.temperature,
            top_k=cfg.top_k, top_p=cfg.top_p, repetition_penalty=cfg.repetition_penalty,
            temp_decay=(cfg.t_start, cfg.t_end, cfg.decay_steps),
        )
        seed_b = int(np.random.SeedSequence([seed, 2, batch_index, 1]).generate_state(1)[0])
        new = generate_tokens(model, [prompt] * rows, settings, pad_id_of(tok), tok.eos_token_id, seed_b, device,
                              backend)
        out = []
        for row, ids in enumerate(new):
            cand = Candidate(
                id=f"{self.phase}-{anchor_index:05d}-{rep * cfg.batch_size + row:04d}",
                phase=self.phase,
                strategy=self.name,
                prompt_ids=list(prompt),
                gen_ids=list(ids),
                prompt_text=prefix_text,
                continuation=tok.decode(ids, skip_special_tokens=True).strip(),
                full_text="",
                meta={"batch": batch_index, "anchor": anchor_index, "prefix_len_tokens": length,
                      "parent_id": anchor["parent_id"], "parent_strategy": anchor["parent_strategy"]},
            )
            cand.full_text = join_prompt(tok, cand, cfg.prompt_join)
            out.append(cand)
        return out


def join_prompt(tok: TokenizerLike, cand: Candidate, how: Literal["space", "tokens"] = "space") -> str:
    """Prompt and continuation as one text.

    ``space`` joins the decoded prefix and the decoded continuation with a space (the thesis pipeline).
    When the prefix was cut inside a word, that splits the word in two and shifts every later word
    index. ``tokens`` decodes the prompt and generated token ids together, which keeps the word whole.
    """
    if how == "tokens":
        return tok.decode([*cand.prompt_ids, *cand.gen_ids], skip_special_tokens=True).strip()
    return f"{cand.prompt_text} {cand.continuation}".strip()
