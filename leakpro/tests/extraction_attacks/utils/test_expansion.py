"""Phase 2 helpers: anchors, expansion generation, novelty."""

import torch

from leakpro.attacks.extraction_attacks.memorization_extraction import ExpansionConfig
from leakpro.attacks.utils.extraction.expansion import AnchorGenerator, build_anchors, known_windows, novel_candidates
from leakpro.attacks.utils.extraction.records import Candidate, Match
from leakpro.tests.extraction_attacks.fixtures import BOS, SECRET, CharTokenizer, ParrotLM


def _cand(cid: str, text: str, starts: list) -> Candidate:
    words = text.split()
    matches = [Match("d", 0, s, s + 3, " ".join(words[s:s + 3])) for s in starts]
    return Candidate(id=cid, phase="initial", strategy="baseline", prompt_ids=[], gen_ids=[], prompt_text="",
                     continuation=text, full_text=text, attacker_matches=matches)


def test_anchors_start_at_first_verified_window_and_are_deduplicated() -> None:
    tok = CharTokenizer()
    text = "aa bb cc dd ee ff gg hh"
    s0 = [_cand("c1", text, [4, 2]), _cand("c2", text, [2]), _cand("c3", text, []), _cand("c4", "zz yy xx ww", [0])]
    anchors = build_anchors(s0, tok, "continuation", anchor_tokens=8, min_anchor_tokens=5)
    assert [tok.decode(a["tokens"]) for a in anchors] == ["cc dd ee", "zz yy xx"]
    assert [a["parent_id"] for a in anchors] == ["c1", "c4"]
    assert build_anchors(s0, tok, "continuation", anchor_tokens=8, min_anchor_tokens=9) == []


def test_anchor_generator_batches_prefixes_and_lengths() -> None:
    tok = CharTokenizer()
    anchor = {"tokens": tok.encode(SECRET, add_special_tokens=False)[:40], "parent_id": "p", "parent_strategy": "baseline"}
    cfg = ExpansionConfig(prefix_lengths=[8, 12, 100], gens_per_anchor=5, batch_size=2, max_tokens=60, top_k=None)
    gen = AnchorGenerator([anchor, anchor], cfg, tok)
    assert gen.num_batches() == 6 and gen.total == 10
    lengths = {gen.prefix_length(b, seed=1) for b in range(6)}
    assert lengths <= {8, 12}
    assert gen.prefix_length(3, seed=1) == gen.prefix_length(3, seed=1)

    model = ParrotLM()
    batch = gen.generate_batch(2, model, seed=1, device=torch.device("cpu"), backend="torch")
    assert len(batch) == 1  # the last batch of an anchor holds the remainder
    cand = batch[0]
    n = cand.meta["prefix_len_tokens"]
    assert cand.prompt_ids[0] == BOS and len(cand.prompt_ids) == n + 1
    assert len(cand.gen_ids) == 60 - (n + 1)  # min_new_tokens fills the budget exactly
    assert SECRET.startswith(cand.prompt_text)
    assert SECRET.startswith(cand.prompt_text + cand.continuation)
    assert cand.full_text == f"{cand.prompt_text} {cand.continuation}".strip()
    assert cand.meta["anchor"] == 0 and cand.strategy == "expansion"


def test_prefix_longer_than_the_budget_gives_an_empty_batch() -> None:
    tok = CharTokenizer()
    anchor = {"tokens": tok.encode(SECRET, add_special_tokens=False)[:40], "parent_id": "p", "parent_strategy": "b"}
    cfg = ExpansionConfig(prefix_lengths=[30], gens_per_anchor=2, batch_size=2, max_tokens=20)
    assert AnchorGenerator([anchor], cfg, tok).generate_batch(0, ParrotLM(), 0, torch.device("cpu"), "torch") == []


def test_novelty_needs_a_window_outside_the_known_set() -> None:
    text = "aa bb cc dd ee ff"
    known = known_windows([_cand("s0", text, [0, 1])])
    s1 = [_cand("old", text, [1]), _cand("new", text, [1, 3]), _cand("none", text, [])]
    novel = novel_candidates(s1, known)
    assert [c.id for c in novel] == ["new"]
    assert [c.meta["novel"] for c in s1] == [False, True, False]
