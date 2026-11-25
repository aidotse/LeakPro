"""Phase 2 helpers: anchors, expansion generation, novelty."""

import torch

from leakpro.attacks.extraction_attacks.memorization_extraction import ExpansionConfig
from leakpro.attacks.utils.extraction.expansion import AnchorGenerator, build_anchors, joint_text
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
    assert cand.full_text == joint_text(tok, cand.prompt_ids, cand.gen_ids)
    assert cand.meta["anchor"] == 0 and cand.strategy == "expansion"


def test_prefix_longer_than_the_budget_gives_an_empty_batch() -> None:
    tok = CharTokenizer()
    anchor = {"tokens": tok.encode(SECRET, add_special_tokens=False)[:40], "parent_id": "p", "parent_strategy": "b"}
    cfg = ExpansionConfig(prefix_lengths=[30], gens_per_anchor=2, batch_size=2, max_tokens=20)
    assert AnchorGenerator([anchor], cfg, tok).generate_batch(0, ParrotLM(), 0, torch.device("cpu"), "torch") == []


def test_a_prefix_cut_inside_a_word_does_not_split_it() -> None:
    # 30 characters of the anchor end inside a word ("... w7x w8"); ParrotLM continues with "x w8x ...".
    # Joining the decoded prefix and continuation with a space would give the words "w8" and "x".
    tok = CharTokenizer()
    anchor = {"tokens": tok.encode(SECRET[4:], add_special_tokens=False)[:40], "parent_id": "p", "parent_strategy": "b"}
    cfg = ExpansionConfig(prefix_lengths=[30], gens_per_anchor=1, batch_size=1, max_tokens=80, top_k=None)
    (cand,) = AnchorGenerator([anchor], cfg, tok).generate_batch(0, ParrotLM(), 0, torch.device("cpu"), "torch")
    assert cand.prompt_text.endswith(" w8") and cand.continuation.startswith("x ")
    words = cand.full_text.split()
    assert all(w.startswith("w") and w.endswith("x") for w in words), words
    assert words[:8] == [f"w{i}x" for i in range(1, 9)]
