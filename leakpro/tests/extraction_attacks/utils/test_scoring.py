"""Tests for the zlib / lowercase scores and top-k union selection."""

import math
import zlib

import numpy as np
import pytest
import torch

from leakpro.attacks.utils.extraction.records import Candidate
from leakpro.attacks.utils.extraction.scoring import (
    drop_short_and_duplicates,
    mean_nll,
    membership_scores,
    select_top_union,
)
from leakpro.signals.token_evidence import CausalLMModel
from leakpro.tests.extraction_attacks.fixtures import CharTokenizer, TinyLM


def _cand(i: int, text: str, **scores: float) -> Candidate:
    return Candidate(id=f"c{i}", phase="initial", strategy="s", prompt_ids=[1], gen_ids=[],
                     prompt_text="", continuation=text, full_text=text, scores=dict(scores))


def test_scores_follow_the_filtering_script() -> None:
    text = "Some Example Text with Capitals"
    s = membership_scores(math.log(20.0), math.log(30.0), text)
    assert s["ppl"] == pytest.approx(20.0)
    assert s["zlib"] == pytest.approx(len(zlib.compress(text.encode())) / math.log(20.0))
    assert s["lowercase"] == pytest.approx(math.log(30.0) / math.log(20.0))
    assert np.isfinite(membership_scores(-1.0, -1.0, text)["zlib"])  # ppl clamped above 1


def test_scores_stay_finite_for_extremely_unlikely_text() -> None:
    s = membership_scores(2e4, 3e4, "text")
    assert all(np.isfinite(v) for v in s.values())
    assert s["lowercase"] == pytest.approx(1.5)


def test_mean_nll_includes_special_tokens() -> None:
    model, tok = TinyLM(), CharTokenizer()
    texts = ["hello world", "abc"]
    got = mean_nll(texts, CausalLMModel(model, device=torch.device("cpu")), tok, batch_size=2)
    for text, value in zip(texts, got):
        ids = torch.tensor([tok.encode(text)])
        logp = torch.log_softmax(model(ids).logits[0, :-1].float(), -1)
        nll = -logp.gather(-1, ids[0, 1:, None]).mean()
        assert value == pytest.approx(nll.item(), rel=1e-5)


def test_short_texts_and_exact_duplicates_are_dropped() -> None:
    words = " ".join(["w"] * 10)
    cands = [_cand(0, words), _cand(1, "too short"), _cand(2, words), _cand(3, ""), _cand(4, words + " x")]
    assert [c.id for c in drop_short_and_duplicates(cands, "continuation", 10)] == ["c0", "c4"]


def test_top_union_matches_the_filtering_script() -> None:
    rng = np.random.default_rng(0)
    cands = [_cand(i, f"t{i}", zlib=float(z), lowercase=float(lo)) for i, (z, lo) in enumerate(rng.random((60, 2)))]
    top_n = int(60 * 0.05)  # 3
    by_zlib = {c.id for c in sorted(cands, key=lambda c: -c.scores["zlib"])[:top_n]}
    by_lower = {c.id for c in sorted(cands, key=lambda c: -c.scores["lowercase"])[:top_n]}
    selected = select_top_union(cands, 0.05)
    assert {c.id for c in selected} == by_zlib | by_lower
    for c in selected:
        assert set(c.selected_by) == {m for m, ids in (("zlib", by_zlib), ("lowercase", by_lower)) if c.id in ids}
    assert [c.scores["zlib"] for c in selected] == sorted((c.scores["zlib"] for c in selected), reverse=True)


def test_top_union_keeps_at_least_one() -> None:
    cands = [_cand(0, "a", zlib=1.0, lowercase=2.0), _cand(1, "b", zlib=3.0, lowercase=0.0)]
    assert {c.id for c in select_top_union(cands, 0.05)} == {"c0", "c1"}
    assert select_top_union([], 0.05) == []
