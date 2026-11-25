"""Tests for candidate generation: processors, both backends and the strategy generator."""

import pytest
import torch

from leakpro.attacks.utils.extraction.generation import (
    SamplingSettings,
    StrategyGenerator,
    TempDecayLogitsProcessor,
    _apply_repetition_penalty,
    _apply_top_k,
    _apply_top_p,
    batch_seed,
    generate_tokens,
    left_pad,
)
from leakpro.attacks.extraction_attacks.memorization_extraction import GenStrategyConfig
from leakpro.tests.extraction_attacks.fixtures import BOS, EOS, PAD, SECRET, CharTokenizer, ParrotLM, TinyLM

CPU = torch.device("cpu")


def test_temperature_decays_linearly_then_stays_at_t_end() -> None:
    proc = TempDecayLogitsProcessor(10.0, 1.0, 20, prompt_len=1)
    assert proc.temperature(0) == pytest.approx(10.0)
    assert proc.temperature(10) == pytest.approx(5.5)
    assert proc.temperature(20) == proc.temperature(100) == pytest.approx(1.0)
    scores = torch.ones(2, 5)
    assert torch.allclose(proc(torch.zeros(2, 1), scores), scores / 10)   # first generated token
    assert torch.equal(proc(torch.zeros(2, 0), scores), scores)           # still inside the prompt


def test_left_pad() -> None:
    ids, mask = left_pad([[5, 6, 7], [8]], pad_id=0)
    assert ids.tolist() == [[5, 6, 7], [0, 0, 8]]
    assert mask.tolist() == [[1, 1, 1], [0, 0, 1]]


def test_processors_match_transformers() -> None:
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    scores, ids = torch.randn(3, 40), torch.randint(0, 40, (3, 7))
    assert torch.allclose(_apply_repetition_penalty(ids, scores.clone(), 1.1),
                          transformers.RepetitionPenaltyLogitsProcessor(1.1)(ids, scores.clone()))
    assert torch.equal(_apply_top_k(scores, 5), transformers.TopKLogitsWarper(5)(ids, scores.clone()))
    assert torch.equal(_apply_top_p(scores, 0.9), transformers.TopPLogitsWarper(0.9)(ids, scores.clone()))


def test_greedy_torch_backend_is_the_argmax_rollout() -> None:
    model = TinyLM()
    settings = SamplingSettings(max_new_tokens=6, do_sample=False)
    out = generate_tokens(model, [[BOS, 5]], settings, PAD, None, seed=0, device=CPU, backend="torch")
    ids = torch.tensor([[BOS, 5]])
    for _ in range(6):
        nxt = model(ids).logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    assert out == [ids[0, 2:].tolist()]


def test_sampling_is_reproducible_per_seed() -> None:
    model = TinyLM()
    settings = SamplingSettings(max_new_tokens=12, temp_decay=(10.0, 1.0, 20), top_p=0.95)
    run = lambda seed: generate_tokens(model, [[BOS]] * 4, settings, PAD, None, seed, CPU, "torch")  # noqa: E731
    assert run(3) == run(3)
    assert run(3) != run(4)


def test_generation_stops_at_eos_unless_min_new_tokens() -> None:
    model = ParrotLM()
    settings = SamplingSettings(max_new_tokens=500, do_sample=False)
    out = generate_tokens(model, [[BOS]], settings, PAD, EOS, 0, CPU, "torch")[0]
    assert out[-1] == EOS and CharTokenizer().decode(out, skip_special_tokens=True) == SECRET
    forced = generate_tokens(model, [[BOS]], SamplingSettings(max_new_tokens=500, min_new_tokens=300, do_sample=False),
                             PAD, EOS, 0, CPU, "torch")[0]
    assert EOS not in forced[:300]


def test_batch_seed_is_stable_and_distinct() -> None:
    assert batch_seed(42, 0, 3) == batch_seed(42, 0, 3)
    assert len({batch_seed(42, s, b) for s in range(3) for b in range(50)}) == 150


def test_conditional_rows_are_cut_to_max_tokens_in_total() -> None:
    tok = CharTokenizer()
    prefixes = [SECRET[:5], SECRET[:20]]
    cfg = GenStrategyConfig(type="conditional", n=2, max_tokens=40, prefix_path="prefixes.jsonl")
    gen = StrategyGenerator("conditional", 2, cfg, tok, prefixes)
    cands = gen.generate_batch(0, ParrotLM(), seed=1, device=CPU, backend="torch")
    assert [len(c.prompt_ids) + len(c.gen_ids) for c in cands] == [40, 40]
    assert [c.full_text for c in cands] == [f"{SECRET[:5]} {SECRET[5:40]}".strip(), f"{SECRET[:20]} {SECRET[20:40]}".strip()]
    assert cands[0].continuation == SECRET[5:40]


def test_bos_strategies_produce_n_candidates_in_batches() -> None:
    cfg = GenStrategyConfig(type="temp_decay", n=5, batch_size=2, max_tokens=8)
    gen = StrategyGenerator("temp_decay", 1, cfg, CharTokenizer())
    assert gen.num_batches() == 3
    cands = [c for b in range(3) for c in gen.generate_batch(b, TinyLM(), 0, CPU, "torch")]
    assert len(cands) == 5 and len({c.id for c in cands}) == 5
    assert all(c.prompt_ids == [BOS] and c.prompt_text == "" for c in cands)


def test_hf_backend_matches_torch_backend() -> None:
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.GPT2Config(vocab_size=48, n_positions=64, n_embd=16, n_layer=1, n_head=2)
    model = transformers.GPT2LMHeadModel(config).double().eval()
    for settings in (
        SamplingSettings(max_new_tokens=20, do_sample=False),
        SamplingSettings(max_new_tokens=20, temp_decay=(10.0, 1.0, 20), top_k=50, top_p=0.95),
        SamplingSettings(max_new_tokens=20, temp_decay=(3.0, 1.0, 20), top_p=0.9, repetition_penalty=1.1),
    ):
        hf = generate_tokens(model, [[1]] * 3, settings, 0, None, 7, CPU, "hf")
        tc = generate_tokens(model, [[1]] * 3, settings, 0, None, 7, CPU, "torch")
        assert hf == tc
