"""End-to-end tests of Phase 2 (expansion) and the optional Phases 3-4 (fine-tune, re-extract)."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import pytest
import yaml

from leakpro import LeakPro
from leakpro.attacks.extraction_attacks.memorization_extraction import (
    AttackMemorizationExtraction,
    ExpansionConfig,
    MemorizationExtractionConfig,
)
from leakpro.attacks.utils.extraction import finetune as finetune_module
from leakpro.attacks.utils.extraction.finetune import TokenWindowIndex
from leakpro.attacks.utils.extraction.records import Candidate
from leakpro.attacks.utils.extraction.store import ArtifactStore
from leakpro.tests.extraction_attacks import fixtures
from leakpro.tests.extraction_attacks.fixtures import SECRET, ExtractionInputHandler


def _write_jsonl(path: Path, records: List[Dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return str(path)


def _config(tmp_path: Path, model_class: str, attack: Dict, name: str = "audit.yaml") -> str:
    prefixes = _write_jsonl(tmp_path / "prefixes.jsonl", [{"prefix": SECRET[:k]} for k in (4, 8, 12)])
    for strategy in attack.get("initial", []):
        if strategy["type"] == "conditional":
            strategy.setdefault("prefix_path", prefixes)
    config = {
        "audit": {"random_seed": 3, "attack_type": "extraction", "data_modality": "text",
                  "output_dir": str(tmp_path / "out"),
                  "attack_list": [{"attack": "memorization_extraction", "generation_backend": "torch", **attack}]},
        "target": {"module_path": fixtures.__file__, "model_class": model_class,
                   "target_folder": str(tmp_path / "target")},
    }
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return str(path)


def _run(config: str) -> tuple:
    leakpro = LeakPro(ExtractionInputHandler, config)
    results = leakpro.run_audit()
    return results[0], leakpro


# ------------------------------------------------------------------------------------- Phase 2

def _expansion_attack(tmp_path: Path, initial_tokens: int = 300) -> Dict:
    corpus = _write_jsonl(tmp_path / "aux.jsonl", [{"doc_id": "aux-1", "text": f"Header. {SECRET} Footer."}])
    return {
        # S0: the continuation of "w0x ", i.e. SECRET from w1x on, up to `initial_tokens` characters in total
        "initial": [{"type": "conditional", "n": 1, "batch_size": 1, "max_tokens": initial_tokens,
                     "prefix_path": str(tmp_path / "prefixes.jsonl")}],  # written by _config
        "attacker_verifier": {"path": corpus},
        # The anchor starts at w1x. ParrotLM continues by position, so the 32-character prefix
        # "w1x ... w8x " is followed by "w8x w9x w10x ...": the repeated w8x moves the window grid by
        # one word, so the expansion's window strings differ from S0's although the text is the same.
        "expansion": {"anchor_tokens": 40, "prefix_lengths": [32], "gens_per_anchor": 4, "batch_size": 2,
                      "max_tokens": 300},
    }


def test_expansion_of_s0_text_on_a_shifted_grid_is_novel(tmp_path: Path) -> None:
    result, _ = _run(_config(tmp_path, "ParrotLM", _expansion_attack(tmp_path)))
    assert result.metrics["initial"]["total"]["verified"] == 1
    exp = result.metrics["expansion"]["expansion"]
    assert exp["anchors"] == 1 and exp["generated"] == 4
    assert exp["verified"] == exp["selected"] >= 1
    assert exp["unique"] == 1
    # Every verified window is S0 text on a grid shifted by one word, so every window string is new:
    # novel under the thesis test, which compares window strings.
    assert exp["novel"] == exp["verified"] and exp["novel_unique"] == 1


def test_expansion_reaching_past_s0_is_novel(tmp_path: Path) -> None:
    # S0 now ends at w39x (160 characters with the prefix); the expansion continues SECRET to w59x
    attack = _expansion_attack(tmp_path, initial_tokens=160)
    result, _ = _run(_config(tmp_path, "ParrotLM", attack))
    assert result.metrics["initial"]["total"]["verified"] == 1
    exp = result.metrics["expansion"]["expansion"]
    assert exp["verified"] == exp["selected"] >= 1
    assert exp["novel"] == exp["verified"]
    assert exp["novel_unique"] == exp["unique"] == 1  # the same continuation sampled several times
    assert result.metrics["expansion"]["total"]["novel"] == exp["novel"]

    records = [r for r in result.extracted if r["phase"] == "expansion"]
    assert len(records) == exp["verified"]
    record = records[0]
    assert record["prompt_text"].startswith("w1x w2x") and record["meta"]["novel"] is True
    assert record["full_text"].split()[7:10] == ["w8x", "w8x", "w9x"]
    # every window is reported (exhaustive verification); the windows across the repeated w8x do not
    # verify, so the first verified one starts at w10x
    windows = [m["window"] for m in record["attacker_matches"]]
    assert len(windows) > 1 and windows[0].startswith("w10x w11x")
    # Phase 1's record keeps its first-match result
    initial = next(r for r in result.extracted if r["phase"] == "initial")
    assert len(initial["attacker_matches"]) == 1


def test_expansion_is_on_by_default_and_needs_an_attacker_verifier(tmp_path: Path) -> None:
    attack = _expansion_attack(tmp_path)
    default = {k: v for k, v in attack.items() if k != "expansion"}
    assert MemorizationExtractionConfig(**default).expansion == ExpansionConfig()  # Phase 2 on, default settings
    result, _ = _run(_config(tmp_path, "ParrotLM", {**default, "expansion": None}))
    assert list(result.metrics) == ["initial"]  # `expansion: null` runs Phase 1 only

    # Without an attacker verifier there is no S0: the default Phase 2 is dropped, an explicit one is an error
    no_verifier = {k: v for k, v in default.items() if k != "attacker_verifier"}
    assert MemorizationExtractionConfig(**no_verifier).expansion is None
    with pytest.raises(ValueError, match="attacker_verifier"):
        MemorizationExtractionConfig(**no_verifier, expansion={})
    with pytest.raises(ValueError, match="attacker_verifier"):
        MemorizationExtractionConfig(**no_verifier, finetune={})


def test_phases_1_and_2_by_default_3_and_4_only_when_configured(tmp_path: Path) -> None:
    attack = {k: v for k, v in _expansion_attack(tmp_path).items() if k != "expansion"}
    attack["expansion"] = {"gens_per_anchor": 2, "batch_size": 2, "prefix_lengths": [32], "max_tokens": 300}
    result, _ = _run(_config(tmp_path, "ParrotLM", attack))
    assert list(result.metrics) == ["initial", "expansion"]
    assert MemorizationExtractionConfig(**attack).finetune is None


def test_s0_windows_are_known_so_nothing_is_novel(tmp_path: Path) -> None:
    attack = _expansion_attack(tmp_path)
    attack["initial"] = [{"type": "baseline", "n": 2, "batch_size": 2, "max_tokens": 300}]
    # S0 = SECRET from w0x on; anchor prefix "w0x ... w7x " is continued correctly, so every
    # verified window of the expansion sequences is one S0 already had.
    result, _ = _run(_config(tmp_path, "ParrotLM", attack))
    exp = result.metrics["expansion"]["expansion"]
    assert exp["verified"] >= 1 and exp["novel"] == 0


# ----------------------------------------------------------------------------------- Phases 3-4

def _finetune_attack(tmp_path: Path, attacker: bool = True) -> Dict:
    attack: Dict = {
        "initial": [{"type": "baseline", "n": 6, "batch_size": 3, "max_tokens": 120}],
        "windowing": {"size": 5, "stride": 5, "min_span": 5},
        "filter": {"top_percent": 1.0, "min_words": 5},
        "expansion": None,  # Phases 3-4 on S0 alone keep these tests short
    }
    if attacker:
        attack["attacker_verifier"] = {"path": str(tmp_path / "aux.jsonl")}
        attack["finetune"] = {"target_modules": ["c_attn"], "epochs": 1, "grad_accum": 2, "lr": 1e-3,
                              "overlap_window": 10, "corpus_min_words": 5}
    return attack


def _corpus_from_cached_generations(tmp_path: Path) -> None:
    """The attacker corpus is the model's own Phase 1 output, so every candidate verifies."""
    (stage,) = (tmp_path / "out" / "artifacts").glob("initial-generate-baseline-*")
    with (stage / "candidates.jsonl").open(encoding="utf-8") as f:
        texts = [json.loads(line)["continuation"] for line in f]
    _write_jsonl(tmp_path / "aux.jsonl", [{"doc_id": f"d{i}", "text": t} for i, t in enumerate(texts)])


def test_finetune_and_reextract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("peft")
    _run(_config(tmp_path, "TinyGPT2LM", _finetune_attack(tmp_path, attacker=False), name="gen.yaml"))
    _corpus_from_cached_generations(tmp_path)

    config = _config(tmp_path, "TinyGPT2LM", _finetune_attack(tmp_path))
    result, leakpro = _run(config)
    s0 = result.metrics["initial"]["total"]["verified"]
    assert s0 >= 1
    ft = result.metrics["finetuned"]
    assert ft["total"]["finetune_corpus"] == s0
    assert ft["baseline"]["generated"] == 6
    # Phase 4 starts from the same seeds, and a lightly tuned model mostly repeats its Phase 1
    # output, which is in the fine-tuning corpus: those generations are dropped before filtering.
    assert ft["baseline"]["overlap_removed"] >= 1
    assert ft["baseline"]["kept"] <= 6 - ft["baseline"]["overlap_removed"]
    assert not any("lora" in name for name, _ in leakpro.handler.target_model.named_modules())

    # a rerun reads the adapter from the cache instead of training again
    def _fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("the adapter should have been read from the cache")

    monkeypatch.setattr(finetune_module, "train_lora", _fail)
    import leakpro.attacks.extraction_attacks.memorization_extraction as attack_module
    monkeypatch.setattr(attack_module, "train_lora", _fail)
    again, _ = _run(config)
    assert again.metrics == result.metrics


def test_seed_checks_report_phase1_overlap_and_an_adapter_off_control(tmp_path: Path) -> None:
    pytest.importorskip("peft")
    _run(_config(tmp_path, "TinyGPT2LM", _finetune_attack(tmp_path, attacker=False), name="gen.yaml"))
    _corpus_from_cached_generations(tmp_path)
    attack = _finetune_attack(tmp_path)
    # LoRA's B matrices start at zero, so an adapter trained at lr 1e-12 leaves the model unchanged:
    # Phase 4 then regenerates Phase 1's pool with the same seeds, which is exactly what the control uses.
    attack["finetune"]["lr"] = 1e-12
    result, _ = _run(_config(tmp_path, "TinyGPT2LM", attack))
    ft, control = result.metrics["finetuned"], result.metrics["finetuned_control"]
    for key in ("generated", "overlap_removed", "kept", "selected", "verified", "unique"):
        assert control["total"][key] == ft["total"][key], key
    # every S_new sequence is a Phase 1 generation here
    assert ft["total"]["in_phase1_text"] == ft["total"]["in_phase1_window"] == ft["total"]["verified"]
    assert list(result.metrics) == ["initial", "finetuned", "finetuned_control"]
    assert not [r for r in result.extracted if r["phase"] == "finetuned_control"]  # reported, not extracted


def test_phases_one_at_a_time_match_run_attack(tmp_path: Path) -> None:
    pytest.importorskip("peft")
    _run(_config(tmp_path, "TinyGPT2LM", _finetune_attack(tmp_path, attacker=False), name="gen.yaml"))
    _corpus_from_cached_generations(tmp_path)
    attack_cfg = _finetune_attack(tmp_path)
    attack_cfg["expansion"] = {"gens_per_anchor": 2, "batch_size": 2, "prefix_lengths": [5], "max_tokens": 60,
                               "max_anchors": 1}
    config = _config(tmp_path, "TinyGPT2LM", attack_cfg)
    full, _ = _run(config)

    leakpro = LeakPro(ExtractionInputHandler, config)
    attack = leakpro.attack_scheduler.attacks[0]
    attack.prepare_attack()
    with pytest.raises(RuntimeError, match="run_phase1"):
        attack.run_phase2()
    attack.run_phase1()
    with pytest.raises(RuntimeError, match="run_phase2"):
        attack.run_phase3()
    attack.run_phase2()
    assert attack.run_phase3()["finetune_corpus"] >= 1
    assert any("lora" in name for name, _ in leakpro.handler.target_model.named_modules())
    attack._release_adapter()  # as after an interrupted Phase 4: the cached adapter is attached again
    attack.run_phase4()
    assert not any("lora" in name for name, _ in leakpro.handler.target_model.named_modules())
    result = attack.finish()
    assert result.metrics == full.metrics
    assert [r["phase"] for r in result.extracted] == [r["phase"] for r in full.extracted]


class _WordTokenizer:
    """One token per whitespace-separated word."""

    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:  # noqa: ARG002
        return [self.vocab.setdefault(word, len(self.vocab)) for word in text.split()]


def test_overlap_checks_the_generated_text_not_the_prompt(tmp_path: Path) -> None:
    finetune_text = " ".join(f"t{i}" for i in range(40))
    index = TokenWindowIndex([finetune_text], _WordTokenizer(), window=10)
    fresh = " ".join(f"g{i}" for i in range(40))

    def candidate(name: str, prompt_text: str, continuation: str) -> Candidate:
        return Candidate(id=name, phase="finetuned", strategy="conditional", prompt_ids=[], gen_ids=[],
                         prompt_text=prompt_text, continuation=continuation,
                         full_text=f"{prompt_text} {continuation}".strip())

    generated = {"key": "k", "candidates": [
        candidate("prefix-overlaps", finetune_text, fresh),       # only the prompt is fine-tuning text: kept
        candidate("generation-overlaps", "a prefix", finetune_text.upper()),  # generated text overlaps: dropped
        candidate("no-overlap", "", fresh),
    ]}
    attack = SimpleNamespace(store=ArtifactStore(str(tmp_path)))
    out = AttackMemorizationExtraction._drop_overlap(attack, generated, index, "idx", "conditional", "finetuned")
    assert [c.id for c in out["candidates"]] == ["prefix-overlaps", "no-overlap"]
