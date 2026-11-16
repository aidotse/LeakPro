"""End-to-end tests: LeakPro(...).run_audit() with attack_type: extraction.

The target is ParrotLM, which reproduces ``SECRET`` whatever it is prompted with, so every strategy
generates the same text and the expected counts are known exactly.
"""

import json
from pathlib import Path
from typing import Optional

import pytest
import yaml

from leakpro import LeakPro
from leakpro.attacks.utils.extraction import generation
from leakpro.reporting.extraction_result import ExtractionResult
from leakpro.tests.extraction_attacks import fixtures
from leakpro.tests.extraction_attacks.fixtures import SECRET, ExtractionInputHandler

N = 6


def _write_jsonl(path: Path, records: list) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return str(path)


def _audit(tmp_path: Path, attack: dict, name: str = "audit.yaml") -> str:
    prefixes = _write_jsonl(tmp_path / "prefixes.jsonl", [{"prefix": SECRET[:k]} for k in (4, 8, 12, 16, 20, 24)])
    config = {
        "audit": {
            "random_seed": 3,
            "attack_type": "extraction",
            "data_modality": "text",
            "output_dir": str(tmp_path / "out"),
            "attack_list": [{"attack": "memorization_extraction", **attack}],
        },
        "target": {
            "module_path": fixtures.__file__,
            "model_class": "ParrotLM",
            "target_folder": str(tmp_path / "target"),
            "data_path": prefixes,
        },
    }
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return str(path)


def _attack(tmp_path: Path, oracle_text: Optional[str] = None, with_attacker: bool = True) -> dict:
    corpus = _write_jsonl(tmp_path / "aux.jsonl", [{"doc_id": "noise", "text": "unrelated words"},
                                                   {"doc_id": "aux-1", "text": f"Header. {SECRET} Footer."}])
    attack = {
        "initial": [
            {"type": "baseline", "n": N, "batch_size": 4, "max_tokens": 300},
            {"type": "temp_decay", "n": N, "batch_size": 4, "max_tokens": 300},
            {"type": "conditional", "n": N, "batch_size": 4, "max_tokens": 300},
        ],
        "generation_backend": "torch",
        "expansion": None,  # these tests are about Phase 1
    }
    if with_attacker:
        attack["attacker_verifier"] = {"path": corpus}
    if oracle_text is not None:
        attack["audit_oracle"] = {"path": _write_jsonl(tmp_path / "train.jsonl", [{"doc_id": "train-1", "text": oracle_text}])}
    return attack


def _run(config_path: str) -> ExtractionResult:
    results = LeakPro(ExtractionInputHandler, config_path).run_audit()
    assert len(results) == 1
    return results[0]


def test_extraction_audit_end_to_end(tmp_path: Path) -> None:
    result = _run(_audit(tmp_path, _attack(tmp_path, oracle_text="nothing from the secret")))
    m = result.metrics["initial"]
    for strategy in ("baseline", "temp_decay"):
        # all N generations are SECRET: one survives dedupe, it is selected and verified
        assert m[strategy] == {"generated": N, "kept": 1, "selected": 1, "verified": 1, "extraction_rate": 1 / N,
                               "oracle_verified": 0, "oracle_rate": 0.0}
    # conditional continuations differ per prefix (each starts where its prefix ends); top 1 by zlib and
    # top 1 by lowercase may be different candidates, and every continuation is part of SECRET
    cond = m["conditional"]
    assert cond["generated"] == N and cond["kept"] == N
    assert cond["selected"] in (1, 2) and cond["verified"] == cond["selected"]
    assert m["total"]["generated"] == 3 * N and m["total"]["verified"] == 2 + cond["selected"]

    assert len(result.extracted) == 2 + cond["selected"]
    record = next(r for r in result.extracted if r["strategy"] == "baseline")
    assert record["continuation"] == SECRET
    assert record["attacker_matches"][0]["doc_id"] == "aux-1" and record["attacker_matches"][0]["cand_start"] == 0
    assert record["oracle_matches"] == []

    out = tmp_path / "out"
    assert (out / "data_objects" / f"{result.id}.json").exists()
    loaded = ExtractionResult.load(str(out / "results" / result.id))
    assert loaded.metrics == json.loads(json.dumps(result.metrics))
    assert len(loaded.extracted) == len(result.extracted)


def test_oracle_is_reported_but_does_not_change_the_attack(tmp_path: Path) -> None:
    without = _run(_audit(tmp_path, _attack(tmp_path)))
    with_oracle = _run(_audit(tmp_path, _attack(tmp_path, oracle_text=SECRET), name="audit2.yaml"))
    for strategy in ("baseline", "temp_decay", "conditional", "total"):
        a, b = without.metrics["initial"][strategy], with_oracle.metrics["initial"][strategy]
        assert {k: a[k] for k in ("generated", "kept", "selected", "verified")} == \
               {k: b[k] for k in ("generated", "kept", "selected", "verified")}
        assert a["oracle_verified"] is None
    total = with_oracle.metrics["initial"]["total"]
    assert total["oracle_verified"] == total["verified"] == total["selected"]


def test_without_attacker_verifier_nothing_is_verified(tmp_path: Path) -> None:
    result = _run(_audit(tmp_path, _attack(tmp_path, with_attacker=False)))
    assert result.metrics["initial"]["total"]["verified"] is None
    m = result.metrics["initial"]
    assert m["total"]["selected"] == 2 + m["conditional"]["selected"]
    assert result.extracted == []


def test_rerun_reuses_cached_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _audit(tmp_path, _attack(tmp_path))
    first = _run(config)

    def _fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("generation should have been read from the cache")

    monkeypatch.setattr(generation.StrategyGenerator, "generate_batch", _fail)
    second = _run(config)
    assert second.metrics == first.metrics
    assert second.id == first.id


def test_interrupted_generation_resumes_missing_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _audit(tmp_path, _attack(tmp_path))
    original = generation.StrategyGenerator.generate_batch
    calls = []

    def _crash_on_second(self: generation.StrategyGenerator, batch_index: int, *args: object, **kwargs: object) -> list:
        calls.append((self.name, batch_index))
        if self.name == "baseline" and batch_index == 1:
            raise RuntimeError("simulated crash")
        return original(self, batch_index, *args, **kwargs)

    monkeypatch.setattr(generation.StrategyGenerator, "generate_batch", _crash_on_second)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(config)
    monkeypatch.setattr(generation.StrategyGenerator, "generate_batch", original)

    resumed = []
    monkeypatch.setattr(generation.StrategyGenerator, "generate_batch",
                        lambda self, b, *a, **k: resumed.append((self.name, b)) or original(self, b, *a, **k))
    result = _run(config)
    assert ("baseline", 0) not in resumed and ("baseline", 1) in resumed
    assert result.metrics["initial"]["baseline"]["generated"] == N


def test_changing_a_setting_recomputes_only_downstream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    attack = _attack(tmp_path)
    _run(_audit(tmp_path, attack))
    attack["filter"] = {"top_percent": 0.5}  # selection only: generation and scoring stay cached

    def _fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("generation should have been read from the cache")

    monkeypatch.setattr(generation.StrategyGenerator, "generate_batch", _fail)
    result = _run(_audit(tmp_path, attack, name="audit2.yaml"))
    assert 3 <= result.metrics["initial"]["conditional"]["selected"] <= 6  # int(6 * 0.5) = 3 per metric, union


@pytest.mark.parametrize("store_text", ["hash", "none"])
def test_store_text_redacts_saved_records(tmp_path: Path, store_text: str) -> None:
    attack = {**_attack(tmp_path), "store_text": store_text}
    result = _run(_audit(tmp_path, attack))
    lines = (tmp_path / "out" / "results" / result.id / "extracted.jsonl").read_text(encoding="utf-8")
    assert SECRET not in lines and "w10x" not in lines
    assert result.extracted[0]["continuation"]  # the in-memory result keeps the text


def test_editing_the_prefix_pool_regenerates_conditional(tmp_path: Path) -> None:
    config = _audit(tmp_path, _attack(tmp_path))
    _run(config)
    generated_dirs = lambda: {p.name for p in (tmp_path / "out" / "artifacts").glob("initial-generate-conditional-*")}  # noqa: E731
    before = generated_dirs()
    prefixes = tmp_path / "prefixes.jsonl"  # same number of prefixes, different contents
    _write_jsonl(prefixes, [{"prefix": SECRET[:k]} for k in (28, 32, 36, 40, 44, 48)])
    _run(config)
    assert len(generated_dirs() - before) == 1
