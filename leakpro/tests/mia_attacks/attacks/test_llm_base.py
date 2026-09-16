#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for leakpro.attacks.mia_attacks.llm.abstract_llm_mia.

Tests cover:
- rank_top(): forced and +inf rows end up strictly above every ordinary row, ordered by the
  tiebreak; -inf rows go below every ordinary row; unforced nan raises; all outputs finite,
  ordinary rows untouched; all-forced and none-forced cases
- ReferenceModelConfig / LLMAttackConfig validation (extra="forbid", source literal)
- load_reference(): `self` wraps the handler's target; `random_init` re-initialises the blueprint
  so it differs from the target; `pretrained` without a path raises
- AbstractLLMMIA against a fake handler + pure-torch tiny LM: opts out of aux handlers, wraps the
  target as CausalLMModel, membership_labels match the audit split, evidence() rows align with
  indices and references are scored on the same rows
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from leakpro.attacks.mia_attacks.llm.abstract_llm_mia import (
    AbstractLLMMIA,
    LLMAttackConfig,
    ReferenceModelConfig,
    load_reference,
    rank_top,
)
from leakpro.reporting.mia_result import MIAResult
from leakpro.signals.token_evidence import CausalLMModel
from leakpro.tests.signals.test_token_evidence import VOCAB, TinyCausalLM

# --------------------------------------------------------------------------------------- rank_top


def test_rank_top_leaves_ordinary_rows_alone_when_nothing_forced() -> None:
    """No forced or non-finite rows → identity."""
    s = np.array([0.3, -1.0, 2.5])
    np.testing.assert_array_equal(rank_top(s, np.zeros(3, bool), s), s)


def test_rank_top_places_forced_rows_above_all_others_ordered_by_tiebreak() -> None:
    """Forced rows and +inf beat the max ordinary score; among themselves larger tiebreak ranks higher."""
    s = np.array([0.5, 9.0, 0.1, 2.0, np.inf, np.nan])
    force = np.array([True, False, False, True, False, True])   # the nan row must be explicitly forced
    tb = np.array([1.0, 0.0, 0.0, 5.0, 3.0, 2.0])
    out = rank_top(s, force, tb)
    assert np.all(np.isfinite(out))
    forced_rows = force | np.isposinf(s)  # rows 0, 3, 4, 5
    assert out[forced_rows].min() > out[~forced_rows].max()
    np.testing.assert_array_equal(out[~forced_rows], s[~forced_rows])
    # tiebreak order among forced: row3 (5) > row4 (3) > row5 (2) > row0 (1)
    assert out[3] > out[4] > out[5] > out[0]


def test_rank_top_neg_inf_goes_to_bottom_and_only_pos_inf_is_forced() -> None:
    """-inf is the weakest signal (log(P/N) with P == 0) and must rank below every ordinary row."""
    s = np.array([0.5, -np.inf, 2.0, np.inf, -np.inf])
    out = rank_top(s, np.zeros(5, bool), np.zeros(5))
    assert np.all(np.isfinite(out))
    assert out[3] > max(out[0], out[2])           # +inf forced to the top
    assert out[1] < min(out[0], out[2])           # -inf below the ordinary rows
    assert out[1] == out[4]                       # ties preserved
    np.testing.assert_array_equal(out[[0, 2]], s[[0, 2]])


def test_rank_top_rejects_nan_outside_forced_rows() -> None:
    """A nan in an ordinary row is a caller bug, not something to silently rank."""
    with pytest.raises(ValueError, match="nan"):
        rank_top(np.array([1.0, np.nan]), np.array([False, False]), np.zeros(2))
    # nan in a forced row is fine
    out = rank_top(np.array([1.0, np.nan]), np.array([False, True]), np.zeros(2))
    assert np.all(np.isfinite(out))
    assert out[1] > out[0]


def test_rank_top_all_forced_is_finite_and_ordered() -> None:
    """Every row forced (e.g. every sequence memorised) still yields finite, tiebreak-ordered scores."""
    s = np.full(4, np.inf)
    out = rank_top(s, np.ones(4, bool), np.array([2.0, 0.5, 3.0, 1.0]))
    assert np.all(np.isfinite(out))
    assert out[2] > out[0] > out[3] > out[1]


# ---------------------------------------------------------------------------------------- configs


def test_reference_config_validation() -> None:
    """Unknown sources and stray keys are rejected; the memo key is canonical."""
    cfg = ReferenceModelConfig(source="pretrained", pretrained_name_or_path="gpt2")
    assert cfg.key() == ReferenceModelConfig(pretrained_name_or_path="gpt2", source="pretrained").key()
    with pytest.raises(ValidationError):
        ReferenceModelConfig(source="distilled")
    with pytest.raises(ValidationError):
        ReferenceModelConfig(folder="x")
    with pytest.raises(ValidationError):
        LLMAttackConfig(batch_size=0)
    assert LLMAttackConfig().references == []


# ---------------------------------------------------------------------------------- fake handler


class _Population:
    """Object-dtype ragged population, as MIAHandler.get_dataset expects (indexable by ndarray)."""

    def __init__(self, seqs: list) -> None:
        self.data = np.empty(len(seqs), dtype=object)
        self.targets = np.empty(len(seqs), dtype=object)
        for i, s in enumerate(seqs):
            self.data[i] = s
            self.targets[i] = s

    def __len__(self) -> int:
        return len(self.data)


class _ListDataset(torch.utils.data.Dataset):
    """What a text UserDataset looks like once sliced: (ids, ids) per row."""

    def __init__(self, data: np.ndarray, targets: np.ndarray) -> None:
        self.data, self.targets = data, targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> tuple:
        return self.data[i], self.targets[i]


def _fake_handler(n_train: int = 4, n_test: int = 3, n_extra: int = 2, seed: int = 0) -> SimpleNamespace:
    """Enough of MIAHandler for AbstractMIA.__init__, evidence() and load_reference()."""
    torch.manual_seed(seed)
    n = n_train + n_test + n_extra
    seqs = [torch.randint(0, VOCAB, (int(length),)) for length in torch.randint(3, 9, (n,))]
    population = _Population(seqs)
    target = TinyCausalLM()
    handler = SimpleNamespace(
        population=population,
        population_size=n,
        target_model=target,
        train_indices=np.arange(n_train),
        test_indices=np.arange(n_train, n_train + n_test),
        target_model_blueprint=TinyCausalLM,
        target_model_metadata=SimpleNamespace(init_params={}),
        configs=SimpleNamespace(audit=SimpleNamespace(random_seed=1234)),
    )
    handler.get_dataset = lambda idx, params=None: _ListDataset(population.data[idx], population.targets[idx])  # noqa: ARG005
    handler.get_criterion = lambda: None
    handler.get_target_replica = lambda: (TinyCausalLM(), None, None)
    return handler


class _Cfg(LLMAttackConfig):
    """Config for the probe attack."""


class _Probe(AbstractLLMMIA):
    """Minimal concrete attack exposing the base machinery."""

    AttackConfig = _Cfg

    def description(self) -> dict:
        """Four-key description."""
        return {"title_str": "probe", "reference": "", "summary": "", "detailed": ""}

    def prepare_attack(self) -> None:
        """Nothing."""

    def run_attack(self) -> None:
        """Nothing."""


# --------------------------------------------------------------------------------- load_reference


def test_load_reference_self_wraps_target_module() -> None:
    """source=self returns the handler's own target behind a CausalLMModel."""
    handler = _fake_handler()
    ref = load_reference(ReferenceModelConfig(source="self"), handler, torch.device("cpu"))
    assert isinstance(ref, CausalLMModel)
    assert ref.model_obj is handler.target_model
    assert not any(p.requires_grad for p in ref.model_obj.parameters())


def test_load_reference_random_init_differs_from_target_and_is_deterministic() -> None:
    """source=random_init builds a fresh blueprint whose weights differ from the target, reproducibly."""
    handler = _fake_handler()
    cfg = ReferenceModelConfig(source="random_init")
    a = load_reference(cfg, handler, torch.device("cpu"))
    b = load_reference(cfg, handler, torch.device("cpu"))
    assert a.model_obj is not handler.target_model
    assert not torch.equal(a.model_obj.head.weight, handler.target_model.head.weight)
    assert torch.equal(a.model_obj.head.weight, b.model_obj.head.weight)


class _Conv1DLike(torch.nn.Module):
    """Mimics HuggingFace's Conv1D: owns parameters, defines no reset_parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, 3))
        self.bias = torch.nn.Parameter(torch.ones(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias


def test_gpt2_style_init_covers_every_parameter_at_one_scale() -> None:
    """Conv1D (no reset_parameters), Embedding (torch default N(0,1)) and LayerNorm all land at GPT-2's scale."""
    from leakpro.attacks.mia_attacks.llm.abstract_llm_mia import gpt2_style_init_

    model = torch.nn.Sequential(torch.nn.Embedding(5000, 64), torch.nn.Linear(64, 64), _Conv1DLike(), torch.nn.LayerNorm(3))
    with torch.no_grad():
        model[3].weight.fill_(7.0)
        model[3].bias.fill_(7.0)
    torch.manual_seed(0)
    gpt2_style_init_(model)
    emb, lin, conv, ln = model
    assert abs(emb.weight.std().item() - 0.02) < 0.003        # torch's own reset would give ~1.0
    assert abs(lin.weight.std().item() - 0.02) < 0.005
    assert not torch.equal(conv.weight, torch.ones(4, 3))
    assert abs(conv.weight.std().item() - 0.02) < 0.01
    assert torch.equal(conv.bias, torch.zeros(3))
    assert torch.equal(lin.bias, torch.zeros(64))
    assert torch.equal(ln.weight, torch.ones(3))
    assert torch.equal(ln.bias, torch.zeros(3))


def test_load_reference_pretrained_requires_path() -> None:
    """source=pretrained without a checkpoint name is a config error, not a transformers error."""
    with pytest.raises(ValueError, match="pretrained_name_or_path"):
        load_reference(ReferenceModelConfig(source="pretrained"), _fake_handler(), torch.device("cpu"))


# ---------------------------------------------------------------------------------- AbstractLLMMIA


def test_base_attack_opts_out_and_wraps_target_as_causal_lm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flags are False, the shared target wrapper is a CausalLMModel, labels follow the split."""
    monkeypatch.setattr("leakpro.signals.token_evidence.get_device", lambda: torch.device("cpu"))
    handler = _fake_handler(n_train=4, n_test=3)
    attack = _Probe(handler, {"batch_size": 2})
    assert _Probe.requires_shadow_models is False
    assert _Probe.requires_distillation_models is False
    assert isinstance(attack.target_model, CausalLMModel)
    assert attack.batch_size == 2
    np.testing.assert_array_equal(attack.membership_labels, [1, 1, 1, 1, 0, 0, 0])
    assert len(attack.description()) == 4


def test_evidence_rows_align_with_indices_and_references(monkeypatch: pytest.MonkeyPatch) -> None:
    """evidence() scores exactly the requested rows, in order, for target and every reference."""
    monkeypatch.setattr("leakpro.signals.token_evidence.get_device", lambda: torch.device("cpu"))
    handler = _fake_handler(n_train=4, n_test=3, n_extra=2)
    attack = _Probe(handler, {"batch_size": 3, "references": [{"source": "self"}, {"source": "random_init"}]})

    indices = attack.audit_dataset["data"]
    ev = attack.evidence(indices)

    assert ev.target.num_sequences == len(indices) == 7
    np.testing.assert_array_equal(ev.indices, indices)
    expected_lengths = [len(handler.population.data[i]) - 1 for i in indices]
    np.testing.assert_array_equal(ev.target.lengths, expected_lengths)

    assert len(ev.references) == 2
    # self-reference is the target: identical evidence, delta == 0 everywhere
    np.testing.assert_allclose(ev.ref(0).logprob, ev.target.logprob, rtol=1e-6, atol=1e-6)
    # random-init reference: same rows, same masks, different numbers
    np.testing.assert_array_equal(ev.ref(1).mask, ev.target.mask)
    assert not np.allclose(ev.ref(1).logprob, ev.target.logprob)


def test_evidence_is_memoised_per_handler_across_attacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two attacks on the same handler and indices share forward passes; different indices do not."""
    monkeypatch.setattr("leakpro.signals.token_evidence.get_device", lambda: torch.device("cpu"))
    calls = {"n": 0}
    original = CausalLMModel.evidence_from_loader

    def counting(self, loader, request=None):  # noqa: ANN001, ANN202
        calls["n"] += 1
        return original(self, loader, request)

    monkeypatch.setattr(CausalLMModel, "evidence_from_loader", counting)
    handler = _fake_handler(n_train=4, n_test=3)
    a = _Probe(handler, {"references": [{"source": "random_init"}]})
    b = _Probe(handler, {"references": [{"source": "random_init"}]})
    idx = a.audit_dataset["data"]

    ev_a = a.evidence(idx)
    assert calls["n"] == 2                                   # target + reference
    ev_b = b.evidence(idx)
    assert calls["n"] == 2                                   # both reused
    assert ev_b.target is ev_a.target
    b.evidence(idx[:3])
    assert calls["n"] == 4                                   # different rows → recomputed


def test_self_reference_reuses_target_evidence_without_a_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """source=self is the target: delta ≡ 0 must not cost a second forward pass."""
    monkeypatch.setattr("leakpro.signals.token_evidence.get_device", lambda: torch.device("cpu"))
    calls = {"n": 0}
    original = CausalLMModel.evidence_from_loader

    def counting(self, loader, request=None):  # noqa: ANN001, ANN202
        calls["n"] += 1
        return original(self, loader, request)

    monkeypatch.setattr(CausalLMModel, "evidence_from_loader", counting)
    attack = _Probe(_fake_handler(), {"references": [{"source": "self"}]})
    ev = attack.evidence(attack.audit_dataset["data"])
    assert calls["n"] == 1
    assert ev.ref(0) is ev.target


def test_attacks_on_different_handlers_do_not_share_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """The #462 reproduction: building a second attack on another handler must not touch the first."""
    monkeypatch.setattr("leakpro.signals.token_evidence.get_device", lambda: torch.device("cpu"))
    h1 = _fake_handler(n_train=4, n_test=3, n_extra=1, seed=1)
    h2 = _fake_handler(n_train=2, n_test=5, n_extra=3, seed=2)
    a = _Probe(h1, {"references": [{"source": "random_init"}]})
    b = _Probe(h2, {})
    assert a.handler is h1
    assert b.handler is h2
    assert a.population is h1.population
    assert a.population_size == 8
    assert b.population_size == 10
    np.testing.assert_array_equal(a.membership_labels, [1, 1, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(b.membership_labels, [1, 1, 0, 0, 0, 0, 0])
    a.evidence(a.audit_dataset["data"])
    assert hasattr(h1, "_llm_run_memo")
    assert not hasattr(h2, "_llm_run_memo")                  # the memo landed on the right handler


def test_llm_attack_id_uses_cheap_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM attacks must not hash the full state dict; the id still separates configs and targets."""
    import leakpro.attacks.mia_attacks.llm.abstract_llm_mia as base_module
    import leakpro.utils.save_load as save_load

    monkeypatch.setattr("leakpro.signals.token_evidence.get_device", lambda: torch.device("cpu"))

    def boom(model):  # noqa: ANN001, ANN202, ARG001
        raise AssertionError("hash_model must not run for LLM attacks")

    monkeypatch.setattr(save_load, "hash_model", boom)
    # hash_attack's default `model_hasher=hash_model` was bound at import; the LLM override passes its own.
    handler = _fake_handler()
    a = _Probe(handler, {"batch_size": 2})
    b = _Probe(handler, {"batch_size": 4})
    c = _Probe(_fake_handler(seed=9), {"batch_size": 2})
    assert len(a.attack_id) == 64
    assert a.attack_id != b.attack_id                        # config differs
    assert a.attack_id != c.attack_id                        # target weights differ
    assert base_module.fingerprint_model(handler.target_model) == base_module.fingerprint_model(handler.target_model)


def test_require_references_gives_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An attack that needs a reference but has none fails with a message naming the config key."""
    monkeypatch.setattr("leakpro.signals.token_evidence.get_device", lambda: torch.device("cpu"))
    attack = _Probe(_fake_handler(), {})
    with pytest.raises(ValueError, match="references"):
        attack._require_references(1)


# --------------------------------------------------------------------------------- max_samples


def test_audit_indices_and_labels_defaults_to_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    """max_samples unset (the default) audits the whole set, unchanged from membership_labels."""
    monkeypatch.setattr("leakpro.signals.token_evidence.get_device", lambda: torch.device("cpu"))
    handler = _fake_handler(n_train=4, n_test=3)
    attack = _Probe(handler, {})
    indices, labels = attack._audit_indices_and_labels()
    np.testing.assert_array_equal(indices, handler.train_indices.tolist() + handler.test_indices.tolist())
    np.testing.assert_array_equal(labels, attack.membership_labels)


def test_max_samples_is_stratified_deterministic_and_smaller_than_max_samples_is_a_no_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """max_samples keeps the member/non-member ratio, is seeded (reproducible), and never grows the set."""
    monkeypatch.setattr("leakpro.signals.token_evidence.get_device", lambda: torch.device("cpu"))
    handler = _fake_handler(n_train=40, n_test=30, n_extra=5)
    attack = _Probe(handler, {"max_samples": 20})
    indices, labels = attack._audit_indices_and_labels()
    assert len(indices) == 20
    assert len(labels) == 20
    assert 0 < labels.sum() < 20  # both classes present -- stratified, not all-members or all-non-members

    indices2, labels2 = attack._audit_indices_and_labels()
    np.testing.assert_array_equal(indices, indices2)  # seeded by handler.configs.audit.random_seed -> reproducible
    np.testing.assert_array_equal(labels, labels2)

    # max_samples >= population size is a no-op, same as unset
    attack_noop = _Probe(handler, {"max_samples": 10_000})
    indices3, labels3 = attack_noop._audit_indices_and_labels()
    assert len(indices3) == 70


# ------------------------------------------------------------------------------ n_bootstrap_samples


def test_bootstrap_metrics_uses_mia_result_definitions_and_reports_ci() -> None:
    """Bootstrap AUC/TPR come from MIAResult itself: same keys and definitions as the point estimate."""
    rng = np.random.RandomState(0)
    labels = np.array([1] * 50 + [0] * 50)
    scores = np.concatenate([rng.normal(1.0, 1.0, 50), rng.normal(0.0, 1.0, 50)])
    point = MIAResult.from_full_scores(true_membership=labels, signal_values=scores, result_name="p")
    block = MIAResult.bootstrap_metrics(labels, scores, n_resamples=20, seed=0)
    assert block["n_bootstrap_samples"] == 20
    assert block["n_used"] == 20
    assert 0.5 < block["roc_auc"]["mean"] <= 1.0
    assert block["roc_auc"]["ci_low"] <= block["roc_auc"]["mean"] <= block["roc_auc"]["ci_high"]
    assert set(block["fixed_fpr_table"]) == set(point.fixed_fpr_table)          # identical metric keys
    # the point estimate lies inside (or at the edge of) its own bootstrap interval
    assert block["roc_auc"]["ci_low"] - 0.05 <= point.roc_auc <= block["roc_auc"]["ci_high"] + 0.05
    # a one-class resample contributes nothing rather than crashing
    degenerate = MIAResult.bootstrap_metrics(np.ones(6), np.arange(6.0), n_resamples=3, seed=0)
    assert degenerate["n_used"] == 0
    assert degenerate["roc_auc"]["mean"] is None


def test_bootstrap_is_opt_in_and_round_trips_through_save_and_load(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """n_bootstrap_samples unset attaches nothing; set, it is persisted in the result JSON and restored by load()."""
    monkeypatch.setattr("leakpro.signals.token_evidence.get_device", lambda: torch.device("cpu"))
    handler = _fake_handler(n_train=6, n_test=6, n_extra=2)
    labels, scores = np.array([1, 0, 1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])

    plain = _Probe(handler, {})
    result = MIAResult.from_full_scores(true_membership=labels, signal_values=scores, result_name="probe")
    assert plain._attach_bootstrap_if_configured(result, labels, scores) is result
    assert getattr(result, "bootstrap", None) is None
    assert result.result.bootstrap is None

    boot = _Probe(handler, {"n_bootstrap_samples": 5})
    result2 = MIAResult.from_full_scores(true_membership=labels, signal_values=scores, result_name="probe")
    boot._attach_bootstrap_if_configured(result2, labels, scores)
    assert result2.bootstrap["n_bootstrap_samples"] == 5
    assert result2.result.bootstrap == result2.bootstrap                          # in the schema object

    result2.save(attack_obj=boot, output_dir=str(tmp_path))
    loaded = MIAResult.load(str(tmp_path / "data_objects" / f"{boot.attack_id}.json"))
    assert loaded.bootstrap == result2.bootstrap                                   # survives the JSON round trip
