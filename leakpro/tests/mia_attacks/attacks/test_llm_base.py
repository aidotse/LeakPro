#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for leakpro.attacks.mia_attacks.llm.abstract_llm_mia.

Tests cover:
- rank_top(): forced and non-finite rows end up strictly above every ordinary row, ordered by the
  tiebreak, all outputs finite, ordinary rows untouched; all-forced and none-forced cases
- ReferenceModelConfig / LLMAttackConfig validation (extra="forbid", source literal)
- load_reference(): `self` wraps the handler's target; `random_init` re-initialises the blueprint
  so it differs from the target; `pretrained` without a path raises
- AbstractLLMMIA against a fake handler + pure-torch tiny LM: opts out of aux handlers, wraps the
  target as CausalLMModel, membership_labels match the audit split, evidence() rows align with
  indices and references are scored on the same rows
"""

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
from leakpro.signals.token_evidence import CausalLMModel
from leakpro.tests.signals.test_token_evidence import VOCAB, TinyCausalLM

# --------------------------------------------------------------------------------------- rank_top


def test_rank_top_leaves_ordinary_rows_alone_when_nothing_forced() -> None:
    """No forced or non-finite rows → identity."""
    s = np.array([0.3, -1.0, 2.5])
    np.testing.assert_array_equal(rank_top(s, np.zeros(3, bool), s), s)


def test_rank_top_places_forced_rows_above_all_others_ordered_by_tiebreak() -> None:
    """Forced rows beat the max ordinary score; among themselves larger tiebreak ranks higher."""
    s = np.array([0.5, 9.0, 0.1, 2.0, np.inf, np.nan])
    force = np.array([True, False, False, True, False, False])
    tb = np.array([1.0, 0.0, 0.0, 5.0, 3.0, 2.0])
    out = rank_top(s, force, tb)
    assert np.all(np.isfinite(out))
    forced_rows = force | ~np.isfinite(s)  # rows 0, 3, 4, 5
    assert out[forced_rows].min() > out[~forced_rows].max()
    np.testing.assert_array_equal(out[~forced_rows], s[~forced_rows])
    # tiebreak order among forced: row3 (5) > row4 (3) > row5 (2) > row0 (1)
    assert out[3] > out[4] > out[5] > out[0]


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


def test_require_references_gives_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An attack that needs a reference but has none fails with a message naming the config key."""
    monkeypatch.setattr("leakpro.signals.token_evidence.get_device", lambda: torch.device("cpu"))
    attack = _Probe(_fake_handler(), {})
    with pytest.raises(ValueError, match="references"):
        attack._require_references(1)
