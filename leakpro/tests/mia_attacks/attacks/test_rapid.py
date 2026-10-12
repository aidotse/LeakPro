#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""RAPID scoring, separation, cache reuse, and public audit regressions."""

from copy import deepcopy
from inspect import getfile
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence, Union
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import yaml
from pydantic import ValidationError
from scipy.special import logsumexp
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import RandAugment, RandomHorizontalFlip

from examples.mia.cifar.cifar_data_handler import CifarDataHandler
from leakpro import LeakPro
from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.mia_attacks.rapid import AttackRAPID
from leakpro.attacks.mia_attacks.rmia import AttackRMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.signals.signal import ModelLogits
from leakpro.signals.signal_extractor import PytorchModel
from leakpro.tests.mia_attacks.attacks.test_all_attacks_end_to_end import (
    TinyImageInputHandler,
    TinyImageTargetModel,
    _build_tiny_image_population,
    _create_target_artifacts,
)


class QueryModel:
    """Provide distinct logits on successive queries, preserving sample order."""

    def __init__(self, queries: Sequence[Union[np.ndarray, list]]) -> None:
        self.queries = queries
        self.calls = 0

    def get_logits(self, data: torch.Tensor) -> np.ndarray:
        """Return the next query's logits without changing row order."""
        logits = self.queries[self.calls % len(self.queries)]
        self.calls += 1
        return np.asarray(logits, dtype=np.float64)[data[:, 0].numpy()]


def loss_attack(labels: Union[np.ndarray, list], batch_size: Optional[int] = None, drop_last: bool = False) -> AttackRAPID:
    """Build the loss-query boundary without initializing global model handlers."""
    attack = object.__new__(AttackRAPID)
    attack.query_num = 2
    attack.augment_strength = None
    attack.source_attack = None
    attack.batch_size = len(labels) if batch_size is None else batch_size
    dataset = TensorDataset(torch.arange(len(labels)).reshape(-1, 1), torch.as_tensor(labels))
    attack.handler = SimpleNamespace(get_dataloader=lambda *_args, **kwargs: DataLoader(
        dataset, batch_size=kwargs["batch_size"], shuffle=kwargs["shuffle"], drop_last=drop_last
    ))
    return attack


def test_features_match_independent_loss_oracle() -> None:
    """Compare query and reference averaging with an independent SciPy oracle."""
    labels = np.array([1, 0, 2])
    original_logits = [np.array([[0, 3, -2], [2, 0, 1], [4, 3, 1]]),
                       np.array([[1, 1, 0], [-1, 2, 0], [0, 0, 0]])]
    reference_logits = [np.array([[0, 0, 0], [0, 1, -1], [0, 1, 2]]),
                        np.array([[3, 2, 1], [1, 0, 3], [1, 2, -2]])]
    attack = loss_attack(labels)
    model = QueryModel(original_logits)
    references = [QueryModel([logits]) for logits in reference_logits]
    features = attack._get_mia_features(model, references, np.array([17, 4, 29]))
    original_loss = np.mean([logsumexp(logits, axis=1) - logits[np.arange(3), labels]
                             for logits in original_logits], axis=0)
    reference_loss = np.mean([logsumexp(logits, axis=1) - logits[np.arange(3), labels]
                              for logits in reference_logits], axis=0)
    np.testing.assert_allclose(features[:, 0], original_loss, rtol=1e-6)
    np.testing.assert_allclose(features[:, 1], original_loss - reference_loss, rtol=1e-6, atol=1e-7)
    assert model.calls == 2
    assert all(reference.calls == 2 for reference in references)


def test_binary_loss_is_stable_for_extreme_logits() -> None:
    """Keep binary losses finite when sigmoid probabilities would saturate."""
    logits = np.array([[1000.0], [-1000.0], [0.0]])
    labels = np.array([1.0, 1.0, 0.0])
    attack = loss_attack(labels)
    actual = attack._get_mean_loss(QueryModel([logits]), np.arange(3))
    np.testing.assert_allclose(actual, [0.0, 1000.0, np.log(2)])


@pytest.mark.parametrize(("labels", "logits", "match"), [
    ([0], [[np.nan, 0]], "non-finite model logits"),
    ([0.5], [[0, 1]], "integer class labels"),
    ([2], [[0, 1]], "integer class labels"),
    ([2], [[0]], "binary labels"),
    ([0], [[[0, 1]]], "classification logits"),
])
def test_invalid_model_outputs_fail(labels: list, logits: list, match: str) -> None:
    """Reject unsupported output shapes, labels, and non-finite logits."""
    with pytest.raises(ValueError, match=match):
        loss_attack(labels)._get_mean_loss(QueryModel([logits]), np.array([0]))


def test_dropped_query_rows_fail() -> None:
    """Reject dropped rows that would misalign scores and audit membership."""
    attack = loss_attack([0, 1, 0], batch_size=2, drop_last=True)
    with pytest.raises(ValueError, match="disable drop_last"):
        attack._get_mean_loss(QueryModel([[[0, 1], [1, 0]]]), np.arange(3))


@pytest.mark.parametrize(("shuffle", "num_workers", "in_order", "augment_strength"), [
    (False, 0, True, "easy"), (True, 0, True, "easy"),
    (False, 2, False, "easy"), (False, 0, False, "easy"),
    (False, 0, True, "none"), (False, 0, True, "medium"), (False, 0, True, "strong"),
])
def test_cifar_query_augmentation_preserves_normalization_and_order(
    monkeypatch: pytest.MonkeyPatch, shuffle: bool, num_workers: int, in_order: bool, augment_strength: str
) -> None:
    """Use the real CIFAR transforms and reject shuffled query loading."""
    torch.manual_seed(23)
    data = torch.rand(6, 3, 32, 32)
    labels = torch.arange(6) % 3
    population = CifarDataHandler.UserDataset(data, labels)
    monkeypatch.setattr(AbstractMIA, "population", population)
    attack = loss_attack(labels.numpy())
    attack.augment_strength = augment_strength
    # Force a uint8-only operation so strong augmentation cannot pass by chance.
    monkeypatch.setattr(RandAugment, "_augmentation_space", lambda *_args, **_kwargs: {"Equalize": (torch.tensor(0.0), False)})
    query_loader = Mock(side_effect=lambda indices, **kwargs: DataLoader(
        CifarDataHandler.UserDataset(data[indices], labels[indices], **kwargs["params"]),
        batch_size=kwargs["batch_size"], shuffle=shuffle, num_workers=num_workers, in_order=in_order
    ))
    attack.handler = SimpleNamespace(get_dataloader=query_loader)
    model = PytorchModel(nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 3)), nn.CrossEntropyLoss())
    indices = np.array([5, 1, 3])
    if shuffle or (num_workers > 0 and not in_order):
        with pytest.raises(ValueError, match="sequential query dataloader"):
            attack._get_mean_loss(model, indices)
    else:
        losses = attack._get_mean_loss(model, indices)
        assert losses.shape == (3,)
        assert np.isfinite(losses).all()
    params = query_loader.call_args.kwargs["params"]
    assert params["mean"] is population.mean
    assert params["std"] is population.std
    assert params["augment_strength"] == augment_strength
    assert "augment" not in params
    assert "erase_post_norm" not in params
    assert population.augment is None


@pytest.mark.parametrize("config", [
    {"num_reference_models": 0}, {"query_num": 0}, {"shadow_data_fraction": 0},
    {"shadow_data_fraction": 1}, {"training_data_fraction": 0}, {"training_data_fraction": 1},
    {"batch_size": 0}, {"mia_classifier_epochs": 0}, {"learning_rate": 0},
    {"learning_rate": float("nan")}, {"weight_decay": float("inf")},
    {"random_seed": -1}, {"random_seed": 2**32}, {"augment_strength": "invalid"},
    {"online": True}, {"learning_rtae": 0.1},
    {"use_attack": "unknown"}, {"use_attack_config": {"gamma": 1.0}},
    {"use_attack_output": "logits"}, {"use_attack": "lira", "use_attack_output": "invalid"},
    {"use_attack": "rmia", "use_attack_config": {"num_shadow_models": 3}},
])
def test_invalid_config_is_rejected(config: dict) -> None:
    """Reject invalid attack parameters at the configuration boundary."""
    with pytest.raises(ValidationError):
        AttackRAPID.AttackConfig(**config)


@pytest.mark.parametrize(("value", "expected"), [
    (None, None), ("None", None), ("RAPID", None), ("none", None), ("rapid", None),
    (" RaPiD ", None), ("LiRA", "lira"), ("RMIA", "rmia"),
])
def test_use_attack_aliases_are_normalized(value: Optional[str], expected: Optional[str]) -> None:
    """Equivalent selectors share canonical config values and cache identities."""
    config = AttackRAPID.AttackConfig(use_attack=value)
    assert config.use_attack == expected
    if expected is None:
        assert config.model_dump() == AttackRAPID.AttackConfig().model_dump()


def test_yaml_none_and_rapid_fallbacks() -> None:
    """Unquoted YAML None is a string and must still select ordinary RAPID."""
    for value in ("None", "RAPID", "null"):
        config = AttackRAPID.AttackConfig(**yaml.safe_load(f"use_attack: {value}"))
        assert config.use_attack is None


def create_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, num_classes: int = 3,
                 audit_all: bool = False, stochastic: bool = False, shadow_data_fraction: float = 0.5,
                 use_attack: Optional[str] = None, use_attack_config: Optional[dict] = None,
                 use_attack_output: str = "scores") -> LeakPro:
    """Create real YAML, target metadata, and a nonconsecutive audit split."""
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(23)
    population = _build_tiny_image_population()
    if stochastic:
        population.augment = RandomHorizontalFlip()
    population.targets = population.targets % num_classes if num_classes > 1 else population.targets % 2
    model = TinyImageTargetModel(num_classes=num_classes)
    criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
    train_indices = [17, 4, 29, 5, 1]
    test_indices = [33, 9, 48, 11, 2]
    if audit_all:
        train_indices, test_indices = list(range(36)), list(range(36, 72))
    data_path, target_dir = _create_target_artifacts(
        tmp_path, population, model, criterion, torch.optim.SGD(model.parameters(), lr=0.01),
        "rapid_test", train_indices=train_indices, test_indices=test_indices
    )
    config = {
        "audit": {"random_seed": 19, "attack_type": "mia", "data_modality": "image",
                  "output_dir": str(tmp_path / "output"), "attack_list": [
                      {"attack": "rapid", "num_reference_models": 2, "query_num": 2,
                       "mia_classifier_epochs": 3, "batch_size": 7, "shadow_data_fraction": shadow_data_fraction,
                       "use_attack": use_attack, "use_attack_config": use_attack_config or {},
                       "use_attack_output": use_attack_output}]},
        "target": {"module_path": getfile(TinyImageTargetModel),
                   "model_class": "TinyImageTargetModel", "target_folder": str(target_dir), "data_path": str(data_path)},
        "shadow_model": {"epochs": 1},
    }
    path = tmp_path / "audit.yaml"
    path.write_text(yaml.safe_dump(config))
    return LeakPro(TinyImageInputHandler, str(path))


@pytest.mark.parametrize("num_classes", [2, 3])
@pytest.mark.parametrize("stochastic", [False, True])
def test_public_audit_preserves_order_and_reuses_disjoint_cached_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, num_classes: int, stochastic: bool
) -> None:
    """Verify YAML dispatch, reporting, pool separation, and exact cached replay."""
    leakpro = create_audit(tmp_path, monkeypatch, num_classes=num_classes, stochastic=stochastic)
    attack = leakpro.attack_scheduler.attacks[0]
    assert attack.random_seed == 19
    with pytest.raises(RuntimeError, match="prepare_attack"):
        attack.run_attack()
    result = leakpro.run_audit(create_pdf=False, use_optuna=False)[0]
    assert len(result.signal_values) == 10
    np.testing.assert_array_equal(result.true, [1] * 5 + [0] * 5)
    np.testing.assert_array_equal(attack.audit_dataset["data"], [17, 4, 29, 5, 1, 33, 9, 48, 11, 2])
    assert np.isfinite(result.signal_values).all()
    assert np.all((result.signal_values >= 0) & (result.signal_values <= 1))
    assert (tmp_path / "output" / "results" / result.id / "result.txt").exists()
    assert result.metadata == attack.configs.model_dump()

    shadow_handler = ShadowModelHandler()
    assert set(attack.shadow_population).isdisjoint(attack.reference_population)
    assert set(attack.shadow_population).isdisjoint(attack.audit_dataset["data"])
    assert set(attack.reference_population).isdisjoint(attack.audit_dataset["data"])
    for ids, pool in [(attack.shadow_model_indices, attack.shadow_population),
                      (attack.reference_model_indices, attack.reference_population)]:
        for metadata in shadow_handler.get_shadow_model_metadata(ids):
            assert set(metadata.train_indices).issubset(pool)
    assert set(attack.shadow_model_indices).isdisjoint(attack.reference_model_indices)
    metadata_before = sorted((tmp_path / "output" / "attack_objects" / "shadow_model").glob("metadata_*.pkl"))
    scores_before = result.signal_values.copy()
    model_before = deepcopy(attack.scoring_model.state_dict())
    monkeypatch.setattr(leakpro.handler, "train", lambda *_args, **_kwargs: pytest.fail("Cached audit retrained a model"))
    attack.prepare_attack()
    np.testing.assert_array_equal(attack.run_attack().signal_values, scores_before)
    for key, value in attack.scoring_model.state_dict().items():
        torch.testing.assert_close(value, model_before[key], rtol=0, atol=0)
    assert sorted((tmp_path / "output" / "attack_objects" / "shadow_model").glob("metadata_*.pkl")) == metadata_before


@pytest.mark.parametrize("fallback", ["None", "RAPID"])
def test_fallback_alias_preserves_public_audit_and_cached_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fallback: str
) -> None:
    """Aliases run ordinary RAPID with the same identity and cached output as an omitted selector."""
    audit = create_audit(tmp_path, monkeypatch)
    ordinary = audit.attack_scheduler.attacks[0]
    ordinary.prepare_attack()
    expected = ordinary.run_attack()
    alias = AttackRAPID(audit.handler, {**ordinary.configs.model_dump(), "use_attack": fallback})
    assert alias.source_attack is None
    assert alias.attack_id == ordinary.attack_id
    monkeypatch.setattr(audit.handler, "train", lambda *_args, **_kwargs: pytest.fail("Alias retrained cached models"))
    alias.prepare_attack()
    actual = alias.run_attack()
    assert actual.id == expected.id
    np.testing.assert_array_equal(actual.signal_values, expected.signal_values)


@pytest.mark.parametrize(("audit_all", "shadow_data_fraction"), [(True, 0.5), (False, 0.99)])
def test_insufficient_auxiliary_pool_fails_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, audit_all: bool, shadow_data_fraction: float
) -> None:
    """Check both auxiliary pools before any model is trained."""
    leakpro = create_audit(tmp_path, monkeypatch, audit_all=audit_all, shadow_data_fraction=shadow_data_fraction)
    monkeypatch.setattr(leakpro.handler, "train", lambda *_args, **_kwargs: pytest.fail("Should fail before training"))
    with pytest.raises(ValueError, match="at least 1 point per shadow model"):
        leakpro.run_audit(create_pdf=False, use_optuna=False)


@pytest.mark.parametrize("use_attack", ["lira", "rmia"])
@pytest.mark.parametrize("stochastic", [False, True])
@pytest.mark.parametrize("online", [False, True])
def test_composition_runs_existing_attack_on_shadow_and_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_attack: str, stochastic: bool, online: bool
) -> None:
    """Use real source results, align shadow rows, and preserve the parent audit across cached replay."""
    config = {"var_calculation": "carlini" if online else "fixed"} if use_attack == "lira" else {"gamma": 1.5}
    config["online"] = online
    audit = create_audit(tmp_path, monkeypatch, use_attack=use_attack, use_attack_config=config, stochastic=stochastic)
    attack = audit.attack_scheduler.attacks[0]
    original_indices = attack.audit_dataset["data"].copy()
    source = attack.source_attack
    run = Mock(wraps=source.run_attack)
    monkeypatch.setattr(source, "run_attack", run)
    result = audit.run_audit(create_pdf=False, use_optuna=False)[0]
    assert run.call_count == 2
    assert set(attack.source_results) == {"shadow", "target"}
    assert len(result.signal_values) == len(original_indices)
    assert np.isfinite(result.signal_values).all()
    np.testing.assert_array_equal(attack.audit_dataset["data"], original_indices)
    order = np.concatenate((np.flatnonzero(attack.shadow_membership), np.flatnonzero(attack.shadow_membership == 0)))
    np.testing.assert_allclose(attack.shadow_features[order, 1], attack.source_results["shadow"].signal_values, rtol=1e-6)
    np.testing.assert_allclose(attack.signal_data[:, 1], attack.source_results["target"].signal_values, rtol=1e-6)
    assert result.metadata["use_attack"] == use_attack
    assert result.metadata["use_attack_config"] == source.configs.model_dump(exclude={"num_shadow_models",
                                                                                  "training_data_fraction"})
    first = result.signal_values.copy()
    monkeypatch.setattr(audit.handler, "train", lambda *_args, **_kwargs: pytest.fail("Composition retrained cached models"))
    attack.prepare_attack()
    np.testing.assert_array_equal(attack.run_attack().signal_values, first)
    assert run.call_count == 4


@pytest.mark.parametrize("use_attack", ["lira", "rmia"])
@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("use_attack_output", ["scores", "logits"])
def test_composition_ignores_target_membership_when_building_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_attack: str, online: bool, use_attack_output: str
) -> None:
    """Permuting evaluation membership cannot alter calibrated target scores or the fitted classifier."""
    audit = create_audit(tmp_path, monkeypatch, use_attack=use_attack, use_attack_config={"online": online},
                         use_attack_output=use_attack_output)
    attack = audit.attack_scheduler.attacks[0]
    attack.prepare_attack()
    refs, _ = ShadowModelHandler().get_shadow_models(attack.reference_model_indices)
    indices = attack.audit_dataset["data"].copy()
    before = deepcopy(attack.scoring_model.state_dict())
    torch.manual_seed(99)
    np.random.seed(99)
    first = attack._get_mia_features(attack.target_model, refs, indices, np.array([1] * 5 + [0] * 5))
    torch.manual_seed(99)
    np.random.seed(99)
    swapped = attack._get_mia_features(attack.target_model, refs, indices, np.array([0] * 5 + [1] * 5))
    np.testing.assert_array_equal(first, swapped)
    for name, value in attack.scoring_model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    shadow_features = attack.shadow_features.copy()
    target_features = attack.signal_data.copy()
    original_audit = AbstractMIA.audit_dataset
    monkeypatch.setattr(AbstractMIA, "audit_dataset", {
        "data": indices, "in_members": original_audit["out_members"], "out_members": original_audit["in_members"]
    })
    monkeypatch.setattr(audit.handler, "train", lambda *_args, **_kwargs: pytest.fail("Membership swap retrained a model"))
    attack.prepare_attack()
    np.testing.assert_array_equal(attack.shadow_features, shadow_features)
    np.testing.assert_array_equal(attack.signal_data, target_features)
    for name, value in attack.scoring_model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("use_attack", ["lira", "rmia"])
@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("stochastic", [False, True])
def test_composition_reuses_logits_without_source_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_attack: str, online: bool, stochastic: bool
) -> None:
    """Compare reused source logits with an independent loss oracle, including OUT-only online calibration."""
    audit = create_audit(tmp_path, monkeypatch, use_attack=use_attack, use_attack_config={"online": online},
                         use_attack_output="logits", stochastic=stochastic)
    attack = audit.attack_scheduler.attacks[0]
    monkeypatch.setattr(attack.source_attack, "run_attack", lambda: pytest.fail("Logits mode requested source scores"))
    result = audit.run_audit(create_pdf=False, use_optuna=False)[0]
    assert attack.source_results == {}
    refs, _ = ShadowModelHandler().get_shadow_models(attack.reference_model_indices)
    indices = attack.audit_dataset["data"]
    out_indices = ~ShadowModelHandler().get_in_indices_mask(attack.reference_model_indices, indices).T
    assert out_indices.any(axis=0).all()
    if online:
        assert (~out_indices).any(axis=0).all()
    else:
        assert out_indices.all()
    queried = []
    query = ModelLogits.__call__

    def capture_logits(signal: ModelLogits, *args: object, **kwargs: object) -> np.ndarray:
        values = query(signal, *args, **kwargs)
        queried.append(values)
        return values

    monkeypatch.setattr(ModelLogits, "__call__", capture_logits)
    features = attack._get_mia_features(attack.target_model, refs, indices)
    assert len(queried) == attack.query_num
    labels = audit.handler.get_labels(indices).reshape(-1)
    losses = np.array([logsumexp(values, axis=2) - values[:, np.arange(len(indices)), labels] for values in queried])
    original = losses[:, 0].mean(axis=0)
    reference = (np.where(out_indices, losses[:, 1:], 0.0).sum(axis=1) / out_indices.sum(axis=0)).mean(axis=0)
    np.testing.assert_allclose(features, np.column_stack((original, original - reference)), rtol=1e-6, atol=5e-7)
    assert result.metadata["use_attack_output"] == "logits"
    first = result.signal_values.copy()
    monkeypatch.setattr(audit.handler, "train", lambda *_args, **_kwargs: pytest.fail("Logits replay retrained models"))
    attack.prepare_attack()
    np.testing.assert_array_equal(attack.run_attack().signal_values, first)


@pytest.mark.parametrize("use_attack", ["lira", "rmia"])
@pytest.mark.parametrize("use_attack_output", ["scores", "logits"])
def test_online_source_rejects_incomplete_reference_coverage_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_attack: str, use_attack_output: str
) -> None:
    """Do not drop or fabricate scores for points missing IN or OUT references."""
    audit = create_audit(tmp_path, monkeypatch, use_attack=use_attack, use_attack_config={"online": True},
                         use_attack_output=use_attack_output)
    attack = audit.attack_scheduler.attacks[0]
    attack.training_data_fraction = 0.1
    monkeypatch.setattr(audit.handler, "train", lambda *_args, **_kwargs: pytest.fail("Invalid coverage trained a model"))
    with pytest.raises(ValueError, match="IN and OUT references for every point"):
        attack.prepare_attack()


def test_online_source_rejects_cached_random_references(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing random reference models cannot bypass the online IN/OUT coverage check."""
    audit = create_audit(tmp_path, monkeypatch, use_attack="lira", use_attack_config={"online": True})
    attack = audit.attack_scheduler.attacks[0]
    handler = ShadowModelHandler()
    torch.manual_seed(7)
    np.random.seed(7)
    indices = np.arange(attack.population_size)
    refs = handler.create_shadow_models(2, indices, training_fraction=0.5, sampling_method="random")
    counts = handler.get_in_indices_mask(refs, indices).sum(axis=1)
    assert np.any((counts == 0) | (counts == 2))
    with pytest.raises(ValueError, match="Cached online references must provide IN and OUT"):
        attack.prepare_attack()
    assert attack.scoring_model is None
    with pytest.raises(RuntimeError, match="prepare_attack"):
        attack.run_attack()


def test_empty_rmia_population_fails_without_stale_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject source configurations that cannot produce a usable calibrated score."""
    audit = create_audit(tmp_path, monkeypatch, use_attack="rmia", use_attack_config={"attack_data_fraction": 0.0})
    attack = audit.attack_scheduler.attacks[0]
    with pytest.raises(ValueError, match="at least one population sample"):
        attack.prepare_attack()
    with pytest.raises(RuntimeError, match="prepare_attack"):
        attack.run_attack()


@pytest.mark.parametrize("use_attack", ["lira", "rmia"])
def test_composition_rejects_nonfinite_reference_logits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_attack: str
) -> None:
    """Invalid references must not become finite zero membership scores."""
    audit = create_audit(tmp_path, monkeypatch, use_attack=use_attack)
    attack = audit.attack_scheduler.attacks[0]
    attack.prepare_attack()
    references, _ = ShadowModelHandler().get_shadow_models(attack.reference_model_indices)
    monkeypatch.setattr(references[0], "get_logits", lambda data: np.full((len(data), 3), np.nan))
    with pytest.raises(ValueError, match="non-finite model logits"):
        attack._get_mia_features(attack.target_model, references, attack.audit_dataset["data"], np.array([1] * 5 + [0] * 5))


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_rmia_rejects_invalid_temperature(temperature: float) -> None:
    """Softmax temperature must be positive and finite before any attack runs."""
    with pytest.raises(ValidationError, match="temperature"):
        AttackRMIA.AttackConfig(temperature=temperature)


def test_composition_rejects_numerically_invalid_rmia_ratios(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A positive temperature that overflows model probabilities cannot produce a successful audit."""
    audit = create_audit(tmp_path, monkeypatch, use_attack="rmia", use_attack_config={"temperature": 1e-100})
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"), pytest.raises(ValueError, match="must be finite"):
        audit.run_audit(create_pdf=False, use_optuna=False)


def test_rmia_rejects_undefined_zero_ratio_comparisons(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero over zero comparison ratios must not be silently interpreted as non-membership."""
    audit = create_audit(tmp_path, monkeypatch, use_attack="rmia")
    source = audit.attack_scheduler.attacks[0].source_attack
    logits = np.tile([-1000.0, 1000.0, 0.0], (10, 1))
    references = [np.zeros_like(logits)] * 2
    source.prepare_from_logits(logits, references, np.zeros(10, dtype=np.int64), np.ones((2, 10), dtype=bool),
                               logits[:2], [values[:2] for values in references], np.zeros(2, dtype=np.int64))
    with pytest.raises(ValueError, match="comparisons are undefined"):
        source.run_attack()


def test_scorer_training_improves_separable_shadow_membership() -> None:
    """Verify that the scorer learns a separable shadow-membership signal."""
    attack = object.__new__(AttackRAPID)
    attack.batch_size = 16
    attack.mia_classifier_epochs = 50
    attack.learning_rate = AttackRAPID.AttackConfig().learning_rate
    attack.weight_decay = 5e-4
    torch.manual_seed(5)
    features = np.array([[0.1, -1.0], [3.0, 1.0]] * 16, dtype=np.float32)
    membership = np.array([1, 0] * 16, dtype=np.float32)
    scorer = attack._train_mia_classifier(features, membership)
    with torch.no_grad():
        scores = torch.sigmoid(scorer(torch.from_numpy(features))).numpy().reshape(-1)
    assert np.mean(scores[membership == 1]) > 0.9
    assert np.mean(scores[membership == 0]) < 0.1


def test_default_classifier_handles_large_finite_losses() -> None:
    """Reject the former SGD divergence on realizable binary losses up to 300."""
    original = np.array([0.1, 300.0] * 128)
    reference = np.array([100.1, 200.0] * 128)
    attack = loss_attack(np.ones(256, dtype=np.int64))
    for key, value in AttackRAPID.AttackConfig().model_dump().items():
        setattr(attack, key, value)
    # Invert softplus(-logit) without exp(300) overflowing.
    features = attack._get_mia_features(
        QueryModel([-np.log(np.expm1(original)).reshape(-1, 1)]),
        [QueryModel([-np.log(np.expm1(reference)).reshape(-1, 1)])], np.arange(256)
    )
    torch.manual_seed(42)
    membership = np.array([1, 0] * 128, dtype=np.float32)
    scorer = attack._train_mia_classifier(features, membership)
    with torch.no_grad():
        scores = torch.sigmoid(scorer(torch.from_numpy(features))).numpy().reshape(-1)
    assert np.isfinite(scores).all()
    assert np.mean(scores[membership == 1]) > 0.9
    assert np.mean(scores[membership == 0]) < 0.1
