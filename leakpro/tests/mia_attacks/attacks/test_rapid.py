#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""RAPID scoring, separation, cache reuse, and public audit regressions."""

import random
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
from torchvision.transforms import RandomHorizontalFlip

from examples.mia.cifar.cifar_data_handler import CifarDataHandler
from leakpro import LeakPro
from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.mia_attacks.attack_factory_mia import AttackFactoryMIA
from leakpro.attacks.mia_attacks.rapid import AttackRAPID
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.signals.signal import ModelLogits
from leakpro.signals.signal_extractor import PytorchModel
from leakpro.tests.mia_attacks.attacks.test_all_attacks_end_to_end import (
    TinyImageInputHandler,
    TinyImageTargetModel,
    _build_tiny_image_population,
    _create_target_artifacts,
)
from leakpro.utils.seed import seed_everything


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
    attack.use_attack = None
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
    (False, 0, True, "none"), (False, 0, True, "medium"),
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


@pytest.mark.parametrize("source", list(AttackFactoryMIA.attack_classes))
def test_source_name_is_resolved_from_registry(source: str) -> None:
    """Configuration follows the MIA registry without an attack-specific allowlist."""
    config = AttackRAPID.AttackConfig(use_attack=source.upper())
    assert config.use_attack == (None if source == "rapid" else source)


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

    assert set(attack.shadow_population).isdisjoint(attack.reference_population)
    assert set(attack.shadow_population).isdisjoint(attack.audit_dataset["data"])
    assert set(attack.reference_population).isdisjoint(attack.audit_dataset["data"])
    for role, ids, pool in [("shadow", attack.shadow_model_indices, attack.shadow_population),
                            ("references", attack.reference_model_indices, attack.reference_training_pool)]:
        with attack._model_cache(role, pool) as models:
            for metadata in models.get_shadow_model_metadata(ids):
                assert set(metadata.train_indices).issubset(pool)
    metadata_before = sorted((tmp_path / "output" / "attack_objects" / "rapid").rglob("metadata_*.pkl"))
    scores_before = result.signal_values.copy()
    model_before = deepcopy(attack.scoring_model.state_dict())
    monkeypatch.setattr(leakpro.handler, "train", lambda *_args, **_kwargs: pytest.fail("Cached audit retrained a model"))
    attack.prepare_attack()
    np.testing.assert_array_equal(attack.run_attack().signal_values, scores_before)
    for key, value in attack.scoring_model.state_dict().items():
        torch.testing.assert_close(value, model_before[key], rtol=0, atol=0)
    assert sorted((tmp_path / "output" / "attack_objects" / "rapid").rglob("metadata_*.pkl")) == metadata_before


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


@pytest.mark.parametrize("output", ["scores", "logits"])
def test_new_registry_source_needs_no_rapid_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output: str) -> None:
    """A newly registered source works through the same public audit as existing attacks."""
    monkeypatch.setitem(AttackFactoryMIA.attack_classes, "custom_source", AttackFactoryMIA.attack_classes["population"])
    audit = create_audit(tmp_path, monkeypatch, use_attack="CUSTOM_SOURCE", use_attack_output=output)
    result = audit.run_audit(create_pdf=False, use_optuna=False)[0]
    assert result.metadata["use_attack"] == "custom_source"
    assert result.signal_values.shape == (10,)
    assert np.isfinite(result.signal_values).all()


@pytest.mark.parametrize(("use_attack", "output", "online"), [
    (None, "scores", False), ("lira", "scores", False), ("lira", "scores", True),
    ("lira", "logits", False), ("lira", "logits", True),
])
def test_rapid_preserves_following_attack_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_attack: Optional[str], output: str, online: bool
) -> None:
    """Cold and cached RAPID runs preserve another instance's calibration and random streams."""
    audit = create_audit(tmp_path, monkeypatch, use_attack=use_attack, use_attack_output=output,
                         use_attack_config={"online": online} if use_attack else None, stochastic=True)
    attack = audit.attack_scheduler.attacks[0]
    population = AttackFactoryMIA.create_attack("population", {}, audit.handler)
    shared_model, shared_data = population.target_model, population.audit_dataset
    model_handler = ShadowModelHandler()
    handler_paths = model_handler.storage_path, model_handler.attack_cache_folder_path
    seed_everything(321)
    population.prepare_attack()
    expected_calibration = population.attack_signal.copy()
    expected_result = population.run_attack()
    expected_python, expected_numpy, expected_torch = random.random(), np.random.rand(5), torch.rand(5)  # noqa: S311 - Test the audit RNG stream.
    for _ in range(2):
        seed_everything(321)
        monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
        monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
        attack.prepare_attack()
        attack.run_attack()
        assert not torch.backends.cudnn.deterministic
        assert torch.backends.cudnn.benchmark
        assert population.target_model is shared_model
        assert population.audit_dataset is shared_data
        assert population.handler is audit.handler
        assert (model_handler.storage_path, model_handler.attack_cache_folder_path) == handler_paths
        population.prepare_attack()
        actual_result = population.run_attack()
        np.testing.assert_array_equal(population.attack_signal, expected_calibration)
        np.testing.assert_array_equal(actual_result.signal_values, expected_result.signal_values)
        np.testing.assert_array_equal(actual_result.tpr, expected_result.tpr)
        np.testing.assert_array_equal(actual_result.fpr, expected_result.fpr)
        assert random.random() == expected_python  # noqa: S311 - Test the audit RNG stream.
        np.testing.assert_array_equal(np.random.rand(5), expected_numpy)
        torch.testing.assert_close(torch.rand(5), expected_torch, rtol=0, atol=0)


@pytest.mark.parametrize("phase", ["prepare", "run"])
def test_rapid_failure_preserves_random_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    """Exceptions after random draws do not change the caller's next draws or backend flags."""
    audit = create_audit(tmp_path, monkeypatch, use_attack="population")
    attack = audit.attack_scheduler.attacks[0]
    attack.prepare_attack()

    def fail(*_args: object) -> None:
        seed_everything(999)
        random.random()  # noqa: S311 - Test the audit RNG stream.
        np.random.rand(5)
        torch.rand(5)
        raise RuntimeError("RAPID stage failed")

    if phase == "prepare":
        monkeypatch.setattr(attack, "_get_mia_features", fail)
        operation = attack.prepare_attack
    else:
        monkeypatch.setattr(attack.scoring_model, "forward", fail)
        operation = attack.run_attack
    seed_everything(321)
    expected_python, expected_numpy, expected_torch = random.random(), np.random.rand(5), torch.rand(5)  # noqa: S311 - Test the audit RNG stream.
    seed_everything(321)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    with pytest.raises(RuntimeError, match="RAPID stage failed"):
        operation()
    assert not torch.backends.cudnn.deterministic
    assert torch.backends.cudnn.benchmark
    assert random.random() == expected_python  # noqa: S311 - Test the audit RNG stream.
    np.testing.assert_array_equal(np.random.rand(5), expected_numpy)
    torch.testing.assert_close(torch.rand(5), expected_torch, rtol=0, atol=0)


def test_source_failure_restores_parent_handlers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A source exception must not replace the parent model, audit data, or shared model handlers."""
    audit = create_audit(tmp_path, monkeypatch, use_attack="population")
    attack = audit.attack_scheduler.attacks[0]
    parent_model, parent_data = attack.target_model, attack.audit_dataset
    parent_handler = ShadowModelHandler().handler

    def fail(source: AbstractMIA) -> None:
        assert source.handler is not audit.handler
        assert source.handler.target_model is not audit.handler.target_model
        raise RuntimeError("Source preparation failed")

    monkeypatch.setattr(AttackFactoryMIA.attack_classes["population"], "prepare_attack", fail)
    with pytest.raises(RuntimeError, match="Source preparation failed"):
        attack.prepare_attack()
    assert attack.target_model is parent_model
    assert attack.audit_dataset is parent_data
    assert ShadowModelHandler().handler is parent_handler
    with pytest.raises(RuntimeError, match="prepare_attack"):
        attack.run_attack()


def test_source_counts_are_not_fabricated_into_scores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Counts-only results have no per-sample signal for the RAPID classifier."""
    audit = create_audit(tmp_path, monkeypatch, use_attack="population")
    counts = MIAResult.from_confusion_counts([1, 0], tp=1, fp=0, tn=1, fn=0, result_name="counts")
    monkeypatch.setattr(AttackFactoryMIA.attack_classes["population"], "run_attack", lambda _self: counts)
    with pytest.raises(ValueError, match="per-sample scores"):
        audit.run_audit(create_pdf=False, use_optuna=False)


@pytest.mark.parametrize("identity", ["missing", "wrong", "duplicate"])
def test_equal_length_source_scores_require_correct_sample_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identity: str
) -> None:
    """A finite vector of the right length cannot stand in for scores of different or unidentified records."""
    audit = create_audit(tmp_path, monkeypatch, use_attack="population")

    def score(source: AbstractMIA) -> MIAResult:
        ids = source.audit_dataset["data"].copy()
        if identity == "wrong":
            ids += source.population_size
        elif identity == "duplicate":
            ids[-1] = ids[0]
        result = MIAResult.from_full_scores(
            [1] * len(source.train_indices) + [0] * len(source.test_indices), np.arange(len(ids)),
            result_name="source"
        )
        if identity != "missing":
            result.sample_indices = ids
        return result

    monkeypatch.setattr(AttackFactoryMIA.attack_classes["population"], "run_attack", score)
    with pytest.raises(ValueError, match="identify every requested sample exactly once"):
        audit.run_audit(create_pdf=False, use_optuna=False)


def test_source_scores_are_aligned_by_ids_not_report_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A source can reorder records when it supplies their true score identities."""
    audit = create_audit(tmp_path, monkeypatch, use_attack="population")
    attack = audit.attack_scheduler.attacks[0]

    def score(source: AbstractMIA) -> MIAResult:
        ids = source.audit_dataset["data"][::-1]
        labels = np.concatenate((np.ones(len(source.train_indices)), np.zeros(len(source.test_indices))))[::-1]
        result = MIAResult.from_full_scores(labels, ids.astype(float), result_name="source")
        result.sample_indices = ids
        return result

    monkeypatch.setattr(AttackFactoryMIA.attack_classes["population"], "run_attack", score)
    audit.run_audit(create_pdf=False, use_optuna=False)
    np.testing.assert_array_equal(attack.shadow_features[:, 1], attack.shadow_population)
    np.testing.assert_array_equal(attack.signal_data[:, 1], attack.audit_dataset["data"])


def test_reordered_unindexed_source_result_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ordered signal queries cannot justify a source result reordered after signal computation."""
    cls = AttackFactoryMIA.attack_classes["population"]
    run = cls.run_attack

    def reverse_result(source: AbstractMIA) -> MIAResult:
        result = run(source)
        return MIAResult.from_full_scores(result.true[::-1], result.signal_values[::-1], result_name="reordered")

    monkeypatch.setattr(cls, "run_attack", reverse_result)
    audit = create_audit(tmp_path, monkeypatch, use_attack="population")
    with pytest.raises(ValueError, match="identify every requested sample"):
        audit.run_audit(create_pdf=False, use_optuna=False)


def test_cache_setup_failure_restores_shared_handler_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Failure to create a role directory must not redirect other attacks' shared caches."""
    audit = create_audit(tmp_path, monkeypatch)
    attack = audit.attack_scheduler.attacks[0]
    handler = ShadowModelHandler()
    before = handler.storage_path, handler.attack_cache_folder_path
    with attack._model_cache("failure", np.array([1, 2])):
        blocked = Path(handler.attack_cache_folder_path)
    blocked.rmdir()
    blocked.write_text("blocked")
    with pytest.raises(FileExistsError), attack._model_cache("failure", np.array([1, 2])):
        pytest.fail("Cache setup should fail before yielding")
    assert (handler.storage_path, handler.attack_cache_folder_path) == before


def test_loaded_source_scores_preserve_per_record_features(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The source's ordinary saved result retains score IDs and can be consumed after loading."""
    audit = create_audit(tmp_path, monkeypatch, use_attack="population")
    attack = audit.attack_scheduler.attacks[0]
    cls = AttackFactoryMIA.attack_classes["population"]
    run = cls.run_attack
    expected = []

    def reload_result(source: AbstractMIA) -> MIAResult:
        result = run(source)
        expected.append((source.audit_dataset["data"].copy(), result.signal_values.copy()))
        result.save(source, str(tmp_path))
        path = tmp_path / "data_objects" / f"{source.attack_id}.json"
        return MIAResult.load(str(path))

    monkeypatch.setattr(cls, "run_attack", reload_result)
    audit.run_audit(create_pdf=False, use_optuna=False)
    for features, indices, result in zip((attack.shadow_features, attack.signal_data),
                                       (attack.shadow_population, attack.audit_dataset["data"]), expected):
        mapping = dict(zip(*result))
        np.testing.assert_allclose(features[:, 1], [mapping[index] for index in indices], rtol=1e-6)


def test_hsj_source_with_unverifiable_rows_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the existing HSJ position/ID defect outside this PR and reject mismatched source rows."""
    audit = create_audit(tmp_path, monkeypatch, use_attack="HSJ", use_attack_config={
        "attack_data_fraction": 1.0, "batch_size": 2, "initial_num_evals": 1,
        "max_num_evals": 2, "num_iterations": 1, "verbose": False
    })
    with pytest.raises(ValueError, match="identify every requested sample"):
        audit.run_audit(create_pdf=False, use_optuna=False)


@pytest.mark.parametrize("use_attack", ["lira", "rmia", "base", "multi_signal_lira", "population"])
@pytest.mark.parametrize("stochastic", [False, True])
@pytest.mark.parametrize("online", [False, True])
def test_composition_runs_existing_attack_on_shadow_and_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_attack: str, stochastic: bool, online: bool
) -> None:
    """Use real source results, align shadow rows, and preserve the parent audit across cached replay."""
    config = {}
    if "online" in AttackFactoryMIA.attack_classes[use_attack].AttackConfig.model_fields:
        config["online"] = online
    audit = create_audit(tmp_path, monkeypatch, use_attack=use_attack, use_attack_config=config, stochastic=stochastic)
    attack = audit.attack_scheduler.attacks[0]
    original_indices = attack.audit_dataset["data"].copy()
    source_cls = AttackFactoryMIA.attack_classes[use_attack]
    calls = []
    run = source_cls.run_attack

    def run_source(source: AbstractMIA) -> object:
        calls.append(source)
        return run(source)

    monkeypatch.setattr(source_cls, "run_attack", run_source)
    result = audit.run_audit(create_pdf=False, use_optuna=False)[0]
    assert len(calls) == 2
    assert set(attack.source_results) == {"shadow", "target"}
    assert len(result.signal_values) == len(original_indices)
    assert np.isfinite(result.signal_values).all()
    np.testing.assert_array_equal(attack.audit_dataset["data"], original_indices)
    order = np.random.default_rng(attack.random_seed).permutation(np.argsort(attack.shadow_population))
    np.testing.assert_allclose(attack.shadow_features[order, 1], attack.source_results["shadow"].signal_values, rtol=1e-6)
    target_order = np.random.default_rng(attack.random_seed).permutation(np.argsort(original_indices))
    np.testing.assert_allclose(attack.signal_data[target_order, 1], attack.source_results["target"].signal_values, rtol=1e-6)
    assert result.metadata["use_attack"] == use_attack
    assert result.metadata["use_attack_config"] == attack.source_attack.configs.model_dump(exclude={"num_shadow_models",
                                                                                  "training_data_fraction"})
    first = result.signal_values.copy()
    monkeypatch.setattr(audit.handler, "train", lambda *_args, **_kwargs: pytest.fail("Composition retrained cached models"))
    attack.prepare_attack()
    np.testing.assert_array_equal(attack.run_attack().signal_values, first)
    assert len(calls) == 4


@pytest.mark.parametrize("use_attack", ["lira", "rmia", "base", "multi_signal_lira", "population"])
@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("use_attack_output", ["scores", "logits"])
def test_composition_ignores_target_membership_when_building_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_attack: str, online: bool, use_attack_output: str
) -> None:
    """Permuting evaluation membership cannot alter calibrated target scores or the fitted classifier."""
    fields = AttackFactoryMIA.attack_classes[use_attack].AttackConfig.model_fields
    source_config = {"online": online} if "online" in fields else {}
    audit = create_audit(tmp_path, monkeypatch, use_attack=use_attack, use_attack_config=source_config,
                         use_attack_output=use_attack_output)
    attack = audit.attack_scheduler.attacks[0]
    attack.prepare_attack()
    indices = attack.audit_dataset["data"].copy()
    before = deepcopy(attack.scoring_model.state_dict())
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


@pytest.mark.parametrize("use_attack", ["lira", "rmia", "base", "multi_signal_lira", "population"])
@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("stochastic", [False, True])
def test_composition_reuses_logits_without_source_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_attack: str, online: bool, stochastic: bool
) -> None:
    """Compare reused source logits with an independent loss oracle, including OUT-only online calibration."""
    fields = AttackFactoryMIA.attack_classes[use_attack].AttackConfig.model_fields
    source_config = {"online": online} if "online" in fields else {}
    audit = create_audit(tmp_path, monkeypatch, use_attack=use_attack, use_attack_config=source_config,
                         use_attack_output="logits", stochastic=stochastic)
    attack = audit.attack_scheduler.attacks[0]
    monkeypatch.setattr(AttackFactoryMIA.attack_classes[use_attack], "run_attack",
                        lambda *_args: pytest.fail("Logits mode requested source scores"))
    result = audit.run_audit(create_pdf=False, use_optuna=False)[0]
    assert attack.source_results == {}
    with attack._model_cache("references", attack.reference_training_pool) as models:
        refs, _ = models.get_shadow_models(attack.reference_model_indices)
    indices = attack.audit_dataset["data"]
    with attack._model_cache("references", attack.reference_training_pool) as models:
        out_indices = ~models.get_in_indices_mask(attack.reference_model_indices, indices).T
    assert out_indices.any(axis=0).all()
    if getattr(attack.source_configs, "online", False):
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
def test_online_source_rejects_incomplete_reference_coverage_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_attack: str
) -> None:
    """Do not drop or fabricate scores for points missing IN or OUT references."""
    audit = create_audit(tmp_path, monkeypatch, use_attack=use_attack, use_attack_config={"online": True},
                         use_attack_output="logits")
    attack = audit.attack_scheduler.attacks[0]
    attack.training_data_fraction = 0.1
    monkeypatch.setattr(audit.handler, "train", lambda *_args, **_kwargs: pytest.fail("Invalid coverage trained a model"))
    with pytest.raises(ValueError, match="IN and OUT references for every point"):
        attack.prepare_attack()


def test_online_source_rejects_cached_random_references(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing random reference models cannot bypass the online IN/OUT coverage check."""
    audit = create_audit(tmp_path, monkeypatch, use_attack="lira", use_attack_config={"online": True}, use_attack_output="logits")
    attack = audit.attack_scheduler.attacks[0]
    handler = ShadowModelHandler()
    torch.manual_seed(7)
    np.random.seed(7)
    indices = np.arange(attack.population_size)
    with attack._model_cache("references", indices) as handler:
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
    with pytest.raises(ValueError, match="Dataset indices are empty"):
        attack.prepare_attack()
    with pytest.raises(RuntimeError, match="prepare_attack"):
        attack.run_attack()


@pytest.mark.parametrize("use_attack", ["lira", "rmia"])
def test_composition_rejects_nonfinite_reference_logits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_attack: str
) -> None:
    """Invalid references must not become finite zero membership scores."""
    audit = create_audit(tmp_path, monkeypatch, use_attack=use_attack, use_attack_output="logits")
    attack = audit.attack_scheduler.attacks[0]
    attack.prepare_attack()
    with attack._model_cache("references", attack.reference_training_pool) as models:
        references, _ = models.get_shadow_models(attack.reference_model_indices)
    monkeypatch.setattr(references[0], "get_logits", lambda data: np.full((len(data), 3), np.nan))
    with pytest.raises(ValueError, match="non-finite model logits"):
        attack._get_mia_features(attack.target_model, references, attack.audit_dataset["data"])



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
