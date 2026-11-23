#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Mini-scale RAPID audits and focused numerical, identity, and failure regressions."""

import pickle
import sys
from copy import deepcopy
from inspect import getfile
from pathlib import Path
from queue import Queue
from types import MethodType, ModuleType, SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import joblib
import numpy as np
import pytest
import torch
import yaml
from pydantic import ValidationError
from scipy.special import expit, logsumexp
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset, default_collate
from torchvision.transforms import RandomHorizontalFlip

from leakpro import LeakPro
from leakpro.attacks.mia_attacks.attack_factory_mia import AttackFactoryMIA
from leakpro.attacks.mia_attacks.rapid import AttackRAPID
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.input_handler.mia_handler import ordered_query_dataloader
from leakpro.reporting.mia_result import MIAResult
from leakpro.tests.mia_attacks.attacks.test_all_attacks_end_to_end import (
    TinyImageInputHandler,
    TinyImageTargetModel,
    _build_tiny_image_population,
    _create_target_artifacts,
)


def create_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, num_classes: int = 3,
                 **options: object) -> LeakPro:
    """Reuse the runtime fixtures with nonconsecutive IDs, augmentation, and small training budgets."""
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(23)
    population, model = _build_tiny_image_population(), TinyImageTargetModel(num_classes=num_classes)
    population.targets %= num_classes
    population.augment = RandomHorizontalFlip()
    data_path, target_dir = _create_target_artifacts(
        tmp_path, population, model, nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr=0.01),
        "rapid_test", train_indices=[17, 4, 29, 5, 1], test_indices=[33, 9, 48, 11, 2]
    )
    config = {
        "audit": {"random_seed": 19, "attack_type": "mia", "data_modality": "image",
                  "output_dir": str(tmp_path / "output"), "attack_list": [
                      {"attack": "rapid", "num_reference_models": 2, "query_num": 2,
                       "mia_classifier_patience": 2, "batch_size": 7, **options}]},
        "target": {"module_path": getfile(TinyImageTargetModel), "model_class": "TinyImageTargetModel",
                   "target_folder": str(target_dir), "data_path": str(data_path)},
        "shadow_model": {"epochs": 1},
    }
    path = tmp_path / "audit.yaml"
    path.write_text(yaml.safe_dump(config))
    return LeakPro(TinyImageInputHandler, str(path))


def set_query_collation(handler, collate, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use custom preprocessing for queries while leaving the training loader intact."""
    default_loader = type(handler).get_dataloader

    def get_loader(owner, indices, params=None, batch_size=None, shuffle=None):
        loader = default_loader(owner, indices, params, batch_size, shuffle)
        if shuffle is False:
            loader.collate_fn = collate
        return loader

    monkeypatch.setattr(handler, "get_dataloader", MethodType(get_loader, handler))


@pytest.mark.parametrize("drop_last", [False, True])
def test_worker_publishes_rapid_ids_or_reports_dropped_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drop_last: bool
) -> None:
    """The worker publishes aligned RAPID rows and rejects incomplete query batches."""
    from leakpro.webapp.backend import worker

    audit = create_audit(tmp_path, monkeypatch)
    fixture, = audit.attack_scheduler.attacks
    config = yaml.safe_load((tmp_path / "audit.yaml").read_text())
    target_dir = tmp_path / "models" / "fixture"
    target_dir.parent.mkdir()
    Path(config["target"]["target_folder"]).rename(target_dir)
    (tmp_path / "arch.py").write_text(
        "from leakpro.tests.mia_attacks.attacks.test_all_attacks_end_to_end import TinyImageTargetModel\n"
    )
    params = dict(config["audit"]["attack_list"][0])
    params.pop("attack")
    if drop_last:
        metadata_path = target_dir / "model_metadata.pkl"
        metadata = joblib.load(metadata_path)
        metadata.data_loader.params["drop_last"] = True
        metadata_path.write_bytes(pickle.dumps(metadata))

    if "leakpro.dataset" not in sys.modules:
        monkeypatch.setitem(sys.modules, "leakpro.dataset", ModuleType("leakpro.dataset"))
    monkeypatch.syspath_prepend(str(worker._LEAKPRO_ROOT))
    state = {"data_path": config["target"]["data_path"],
             "models": [{"name": "fixture", "attacks": [{"attack": "rapid", "params": params}]}]}
    persist = Mock()

    worker.run_audit_job("fixture", tmp_path, state, Queue(), persist)

    persist.assert_called_once_with("fixture")
    output = tmp_path / "leakpro_output" / "fixture" / "data_objects"
    if drop_last:
        assert state["status"] == "failed", state.get("error")
        assert "must yield every requested sample" in state["error"]
        assert "results" not in state
        assert not list(output.glob("*.json"))
    else:
        assert state["status"] == "done", state.get("error")
        exported, = state["results"][0]["attacks"]
        path, = output.glob("*.json")
        saved = MIAResult.load(str(path))
        np.testing.assert_array_equal(exported["audit_indices"], fixture.audit_dataset["data"])
        np.testing.assert_array_equal(exported["audit_indices"], saved.audit_indices)
        np.testing.assert_array_equal(exported["true_labels"], saved.true)
        np.testing.assert_array_equal(exported["signal_values"], saved.signal_values)


def check_cached_replay(audit: LeakPro, attack: AttackRAPID, result: MIAResult, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cached replay changes reporting membership only, preserving features, fitted weights, and scores."""
    ids = attack.audit_dataset["data"].copy()
    before = deepcopy(attack.scoring_model.state_dict())
    target = attack.signal_data.copy()
    train, test = audit.handler.train_indices.copy(), audit.handler.test_indices.copy()
    monkeypatch.setattr(audit.handler, "train_indices", test)
    monkeypatch.setattr(audit.handler, "test_indices", train)
    monkeypatch.setattr(audit.handler, "target_model_metadata", audit.handler.target_model_metadata.model_copy(
        update={"train_indices": test, "test_indices": train}, deep=True
    ))
    monkeypatch.setattr(attack, "audit_dataset", {"data": ids, "in_members": np.arange(5, 10), "out_members": np.arange(5)})
    monkeypatch.setattr(audit.handler, "train", lambda *_args, **_kwargs: pytest.fail("Cached replay retrained a model"))
    attack.prepare_attack()
    repeated = attack.run_attack()
    np.testing.assert_array_equal(repeated.true, 1 - result.true)
    np.testing.assert_array_equal(repeated.signal_values, result.signal_values)
    np.testing.assert_array_equal(attack.signal_data, target)
    for name, value in attack.scoring_model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("online", [False, True])
def test_rapid_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, online: bool
) -> None:
    """Verify each rotation and split against training IDs, then replay with swapped reporting labels."""
    audit = create_audit(tmp_path, monkeypatch, online=online)
    attack, = audit.attack_scheduler.attacks
    parent, handler = attack.target_model.model_obj, ShadowModelHandler().handler
    fit = Mock(wraps=attack._train_mia_classifier)
    monkeypatch.setattr(attack, "_train_mia_classifier", fit)
    query, split = attack._query_losses, attack._validation_records
    queried, splits = [], []

    def record_losses(models, records):
        losses = query(models, records)
        queried.append((models[0] is attack.target_model, records.copy(), losses.copy()))
        return losses

    monkeypatch.setattr(attack, "_query_losses", record_losses)

    def record_validation(membership, covered):
        validation = split(membership, covered)
        splits.append(validation.copy())
        return validation

    monkeypatch.setattr(attack, "_validation_records", record_validation)

    parent.train()
    parent.linear.eval()
    train = Mock(wraps=audit.handler.train)
    monkeypatch.setattr(audit.handler, "train", train)
    with pytest.raises(RuntimeError, match="prepare_attack"):
        attack.run_attack()
    result, = audit.run_audit(create_pdf=False, use_optuna=False)
    assert train.call_count == attack.num_reference_models + 1
    (is_target, records, losses), (queried_target, target_ids, target_losses) = queried
    assert not is_target and queried_target
    assert set(records).isdisjoint(target_ids)
    np.testing.assert_array_equal(target_ids, [17, 4, 29, 5, 1, 33, 9, 48, 11, 2])
    np.testing.assert_array_equal(result.audit_indices, target_ids)
    np.testing.assert_array_equal(result.true, [1] * 5 + [0] * 5)
    assert np.all((result.signal_values >= 0) & (result.signal_values <= 1))
    assert result.metadata == attack.configs.model_dump()
    assert parent.training and not parent.linear.training
    assert ShadowModelHandler().handler is handler
    assert (tmp_path / "output" / "results" / result.id / "result.txt").exists()

    training_ids = [set(meta.train_indices) for meta in
                    ShadowModelHandler().get_shadow_model_metadata(attack.reference_model_indices)]
    validation, = splits

    def expected_rows(sample_ids, per_model_losses, positions, rotating):
        features, labels = [], []
        for model in range(len(training_ids) if rotating else 1):
            eligible = range(len(training_ids)) if not rotating else [i for i in range(len(training_ids)) if i != model]
            for position in positions:
                index, original = sample_ids[position], per_model_losses[model, position]
                outside = [i for i in eligible if index not in training_ids[i]]
                inside = [i for i in eligible if index in training_ids[i]]
                groups = [outside, inside] if online else [outside]
                if all(groups):
                    offset = int(not rotating)
                    features.append([original, *(original - per_model_losses[np.array(side) + offset, position].mean()
                                                 for side in groups)])
                    if rotating:
                        labels.append(index in training_ids[model])
        return features, labels

    for positions, (features, membership) in zip((np.flatnonzero(~validation), np.flatnonzero(validation)),
                                                 (fit.call_args.args[:2], fit.call_args.args[2:])):
        expected, expected_labels = expected_rows(records, losses, positions, True)
        np.testing.assert_allclose(features, expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(membership, expected_labels)
        assert set(membership) == {False, True}
    expected, _ = expected_rows(target_ids, target_losses, range(len(target_ids)), False)
    np.testing.assert_allclose(attack.signal_data, expected, rtol=1e-6, atol=1e-7)
    original_fit = fit.call_args.args
    check_cached_replay(audit, attack, result, monkeypatch)
    for original, replay in zip(original_fit, fit.call_args.args):
        np.testing.assert_array_equal(original, replay)


@pytest.mark.parametrize("source", ["lira", "rmia"])
def test_reuses_shared_references(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str) -> None:
    """RAPID reuses existing LiRA and RMIA references."""
    audit = create_audit(tmp_path, monkeypatch)
    attack = audit.attack_scheduler.attacks[0]
    previous = AttackFactoryMIA.attack_classes[source](audit.handler, {
        "num_shadow_models": attack.num_reference_models + 1, "training_data_fraction": attack.training_data_fraction,
    })
    previous.prepare_attack()
    models = ShadowModelHandler()
    paths = models.storage_path, models.attack_cache_folder_path
    references, _ = models.get_shadow_models(previous.shadow_model_indices)
    weights = [deepcopy(model.model_obj.state_dict()) for model in references]
    train = Mock(wraps=audit.handler.train)
    monkeypatch.setattr(audit.handler, "train", train)
    attack.prepare_attack()
    train.assert_not_called()
    np.testing.assert_array_equal(attack.reference_model_indices, previous.shadow_model_indices)
    assert (models.storage_path, models.attack_cache_folder_path) == paths
    references, _ = models.get_shadow_models(attack.reference_model_indices)
    for model, state in zip(references, weights):
        for name, value in state.items():
            torch.testing.assert_close(model.model_obj.state_dict()[name], value, rtol=0, atol=0)
    assert np.isfinite(attack.run_attack().signal_values).all()


@pytest.mark.parametrize("source", ["lira", "rmia"])
def test_rapid_references_supply_logits_on_demand(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str) -> None:
    """RAPID saves its reference models and a later audit loads their logits on demand."""
    audit = create_audit(tmp_path, monkeypatch)
    attack, = audit.attack_scheduler.attacks
    audit.run_audit(create_pdf=False, use_optuna=False)
    assert not list(Path(audit.handler.configs.audit.output_dir).rglob("*_logits.npy"))

    other = AttackFactoryMIA.attack_classes[source](audit.handler, {
        "num_shadow_models": attack.num_reference_models + 1, "training_data_fraction": attack.training_data_fraction,
    })
    monkeypatch.setattr(audit.handler, "train", lambda *_args, **_kwargs: pytest.fail("retrained a cached reference"))
    other.prepare_attack()
    np.testing.assert_array_equal(other.shadow_model_indices, attack.reference_model_indices)
    assert np.isfinite(other.run_attack().signal_values).all()

    models = ShadowModelHandler()
    for index in attack.reference_model_indices:
        prefix = Path(models.attack_cache_folder_path) / f"sm_{index}_{models.population_hash}"
        assert prefix.with_name(f"{prefix.name}_logits.npy").is_file()
        cached_ids = np.load(prefix.with_name(f"{prefix.name}_indices.npy"))
        np.testing.assert_array_equal(cached_ids, attack.audit_dataset["data"])


def test_direct_losses_follow_query_preprocessing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Changing the collator changes the queried losses and preserves trained references."""
    audit = create_audit(tmp_path, monkeypatch, num_classes=2)
    attack, = audit.attack_scheduler.attacks
    audit.handler.population.augment = None
    seen = []
    for scale in (1, -1):
        def collate(batch, scale=scale):
            data, labels = default_collate(batch)
            return data * scale, labels

        set_query_collation(audit.handler, collate, monkeypatch)
        result, = audit.run_audit(create_pdf=False, use_optuna=False)
        data, labels = zip(*(attack.population[int(index)] for index in result.audit_indices))
        raw = attack.target_model.get_logits(torch.stack(data) * scale)
        expected = logsumexp(raw, axis=1) - raw[np.arange(len(raw)), labels]
        np.testing.assert_allclose(attack.signal_data[:, 0], expected, atol=3e-7)
        seen.append(attack.signal_data[:, 0].copy())
        monkeypatch.setattr(audit.handler, "train", lambda *_args, **_kwargs: pytest.fail("Query changes retrained a model"))
    assert not np.allclose(*seen)


def test_reference_metadata_retry_uses_complete_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A partial metadata write never publishes, and an older truncated entry is skipped on retry."""
    audit = create_audit(tmp_path, monkeypatch)
    attack, = audit.attack_scheduler.attacks
    models = ShadowModelHandler()
    storage = Path(models.storage_path)
    prior = storage / "metadata_0.pkl"
    incomplete = storage / "metadata_1.pkl"
    dump = pickle.dump
    fragments = []

    def interrupt(value, stream, *args, **kwargs):
        if Path(stream.name).name == incomplete.name:
            fragments.append(pickle.dumps(value)[:8])
            stream.write(fragments[-1])
            raise InterruptedError("incomplete reference metadata")
        dump(value, stream, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(pickle, "dump", interrupt)
        with pytest.raises(InterruptedError, match="incomplete reference metadata"):
            audit.run_audit(create_pdf=False, use_optuna=False)
    assert not incomplete.exists()
    original = prior.read_bytes(), (storage / "shadow_model_0.pkl").read_bytes()
    incomplete.write_bytes(fragments[0])
    result, = audit.run_audit(create_pdf=False, use_optuna=False)
    np.testing.assert_array_equal(result.audit_indices, attack.audit_dataset["data"])
    assert set(attack.reference_model_indices) == {0, 2, 3}
    assert (prior.read_bytes(), (storage / "shadow_model_0.pkl").read_bytes()) == original


@pytest.mark.parametrize("completed_models", [1, 2])
def test_restart_preserves_reference_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, completed_models: int) -> None:
    """Resume after a partial shared pool without changing membership or scores."""
    audit = create_audit(tmp_path, monkeypatch)
    expected, = audit.run_audit(create_pdf=False, use_optuna=False)
    attack, = audit.attack_scheduler.attacks
    metadata = ShadowModelHandler().get_shadow_model_metadata(attack.reference_model_indices)
    expected_indices = [model.train_indices for model in metadata]

    config_path = tmp_path / "audit.yaml"
    config = yaml.safe_load(config_path.read_text())
    output_dir = tmp_path / "resume"
    config["audit"]["output_dir"] = str(output_dir)
    config_path.write_text(yaml.safe_dump(config))
    interrupted = LeakPro(TinyImageInputHandler, str(config_path))
    train, calls = interrupted.handler.train, 0

    def stop_training(*args, **kwargs):
        nonlocal calls
        if calls == completed_models:
            raise InterruptedError("training interrupted")
        calls += 1
        return train(*args, **kwargs)

    monkeypatch.setattr(interrupted.handler, "train", stop_training)
    with pytest.raises(InterruptedError, match="training interrupted"):
        interrupted.run_audit(create_pdf=False, use_optuna=False)
    checkpoints = {path: path.read_bytes() for path in output_dir.rglob("metadata_*.pkl")}
    assert len(checkpoints) == completed_models

    resumed = LeakPro(TinyImageInputHandler, str(config_path))
    result, = resumed.run_audit(create_pdf=False, use_optuna=False)
    resumed_attack, = resumed.attack_scheduler.attacks
    metadata = ShadowModelHandler().get_shadow_model_metadata(resumed_attack.reference_model_indices)
    np.testing.assert_array_equal([model.train_indices for model in metadata], expected_indices)
    np.testing.assert_array_equal(result.audit_indices, expected.audit_indices)
    np.testing.assert_array_equal(result.signal_values, expected.signal_values)
    assert all(path.read_bytes() == data for path, data in checkpoints.items())


def test_rapid_reuses_sparse_reference_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Retained reference files remain reusable when their IDs exceed the directory's entry count."""
    audit = create_audit(tmp_path, monkeypatch)
    attack, = audit.attack_scheduler.attacks
    attack.prepare_attack()
    original = attack.run_attack()
    models = ShadowModelHandler()
    retained = [17, 23, 41]
    for previous, current in zip(attack.reference_model_indices, retained):
        for name in (models.model_storage_name, models.metadata_storage_name):
            path = Path(models.storage_path) / f"{name}_{previous}.pkl"
            path.rename(path.with_name(f"{name}_{current}.pkl"))
    monkeypatch.setattr(audit.handler, "train", lambda *_args, **_kwargs: pytest.fail("Retrained a retained model"))

    attack.prepare_attack()

    np.testing.assert_array_equal(attack.reference_model_indices, retained)
    np.testing.assert_array_equal(attack.run_attack().signal_values, original.signal_values)


def test_rapid_references_respect_a_following_offline_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An offline consumer cannot reuse references trained outside its pool even at the same training size."""
    audit = create_audit(tmp_path, monkeypatch)
    rapid, = audit.attack_scheduler.attacks
    train_size = rapid.population_size // 2
    offline_pool_size = rapid.population_size - rapid.audit_size
    offline = AttackFactoryMIA.create_attack("multi_signal_lira", {
        "online": False, "num_shadow_models": 2, "training_data_fraction": (train_size + 0.25) / offline_pool_size,
    }, audit.handler)
    audit.attack_scheduler.attacks.append(offline)
    audit.attack_scheduler.attack_names.append("multi_signal_lira")

    _, result = audit.run_audit(create_pdf=False, use_optuna=False)

    np.testing.assert_array_equal(result.audit_indices, offline.audit_dataset["data"])
    references = ShadowModelHandler().get_shadow_model_metadata(offline.shadow_model_indices)
    for metadata in references:
        assert len(metadata.train_indices) == train_size
        assert set(metadata.train_indices).issubset(offline.attack_data_indices)


def test_rapid_rejects_references_from_a_reordered_population(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Swapping records invalidates old weights and logits even with the same split IDs and checkpoint."""
    audit = create_audit(tmp_path, monkeypatch)
    previous, = audit.attack_scheduler.attacks
    audit.handler.population.augment = None
    previous.prepare_attack()
    models = ShadowModelHandler()
    old_path = models.attack_cache_folder_path
    old_logits = models.load_logits(name="target")
    data_path = audit.handler.configs.target.data_path
    population = joblib.load(data_path)
    swapped = [2, 9]
    population.data[swapped] = population.data[swapped[::-1]].clone()
    population.targets[swapped] = population.targets[swapped[::-1]].clone()
    joblib.dump(population, data_path)

    replaced = LeakPro(TinyImageInputHandler, str(tmp_path / "audit.yaml"))
    replaced.handler.population.augment = None
    current, = replaced.attack_scheduler.attacks
    np.testing.assert_array_equal(previous.audit_dataset["data"], current.audit_dataset["data"])
    train = Mock(wraps=replaced.handler.train)
    monkeypatch.setattr(replaced.handler, "train", train)
    current.prepare_attack()
    assert set(previous.reference_model_indices).isdisjoint(current.reference_model_indices)
    assert train.call_count == current.num_reference_models + 1
    assert models.attack_cache_folder_path != old_path
    expected_order = np.arange(len(current.audit_dataset["data"]))
    rows = np.flatnonzero(np.isin(current.audit_dataset["data"], swapped))
    expected_order[rows] = expected_order[rows[::-1]]
    assert not np.array_equal(old_logits, old_logits[expected_order])
    np.testing.assert_array_equal(models.load_logits(name="target"), old_logits[expected_order])


def test_query_loader_accepts_inherited_sequential_iteration() -> None:
    """A sampler inheriting sequential iteration keeps the same ordering guarantee."""
    class InheritedSequentialSampler(SequentialSampler):
        pass

    class ReversedSampler(SequentialSampler):
        def __iter__(self):
            return iter(range(len(self.data_source) - 1, -1, -1))

    dataset = TensorDataset(torch.arange(4), torch.arange(4))

    def loader(sampler, *_args, **_kwargs):
        return DataLoader(dataset, batch_size=2, sampler=sampler(dataset))

    indices = np.arange(len(dataset))
    batches = ordered_query_dataloader(lambda *args, **kwargs: loader(InheritedSequentialSampler, *args, **kwargs), indices)
    np.testing.assert_array_equal(torch.cat([labels for _, labels in batches]), indices)
    with pytest.raises(ValueError, match="sequential batches"):
        ordered_query_dataloader(lambda *args, **kwargs: loader(ReversedSampler, *args, **kwargs), indices)


class QueryModel:
    """Provide known per-record logits across repeated and partial-batch queries."""

    def __init__(self, queries: list) -> None:
        self.queries, self.calls = queries, 0

    def get_logits(self, data: torch.Tensor) -> np.ndarray:
        """Advance the query at its first record, rather than at each batch."""
        self.calls += int(data[0, 0] == 0)
        return np.asarray(self.queries[(self.calls - 1) % len(self.queries)], dtype=np.float64)[data[:, 0].numpy()]


def unit_attack(labels: list) -> AttackRAPID:
    """Build the numerical boundary without unrelated model-handler setup."""
    attack = object.__new__(AttackRAPID)
    attack.configs = AttackRAPID.AttackConfig(batch_size=2, query_num=2)
    for key, value in attack.configs.model_dump().items():
        setattr(attack, key, value)
    dataset = TensorDataset(torch.arange(len(labels)).reshape(-1, 1), torch.as_tensor(labels))
    attack._handler = SimpleNamespace(get_dataloader=lambda *_args, **kwargs: DataLoader(
        dataset, batch_size=kwargs["batch_size"], shuffle=kwargs["shuffle"]
    ))
    return attack


def test_offline_rapid_rejects_one_record_without_out_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single uncovered record invalidates the requested audit result."""
    attack = unit_attack([0, 1])
    attack.reference_model_indices = [0, 1]
    logits = [[[0.1, 0.4], [0.7, 0.2]]]
    mask = np.array([[True, True], [False, True]])
    monkeypatch.setattr("leakpro.attacks.mia_attacks.rapid.ShadowModelHandler", lambda: SimpleNamespace(
        get_in_indices_mask=lambda *_args: mask
    ))

    with pytest.raises(ValueError, match="references must cover every requested sample"):
        attack._get_mia_features(QueryModel(logits), [QueryModel(logits), QueryModel(logits)], np.array([17, 4]))


def test_loss_features_and_final_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    """Independent loss and probability oracles catch wrong calibration, row order, direction, and partial batches."""
    labels = [1, 0, 2]
    target = np.array([[[0, 3, -2], [2, 0, 1], [4, 3, 1]], [[1, 1, 0], [-1, 2, 0], [0, 0, 0]]])
    references = np.array([[[0, 0, 0], [0, 1, -1], [0, 1, 2]], [[3, 2, 1], [1, 0, 3], [1, 2, -2]]])
    attack, model = unit_attack(labels), QueryModel(list(target))
    attack.reference_model_indices = [0, 1]
    outside = np.array([[True, False, True], [False, True, True]])
    monkeypatch.setattr("leakpro.attacks.mia_attacks.rapid.ShadowModelHandler", lambda: SimpleNamespace(
        get_in_indices_mask=lambda *_args: ~outside.T
    ))
    reference_models = [QueryModel([values]) for values in references]
    values = attack._get_mia_features(model, reference_models, np.array([17, 4, 29]))
    original = (logsumexp(target, axis=-1) - target[:, np.arange(3), labels]).mean(axis=0)
    losses = logsumexp(references, axis=-1) - references[:, np.arange(3), labels]
    reference = np.where(outside, losses, 0).sum(axis=0) / outside.sum(axis=0)
    np.testing.assert_allclose(values, np.column_stack((original, original - reference)), rtol=1e-6, atol=1e-7)
    assert model.calls == 2
    assert all(ref.calls == 2 for ref in reference_models)
    attack.signal_data, attack.scoring_model = values, nn.Linear(2, 1)
    attack.scoring_model.load_state_dict({"weight": torch.tensor([[0.5, 2.0]]), "bias": torch.tensor([-0.25])})
    attack.audit_dataset = {
        "data": np.array([17, 4, 29]), "in_members": np.array([2]), "out_members": np.array([0, 1])
    }
    result = attack.run_attack()
    np.testing.assert_allclose(result.signal_values, expit(values @ [0.5, 2.0] - 0.25), rtol=1e-6)
    np.testing.assert_array_equal(result.true, [0, 0, 1])


@pytest.mark.parametrize("scenario", ["collated_labels", "binary_heads"])
def test_losses_preserve_handler_outputs(monkeypatch: pytest.MonkeyPatch, scenario: str) -> None:
    """OUT calibration supports emitted batch labels and different binary output widths."""
    labels = np.array([0, 1, 0])
    target = np.array([[0.0, 2.0], [3.0, 1.0], [1.0, 0.0]])
    reference = target[:, :1] if scenario == "binary_heads" else target[:, ::-1]
    attack = unit_attack(labels + (scenario == "collated_labels"))
    if scenario == "collated_labels":
        get_loader = attack.handler.get_dataloader

        def collate(batch: list) -> tuple:
            data, stored_labels = default_collate(batch)
            return data, stored_labels - 1

        def remapped_loader(*args: object, **kwargs: object) -> DataLoader:
            loader = get_loader(*args, **kwargs)
            loader.collate_fn = collate
            return loader

        attack.handler.get_dataloader = remapped_loader
    attack.reference_model_indices = [0]
    monkeypatch.setattr("leakpro.attacks.mia_attacks.rapid.ShadowModelHandler", lambda: SimpleNamespace(
        get_in_indices_mask=lambda *_args: np.zeros((3, 1), dtype=bool)
    ))
    values = attack._get_mia_features(QueryModel([target]), [QueryModel([reference])], np.array([17, 4, 29]))
    original = logsumexp(target, axis=-1) - target[np.arange(3), labels]
    reference_loss = (np.logaddexp(0, reference[:, 0]) - labels * reference[:, 0] if scenario == "binary_heads"
                      else logsumexp(reference, axis=-1) - reference[np.arange(3), labels])
    np.testing.assert_allclose(values, np.column_stack((original, original - reference_loss)), rtol=1e-6, atol=1e-7)


def test_custom_collation_preserves_record_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A collate function that reverses batches must preserve each score's record ID."""
    audit = create_audit(tmp_path, monkeypatch)
    audit.handler.population.augment = None
    audit.handler.dataloader_config.params["collate_fn"] = lambda batch: default_collate(batch[::-1])
    attack, = audit.attack_scheduler.attacks

    result, = audit.run_audit(create_pdf=False, use_optuna=False)

    ids = attack.audit_dataset["data"]
    np.testing.assert_array_equal(result.audit_indices, ids)
    inputs = torch.stack([attack.population[int(index)][0] for index in ids])
    logits = attack.target_model.get_logits(inputs)
    labels = attack.population.targets[ids].numpy().reshape(-1)
    expected = logsumexp(logits, axis=1) - logits[np.arange(len(ids)), labels]
    np.testing.assert_allclose(attack.signal_data[:, 0], expected, rtol=1e-6, atol=3e-7)


@pytest.mark.parametrize(("labels", "logits", "error"), [
    ([1, 1, 0], [[1000], [-1000], [0]], None), ([1, 1, 0], [1000, -1000, 0], None),
    ([0], [[np.nan, 0]], "non-finite"),
    ([0.5], [[0, 1]], "integer class labels"), ([2], [[0]], "binary labels"),
])
def test_classification_losses(labels: list, logits: list, error: Optional[str]) -> None:
    """Retain stable binary loss and meaningful invalid-output checks."""
    attack = unit_attack(labels)
    if error:
        with pytest.raises(ValueError, match=error):
            attack._get_mean_loss(QueryModel([logits]), np.arange(len(labels)))
    else:
        np.testing.assert_allclose(attack._get_mean_loss(QueryModel([logits]), np.arange(3)), [0, 1000, np.log(2)])


def test_probability_targets_from_query_collation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Losses follow randomly flipped inputs and their paired probability labels."""
    audit = create_audit(tmp_path, monkeypatch, num_classes=2)
    audit.handler.population.augment = None
    attack, = audit.attack_scheduler.attacks
    with torch.no_grad():
        attack.target_model.model_obj.linear.bias.zero_()

    def collate(batch):
        data, labels = default_collate(batch)
        flips = torch.rand(len(labels)) < 0.5
        data[flips] *= -1
        labels = labels ^ flips.long()
        return data, 0.8 * nn.functional.one_hot(labels, 2).float() + 0.1

    set_query_collation(audit.handler, collate, monkeypatch)
    result, = audit.run_audit(create_pdf=False, use_optuna=False)
    data, labels = zip(*(attack.population[int(index)] for index in result.audit_indices))
    raw = attack.target_model.get_logits(torch.stack(data))
    probabilities = 0.8 * np.eye(2)[list(labels)] + 0.1
    np.testing.assert_allclose(attack.signal_data[:, 0], logsumexp(raw, axis=1) - (raw * probabilities).sum(axis=1), atol=3e-7)


def test_query_augmentation_overrides_inherited_transform(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass the requested strength to the dataset and keep its normalization."""
    audit = create_audit(tmp_path, monkeypatch, augment_strength="easy")
    attack = audit.attack_scheduler.attacks[0]
    dataset = attack._get_query_dataloader(attack.audit_dataset["data"]).dataset
    assert dataset.augment_strength == "easy"
    assert dataset.augment is None and attack.population.augment is not None
    torch.testing.assert_close(dataset.mean, attack.population.mean)


@pytest.mark.parametrize(("options", "error", "message"), [
    ({"shadow_data_fraction": 0.01}, ValueError, "at least two auxiliary records"),
    ({"online": True, "training_data_fraction": 0.1}, ValueError, "both scorer splits"),
    ({}, RuntimeError, "target query failed"),
])
def test_preparation_failure_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: dict, error: type[Exception], message: str
) -> None:
    """Failed prepares restore parent state and leave no usable result."""
    audit = create_audit(tmp_path, monkeypatch, **options)
    attack, handler = audit.attack_scheduler.attacks[0], ShadowModelHandler()
    model, data, owner = attack.target_model, attack.audit_dataset, handler.handler
    paths = handler.storage_path, handler.attack_cache_folder_path
    if error is RuntimeError:
        def fail_query(_inputs: torch.Tensor) -> np.ndarray:
            model.model_obj.eval()
            raise RuntimeError("target query failed")

        monkeypatch.setattr(model, "get_logits", fail_query)
    elif "auxiliary" in message:
        monkeypatch.setattr(owner, "train", lambda *_args, **_kwargs: pytest.fail("Invalid pool trained a model"))
    batch_size = owner.dataloader_config.params["batch_size"]
    model.model_obj.train()
    model.model_obj.linear.eval()
    with pytest.raises(error, match=message):
        attack.prepare_attack()
    assert attack.target_model is model and attack.audit_dataset is data
    assert model.model_obj.training and not model.model_obj.linear.training
    assert owner.dataloader_config.params["batch_size"] == batch_size
    assert handler.handler is owner
    assert (handler.storage_path, handler.attack_cache_folder_path) == paths
    with pytest.raises(RuntimeError, match="prepare_attack"):
        attack.run_attack()


@pytest.mark.parametrize("scale", [1.0, 100.0])
def test_scorer_requires_both_features(scale: float) -> None:
    """Physically valid XOR loss features require both inputs, including large finite losses."""
    attack = unit_attack([0, 1])
    attack.batch_size = AttackRAPID.AttackConfig().batch_size
    torch.manual_seed(42)
    features = np.array([[scale, -scale], [scale, 0.5 * scale],
                         [3 * scale, -scale], [3 * scale, 0.5 * scale]] * 8, dtype=np.float32)
    membership = np.array([1, 0, 0, 1] * 8, dtype=np.float32)
    scorer = attack._train_mia_classifier(features[:-4], membership[:-4], features[-4:], membership[-4:])
    assert not scorer.training
    with torch.no_grad():
        scores = torch.sigmoid(scorer(torch.from_numpy(features))).numpy().reshape(-1)
    assert np.mean(scores[membership == 1]) > 0.9
    assert np.mean(scores[membership == 0]) < 0.1


def test_early_stopping_restores_best_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop after two worsening validation passes and restore the earlier checkpoint."""
    attack = unit_attack([0, 1])
    attack.mia_classifier_patience = 2
    features = np.array([[0.1, 0.5], [0.7, -0.1]], dtype=np.float32)
    labels = np.array([0, 1])
    sequential, criterion = nn.Sequential, nn.BCEWithLogitsLoss
    models, checkpoints = [], []
    validation_losses = iter([1.0, 0.8, 0.81, 0.82])

    def capture_model(*layers: nn.Module) -> nn.Module:
        model = sequential(*layers)
        models.append(model)
        return model

    class ControlledValidationLoss(criterion):
        def forward(self, output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            if torch.is_grad_enabled():
                return super().forward(output, targets)
            assert not models[0].training
            checkpoints.append(deepcopy(models[0].state_dict()))
            return output.new_tensor(next(validation_losses))

    monkeypatch.setattr(nn, "Sequential", capture_model)
    monkeypatch.setattr(nn, "BCEWithLogitsLoss", ControlledValidationLoss)
    scorer = attack._train_mia_classifier(features, labels, features + 1, labels)
    assert len(checkpoints) == 4
    assert any(not torch.equal(value, checkpoints[-1][key]) for key, value in checkpoints[1].items())
    for key, value in scorer.state_dict().items():
        torch.testing.assert_close(value, checkpoints[1][key], rtol=0, atol=0)


@pytest.mark.parametrize("selector", [None, "None", "RAPID", " RaPiD "])
def test_ordinary_setup_accepts_legacy_selectors(selector: Optional[str]) -> None:
    """Existing selectors for ordinary RAPID produce the default configuration."""
    config = AttackRAPID.AttackConfig(use_attack=selector)
    assert config.model_dump() == AttackRAPID.AttackConfig().model_dump()


@pytest.mark.parametrize("options", [
    {"use_attack": "lira"}, {"use_attack": "rmia"}, {"use_attack_output": "logits"},
    {"use_attack_config": {}}, {"online": True, "num_reference_models": 1},
    {"query_num": 0}, {"learning_rtae": 0.1},
])
def test_configuration_rejects_unsupported_settings(options: dict) -> None:
    """Alternate sources, invalid reference counts, and unknown settings raise config errors."""
    with pytest.raises(ValidationError):
        AttackRAPID.AttackConfig(**options)
