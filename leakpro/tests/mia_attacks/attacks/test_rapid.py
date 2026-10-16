#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Mini-scale RAPID audits and focused numerical, identity, and failure regressions."""

from copy import deepcopy
from inspect import getfile
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import yaml
from pydantic import ValidationError
from scipy.special import expit, logsumexp
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import RandomHorizontalFlip

from leakpro import LeakPro
from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.mia_attacks.attack_factory_mia import AttackFactoryMIA
from leakpro.attacks.mia_attacks.rapid import AttackRAPID
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.signals.signal import ModelLogits
from leakpro.tests.mia_attacks.attacks.test_all_attacks_end_to_end import (
    TinyImageInputHandler,
    TinyImageTargetModel,
    _build_tiny_image_population,
    _create_target_artifacts,
)


def create_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **options: object) -> LeakPro:
    """Reuse the runtime fixtures with nonconsecutive IDs, augmentation, and small training budgets."""
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(23)
    population, model = _build_tiny_image_population(), TinyImageTargetModel()
    population.augment = RandomHorizontalFlip()
    data_path, target_dir = _create_target_artifacts(
        tmp_path, population, model, nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr=0.01),
        "rapid_test", train_indices=[17, 4, 29, 5, 1], test_indices=[33, 9, 48, 11, 2]
    )
    config = {
        "audit": {"random_seed": 19, "attack_type": "mia", "data_modality": "image",
                  "output_dir": str(tmp_path / "output"), "attack_list": [
                      {"attack": "rapid", "num_reference_models": 2, "query_num": 2,
                       "mia_classifier_epochs": 3, "batch_size": 7, **options}]},
        "target": {"module_path": getfile(TinyImageTargetModel), "model_class": "TinyImageTargetModel",
                   "target_folder": str(target_dir), "data_path": str(data_path)},
        "shadow_model": {"epochs": 1},
    }
    path = tmp_path / "audit.yaml"
    path.write_text(yaml.safe_dump(config))
    return LeakPro(TinyImageInputHandler, str(path))


def check_training_and_features(
    attack: AttackRAPID, fit: Mock, features: Mock, source_rows: list, logits: list, online: bool
) -> None:
    """Check cached training identities and actual source inputs with independent per-record oracles."""
    ids = attack.audit_dataset["data"]
    shadow_call, target_call = features.call_args_list
    assert shadow_call.args[0] is not attack.target_model
    assert target_call.args[0] is attack.target_model
    assert set(attack.shadow_population).isdisjoint(ids)
    with attack._model_cache("shadow", attack.shadow_population) as models:
        metadata = models.get_shadow_model_metadata(attack.shadow_model_indices)[0]
        np.testing.assert_array_equal(fit.call_args.args[1], np.isin(attack.shadow_population, metadata.train_indices))
        cached, _ = models.get_shadow_models(attack.shadow_model_indices)
        for name, value in cached[0].model_obj.state_dict().items():
            torch.testing.assert_close(shadow_call.args[0].model_obj.state_dict()[name], value, rtol=0, atol=0)
    np.testing.assert_array_equal(fit.call_args.args[0], attack.shadow_features)
    if not attack.use_attack or attack.use_attack_output == "logits":
        with attack._model_cache("references", attack.reference_training_pool) as models:
            references = models.get_shadow_model_metadata(attack.reference_model_indices)
        if not online:
            assert set(attack.reference_population).isdisjoint(np.concatenate((ids, attack.shadow_population)))
        assert all(set(meta.train_indices).issubset(attack.reference_training_pool) for meta in references)
    for stage, values, indices in zip((0, 1), (attack.shadow_features, attack.signal_data), (attack.shadow_population, ids)):
        if attack.use_attack and attack.use_attack_output == "scores":
            mapping = dict(zip(*source_rows[stage]))
            np.testing.assert_allclose(values[:, 1], [mapping[index] for index in indices], rtol=1e-6)
        elif attack.use_attack:
            queried = np.array([values for rows, values in logits if np.array_equal(rows, indices)])
            assert len(queried) == attack.query_num
            labels = attack.handler.get_labels(indices).reshape(-1)
            losses = logsumexp(queried, axis=-1) - queried[:, :, np.arange(len(indices)), labels]
            outside = np.array([~np.isin(indices, meta.train_indices) for meta in references])
            assert outside.any(axis=0).all()
            if online:
                assert (~outside).any(axis=0).all()
            reference = np.where(outside, losses[:, 1:], 0).sum(axis=1) / outside.sum(axis=0)
            expected = np.column_stack((losses[:, 0].mean(axis=0), (losses[:, 0] - reference).mean(axis=0)))
            np.testing.assert_allclose(values, expected, rtol=1e-6, atol=5e-7)


def check_cached_replay(audit: LeakPro, attack: AttackRAPID, result: MIAResult, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cached replay changes reporting membership only, preserving features, fitted weights, and scores."""
    ids = attack.audit_dataset["data"].copy()
    before = deepcopy(attack.scoring_model.state_dict())
    shadow, target = attack.shadow_features.copy(), attack.signal_data.copy()
    train, test = audit.handler.train_indices.copy(), audit.handler.test_indices.copy()
    monkeypatch.setattr(audit.handler, "train_indices", test)
    monkeypatch.setattr(audit.handler, "test_indices", train)
    monkeypatch.setattr(audit.handler, "target_model_metadata", audit.handler.target_model_metadata.model_copy(
        update={"train_indices": test, "test_indices": train}, deep=True
    ))
    monkeypatch.setattr(AbstractMIA, "audit_dataset", {"data": ids, "in_members": np.arange(5, 10), "out_members": np.arange(5)})
    monkeypatch.setattr(audit.handler, "train", lambda *_args, **_kwargs: pytest.fail("Cached replay retrained a model"))
    attack.prepare_attack()
    repeated = attack.run_attack()
    np.testing.assert_array_equal(repeated.true, 1 - result.true)
    np.testing.assert_array_equal(repeated.signal_values, result.signal_values)
    np.testing.assert_array_equal(attack.shadow_features, shadow)
    np.testing.assert_array_equal(attack.signal_data, target)
    for name, value in attack.scoring_model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize(("source", "output", "online"), [(None, "scores", False)] + [
    (source, output, online) for source in ("lira", "rmia")
    for output in ("scores", "logits") for online in (False, True)
] + [("custom_source", output, False) for output in ("scores", "logits")])
def test_rapid_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: Optional[str], output: str, online: bool
) -> None:
    """Run real audits, verify model/label/feature provenance, then replay with swapped reporting labels."""
    monkeypatch.setitem(AttackFactoryMIA.attack_classes, "custom_source", AttackFactoryMIA.attack_classes["population"])
    audit = create_audit(tmp_path, monkeypatch, use_attack=source, use_attack_output=output,
                         use_attack_config={"online": online} if source in {"lira", "rmia"} else {})
    attack = audit.attack_scheduler.attacks[0]
    parent_model, parent_handler = attack.target_model, ShadowModelHandler().handler
    fit, features = Mock(wraps=attack._train_mia_classifier), Mock(wraps=attack._get_mia_features)
    monkeypatch.setattr(attack, "_train_mia_classifier", fit)
    monkeypatch.setattr(attack, "_get_mia_features", features)
    source_rows, logits = [], []
    query = ModelLogits.__call__

    def capture_logits(signal: ModelLogits, *args: object, **kwargs: object) -> np.ndarray:
        values = query(signal, *args, **kwargs)
        if kwargs.get("data_loader") is not None:
            logits.append((np.asarray(args[2]).copy(), values.copy()))
        return values

    monkeypatch.setattr(ModelLogits, "__call__", capture_logits)
    if source and output == "scores":
        cls, reuse = AttackFactoryMIA.attack_classes[source], Mock(wraps=attack._run_use_attack)
        run = cls.run_attack
        monkeypatch.setattr(attack, "_run_use_attack", reuse)

        def capture_source(child: AbstractMIA) -> MIAResult:
            model = reuse.call_args.args[0].model_obj
            assert child.handler.target_model is not model
            for name, value in model.state_dict().items():
                torch.testing.assert_close(child.handler.target_model.state_dict()[name], value, rtol=0, atol=0)
            result = run(child)
            source_rows.append((child.audit_dataset["data"].copy(), result.signal_values.copy()))
            return result

        monkeypatch.setattr(cls, "run_attack", capture_source)
    elif source:
        monkeypatch.setattr(AttackFactoryMIA.attack_classes[source], "run_attack",
                            lambda *_args: pytest.fail("Logits mode requested source scores"))
    with pytest.raises(RuntimeError, match="prepare_attack"):
        attack.run_attack()
    result = audit.run_audit(create_pdf=False, use_optuna=False)[0]
    ids = attack.audit_dataset["data"].copy()
    np.testing.assert_array_equal(ids, [17, 4, 29, 5, 1, 33, 9, 48, 11, 2])
    np.testing.assert_array_equal(result.true, [1] * 5 + [0] * 5)
    assert result.signal_values.shape == (10,)
    assert np.isfinite(result.signal_values).all()
    assert np.all((result.signal_values >= 0) & (result.signal_values <= 1))
    assert (tmp_path / "output" / "results" / result.id / "result.txt").exists()
    assert result.metadata == attack.configs.model_dump()
    assert attack.target_model is parent_model
    assert ShadowModelHandler().handler is parent_handler
    check_training_and_features(attack, fit, features, source_rows, logits, online)
    check_cached_replay(audit, attack, result, monkeypatch)


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
    attack.handler = SimpleNamespace(get_dataloader=lambda *_args, **kwargs: DataLoader(
        dataset, batch_size=kwargs["batch_size"], shuffle=kwargs["shuffle"]
    ))
    return attack


def test_loss_features_and_final_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    """Independent loss and probability oracles catch wrong calibration, row order, direction, and partial batches."""
    labels = [1, 0, 2]
    target = np.array([[[0, 3, -2], [2, 0, 1], [4, 3, 1]], [[1, 1, 0], [-1, 2, 0], [0, 0, 0]]])
    references = np.array([[[0, 0, 0], [0, 1, -1], [0, 1, 2]], [[3, 2, 1], [1, 0, 3], [1, 2, -2]]])
    attack, model = unit_attack(labels), QueryModel(list(target))
    reference_models = [QueryModel([values]) for values in references]
    values = attack._get_mia_features(model, reference_models, np.array([17, 4, 29]))
    original = (logsumexp(target, axis=-1) - target[:, np.arange(3), labels]).mean(axis=0)
    reference = (logsumexp(references, axis=-1) - references[:, np.arange(3), labels]).mean(axis=0)
    np.testing.assert_allclose(values, np.column_stack((original, original - reference)), rtol=1e-6, atol=1e-7)
    assert model.calls == 2
    assert all(ref.calls == 2 for ref in reference_models)
    attack.signal_data, attack.scoring_model = values, nn.Linear(2, 1)
    attack.scoring_model.load_state_dict({"weight": torch.tensor([[0.5, 2.0]]), "bias": torch.tensor([-0.25])})
    monkeypatch.setattr(AbstractMIA, "audit_dataset", {
        "data": np.array([17, 4, 29]), "in_members": np.array([2]), "out_members": np.array([0, 1])
    })
    result = attack.run_attack()
    np.testing.assert_allclose(result.signal_values, expit(values @ [0.5, 2.0] - 0.25), rtol=1e-6)
    np.testing.assert_array_equal(result.true, [0, 0, 1])


@pytest.mark.parametrize(("labels", "logits", "error"), [
    ([1, 1, 0], [[1000], [-1000], [0]], None), ([0], [[np.nan, 0]], "non-finite"),
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


@pytest.mark.parametrize(("scores", "ids", "error"), [
    (None, [17, 4], "per-sample scores"), ([0, np.nan], [17, 4], "one finite score"),
    ([0], [17, 4], "one finite score"), ([0, 1], None, "identify every requested sample"),
    ([0, 1], [17, 99], "identify every requested sample"), ([0, 1], [17, 17], "identify every requested sample"),
])
def test_source_score_validation(
    monkeypatch: pytest.MonkeyPatch, scores: Optional[list], ids: Optional[list], error: str
) -> None:
    """Reject aggregate, partial, nonfinite, unidentified, foreign, and duplicate source scores."""
    attack = unit_attack([0, 1])
    attack.use_attack = "population"
    result = MIAResult.from_full_scores([1, 0], [0, 1], result_name="source")
    result.signal_values = np.asarray(scores) if scores is not None else None
    monkeypatch.setattr(attack, "_run_use_attack", lambda *_args: (result, np.asarray(ids) if ids is not None else None))
    with pytest.raises(ValueError, match=error):
        attack._get_mia_features(QueryModel([[[0, 1], [1, 0]]]), [], np.array([17, 4]))


@pytest.mark.parametrize("mode", ["indexed", "unindexed", "reload"])
def test_source_result_order_and_reload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """Reordered scores need IDs; ordinary saved/loaded scores retain verifiable signal identity."""
    audit = create_audit(tmp_path, monkeypatch, use_attack="population")
    attack, cls = audit.attack_scheduler.attacks[0], AttackFactoryMIA.attack_classes["population"]
    run, expected = cls.run_attack, []

    def source_result(child: AbstractMIA) -> MIAResult:
        result = run(child)
        rows = child.audit_dataset["data"].copy()
        expected.append(dict(zip(rows, result.signal_values)))
        if mode == "reload":
            result.save(child, str(tmp_path))
            return MIAResult.load(str(tmp_path / "data_objects" / f"{child.attack_id}.json"))
        result = MIAResult.from_full_scores(result.true[::-1], result.signal_values[::-1], result_name="reordered")
        if mode == "indexed":
            result.sample_indices = rows[::-1]
        return result

    monkeypatch.setattr(cls, "run_attack", source_result)
    if mode == "unindexed":
        with pytest.raises(ValueError, match="identify every requested sample"):
            attack.prepare_attack()
    else:
        attack.prepare_attack()
        for values, ids, mapping in zip((attack.shadow_features, attack.signal_data),
                                       (attack.shadow_population, attack.audit_dataset["data"]), expected):
            np.testing.assert_allclose(values[:, 1], [mapping[index] for index in ids], rtol=1e-6)


@pytest.mark.parametrize(("options", "error", "message"), [
    ({"shadow_data_fraction": 0.99}, ValueError, "at least 1 point"),
    ({"use_attack": "lira", "use_attack_output": "logits", "training_data_fraction": 0.1,
      "use_attack_config": {"online": True}}, ValueError, "IN and OUT references"),
    ({"use_attack": "population"}, RuntimeError, "source failed"),
    ({}, FileExistsError, "cache failed"),
    ({"use_attack": "lira", "use_attack_output": "logits", "use_attack_config": {"online": True}},
     ValueError, "Cached online references"),
])
def test_preparation_failure_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: dict, error: type[Exception], message: str
) -> None:
    """Invalid pools fail before training; source exceptions restore the parent and leave no runnable stale scorer."""
    audit = create_audit(tmp_path, monkeypatch, **options)
    attack, handler = audit.attack_scheduler.attacks[0], ShadowModelHandler()
    model, data, owner = attack.target_model, attack.audit_dataset, handler.handler
    paths = handler.storage_path, handler.attack_cache_folder_path
    if error is RuntimeError:
        def fail(child: AbstractMIA) -> None:
            assert child.handler is not audit.handler
            assert child.handler.target_model is not audit.handler.target_model
            raise RuntimeError("source failed")
        monkeypatch.setattr(AttackFactoryMIA.attack_classes["population"], "prepare_attack", fail)
    elif error is FileExistsError:
        mkdir = Path.mkdir

        def fail_cache(path: Path, *args: object, **kwargs: object) -> None:
            if path.name == "attack_cache":
                raise FileExistsError("cache failed")
            mkdir(path, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", fail_cache)
    elif message.startswith("Cached"):
        torch.manual_seed(7)
        np.random.seed(7)
        indices = np.arange(attack.population_size)
        with attack._model_cache("references", indices) as models:
            refs = models.create_shadow_models(2, indices, training_fraction=0.5, sampling_method="random")
            counts = models.get_in_indices_mask(refs, indices).sum(axis=1)
        assert np.any((counts == 0) | (counts == 2))
    else:
        monkeypatch.setattr(audit.handler, "train", lambda *_args, **_kwargs: pytest.fail("Invalid pools trained a model"))
    with pytest.raises(error, match=message):
        attack.prepare_attack()
    assert attack.target_model is model
    assert attack.audit_dataset is data
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
    scorer = attack._train_mia_classifier(features, membership)
    assert not scorer.training
    with torch.no_grad():
        scores = torch.sigmoid(scorer(torch.from_numpy(features))).numpy().reshape(-1)
    assert np.mean(scores[membership == 1]) > 0.9
    assert np.mean(scores[membership == 0]) < 0.1


@pytest.mark.parametrize(("options", "expected"), [
    *[({"use_attack": name}, None) for name in (None, "None", "RAPID", " RaPiD ")],
    *[({"use_attack": name.upper()}, name) for name in AttackFactoryMIA.attack_classes if name != "rapid"],
    ({"use_attack": "unknown"}, False), ({"use_attack_output": "logits"}, False),
    ({"use_attack": "rmia", "use_attack_config": {"num_shadow_models": 3}}, False),
    ({"query_num": 0}, False), ({"learning_rtae": 0.1}, False),
])
def test_configuration(options: dict, expected: object) -> None:
    """Keep selectors dynamic, normalize fallbacks, and reject invalid ownership and query settings."""
    if expected is False:
        with pytest.raises(ValidationError):
            AttackRAPID.AttackConfig(**options)
    else:
        config = AttackRAPID.AttackConfig(**options)
        assert config.use_attack == expected
        if expected is None:
            assert config.model_dump() == AttackRAPID.AttackConfig().model_dump()
