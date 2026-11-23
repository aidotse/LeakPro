#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Integration checks for attack-owned cached-signal target rotation."""

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import yaml
from scipy.integrate import trapezoid
from scipy.special import logsumexp
from scipy.stats import norm
from torch import nn

from leakpro import LeakPro
from leakpro.attacks.attack_scheduler import AttackScheduler
from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.mia_attacks.attack_factory_mia import AttackFactoryMIA
from leakpro.attacks.robustness import RobustnessConfig, summarize_robustness
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.schemas import EvalOutput, LeakProConfig, LossConfig, MIAMetaDataSchema, ShadowModelConfig, TrainingOutput
from leakpro.attacks.utils.utils import softmax_logits
from leakpro.utils.save_load import hash_config, hash_model
from leakpro.utils.seed import seed_everything


class TinyModel(nn.Module):
    def __init__(self, num_classes=2, gain=1.0):
        super().__init__()
        self.linear = nn.Linear(2, num_classes)
        self.gain = gain

    def forward(self, inputs):
        return self.linear(inputs) * self.gain


class TinyHandler(MIAHandler):
    sample_shadow_indices = AbstractInputHandler.sample_shadow_indices

    class UserDataset(AbstractInputHandler.UserDataset):
        def __init__(self, data, targets, **kwargs):
            self.data, self.targets = data, targets

    def train(self, dataloader, model, criterion, optimizer, epochs):
        for _ in range(epochs):
            model.train()
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                criterion(model(inputs), labels).backward()
                optimizer.step()
        return TrainingOutput(model=model, metrics=self.eval(dataloader, model, criterion))

    def eval(self, dataloader, model, criterion):
        model.eval()
        with torch.no_grad():
            losses = [criterion(model(inputs), labels).item() for inputs, labels in dataloader]
        return EvalOutput(loss=float(np.mean(losses)), accuracy=0.5)


class TinyInputHandler(AbstractInputHandler):
    UserDataset = TinyHandler.UserDataset
    train = TinyHandler.train
    eval = TinyHandler.eval


def test_each_attack_defines_robustness_without_a_base_interface():
    assert not hasattr(AbstractMIA, "run_robustness")
    assert not hasattr(AbstractMIA, "validate_robustness")
    for name in ("lira", "base", "rmia"):
        assert "run_robustness" in AttackFactoryMIA.attack_classes[name].__dict__


@pytest.fixture
def handler(tmp_path, monkeypatch):
    monkeypatch.setenv("LEAKPRO_DEVICE", "cpu")
    # The factory clears class-level Optuna metadata for explicit hyperparameters.
    for name in ("lira", "base", "rmia"):
        for field in AttackFactoryMIA.attack_classes[name].AttackConfig.model_fields.values():
            monkeypatch.setattr(field, "json_schema_extra", field.json_schema_extra)
    torch.manual_seed(12)
    handler = object.__new__(TinyHandler)
    handler.population = TinyHandler.UserDataset(torch.randn(60, 2), torch.arange(60) % 2)
    handler.population_size = 60
    handler.train_indices = np.arange(20)
    handler.test_indices = np.arange(20, 40)
    handler.target_model_blueprint = TinyModel
    handler.target_model = TinyModel()
    handler._criterion = nn.CrossEntropyLoss()
    handler.configs = LeakProConfig(**{
        "audit": {"attack_type": "mia", "attack_list": [{"attack": "lira"}], "data_modality": "tabular",
                  "output_dir": str(tmp_path)},
        "target": {"module_path": __file__, "model_class": "TinyModel", "target_folder": str(tmp_path),
                   "data_path": str(tmp_path / "unused.pkl")},
    })
    handler.target_model_metadata = MIAMetaDataSchema(**{
        "train_indices": handler.train_indices.tolist(), "test_indices": handler.test_indices.tolist(), "num_train": 20,
        "optimizer": {"name": "sgd", "params": {"lr": 0.1}}, "criterion": {"name": "crossentropyloss"},
        "data_loader": {"params": {"batch_size": 10, "shuffle": False}}, "epochs": 1,
        "train_result": {"loss": 1.0, "accuracy": 0.5}, "test_result": {"loss": 1.0, "accuracy": 0.5}, "dataset": "tiny",
    })
    handler.dataloader_config = handler.target_model_metadata.data_loader
    handler.train(handler.get_dataloader(handler.train_indices), handler.target_model, handler.get_criterion(),
                  handler.get_optimizer(handler.target_model), 1)
    monkeypatch.setattr(MIAResult, "create_roc_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(MIAResult, "create_signal_histogram", lambda *args, **kwargs: None)
    yield handler
    ShadowModelHandler.delete_instance()
    AttackFactoryMIA.shadow_model_handler = None
    AttackFactoryMIA.distillation_model_handler = None


@pytest.mark.parametrize("name", ["lira", "base", "rmia"])
@pytest.mark.parametrize("online", [False, True])
def test_attack_owns_rotation_and_reuses_native_pool(handler, mocker, name, online):
    config = {"num_shadow_models": 4, "training_data_fraction": 0.5, "online": online}
    if name == "lira":
        config["var_calculation"] = "fixed"
    attack = AttackFactoryMIA.create_attack(name, config, handler)
    training = mocker.spy(TinyHandler, "train")
    preparation = mocker.spy(attack, "prepare_attack")
    creation = mocker.spy(AttackFactoryMIA, "create_attack")
    original_hash = hash_model(handler.target_model)
    original_config = handler.configs.model_dump()
    original_metadata = handler.target_model_metadata.model_dump()
    runs = attack.run_robustness(n_runs=3, seed=123)
    preparation.assert_called_once_with()
    assert training.call_count == 4
    assert creation.call_count == 0
    assert all(len(call.args[1].dataset) == 30 for call in training.call_args_list)
    assert handler.configs.model_dump() == original_config
    assert handler.target_model_metadata.model_dump() == original_metadata
    assert hash_model(handler.target_model) == original_hash
    assert ShadowModelHandler().handler is handler
    np.testing.assert_array_equal(handler.train_indices, np.arange(20))
    membership = {-1: handler.train_indices}
    membership.update((int(index), meta.train_indices) for index, meta in zip(
        attack.shadow_model_indices, ShadowModelHandler().get_shadow_model_metadata(attack.shadow_model_indices)))
    assert len({result.metadata["robustness"]["target"] for result in runs}) == 3
    for result in runs:
        provenance = result.metadata["robustness"]
        target = provenance["target"]
        complement = set(range(handler.population_size)) - set(membership[target])
        assert provenance["references"] == [index for index, split in membership.items()
                                             if index != target and set(split) != complement]
        assert provenance["target_training_size"] == len(membership[target])
        assert provenance["reference_training_sizes"] == [len(membership[index]) for index in provenance["references"]]
        np.testing.assert_array_equal(result.true, np.isin(result.audit_indices, membership[target]))
        assert result.result.id == result.id
        assert np.isfinite(result.signal_values).all()
    # The original target must use exactly the same scoring rule and prepared values.
    ordinary = attack.run_attack()
    original = original_rotation(runs)
    np.testing.assert_array_equal(original.audit_indices, ordinary.audit_indices)
    np.testing.assert_allclose(original.signal_values, ordinary.signal_values)
    directory = Path(handler.configs.audit.output_dir) / "robustness" / runs[0].metadata["robustness"]["experiment_id"]
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["setup"]["targets"] == [r.metadata["robustness"]["target"] for r in runs]
    assert summary["setup"]["training_sizes"] == [20, 30, 30, 30, 30]


@pytest.mark.parametrize("name", ["lira", "base", "rmia"])
def test_cached_pool_seed_and_scores_repeat(handler, mocker, name):
    config = {"num_shadow_models": 4, "training_data_fraction": 0.5, "online": True}
    first = AttackFactoryMIA.create_attack(name, config, handler)
    original = first.run_robustness(n_runs=3, seed=7)
    training = mocker.spy(TinyHandler, "train")
    second = AttackFactoryMIA.create_attack(name, config, handler)
    repeated = second.run_robustness(n_runs=3, seed=7)
    assert training.call_count == 0
    for before, after in zip(original, repeated):
        assert before.metadata["robustness"] == after.metadata["robustness"]
        np.testing.assert_allclose(before.signal_values, after.signal_values)


@pytest.mark.parametrize("name", ["lira", "base", "rmia"])
def test_original_result_carries_aggregate_through_save_load(handler, mocker, tmp_path, name):
    attack = AttackFactoryMIA.create_attack(name, {"num_shadow_models": 4, "online": True}, handler)
    saving = mocker.spy(MIAResult, "save")
    results = attack.run_robustness(n_runs=3, seed=17)
    original = original_rotation(results)
    directory = Path(handler.configs.audit.output_dir) / "robustness" / original.metadata["robustness"]["experiment_id"]
    summary = json.loads((directory / "summary.json").read_text())["rotated_target_summary"]
    assert original.robustness == original.result.robustness == summary
    assert original.id == original.result.id == f"{original.result_name}-{hash_config(original.metadata)[:16]}"
    assert saving.call_count == len(results)
    for result in results:
        if result is not original:
            assert result.result.robustness is None
    path = next((directory / f"run_{results.index(original)}" / "data_objects").glob("*.json"))
    assert MIAResult.load(str(path)).robustness == summary
    payload = json.loads(path.read_text())
    assert payload.pop("robustness") == summary
    legacy = tmp_path / "legacy-result.json"
    legacy.write_text(json.dumps(payload))
    assert MIAResult.load(str(legacy)).robustness is None


@pytest.mark.parametrize("name", ["lira", "base", "rmia"])
def test_rotations_do_not_train_load_models_or_run_forward_passes(handler, monkeypatch, mocker, name):
    attack = AttackFactoryMIA.create_attack(name, {"num_shadow_models": 4, "online": True}, handler)
    scorer = "_compute_scores" if name == "rmia" else "_score_signals"
    score = getattr(attack, scorer)
    calls = []

    def check(*args):
        # Preparation has finished when scoring starts. Every rotation must use arrays alone.
        monkeypatch.setattr(TinyHandler, "train", lambda *args: pytest.fail("retraining during rotation"))
        monkeypatch.setattr(ShadowModelHandler(), "_load_shadow_model", lambda *args: pytest.fail("model loading during rotation"))
        monkeypatch.setattr(TinyModel, "forward", lambda *args: pytest.fail("forward pass during rotation"))
        calls.append(args)
        return score(*args)

    monkeypatch.setattr(attack, scorer, check)
    assert len(attack.run_robustness(n_runs=3, seed=19)) == 3
    assert len(calls) == 3


@pytest.mark.parametrize("name", ["population", "qmia", "HSJ", "multi_signal_lira"])
def test_other_attacks_explicitly_reject_robustness_before_training(handler, mocker, name):
    handler.configs.audit.attack_list = [{"attack": name, "robustness": {"n_runs": 2}}]
    training = mocker.spy(TinyHandler, "train")
    with pytest.raises(ValueError, match="does not support cached-signal robustness"):
        AttackScheduler(handler, output_dir=handler.configs.audit.output_dir)
    assert training.call_count == 0


@pytest.fixture
def saved_audit_inputs(handler, tmp_path):
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    torch.save(handler.target_model.state_dict(), target_dir / "target_model.pkl")
    with (target_dir / "model_metadata.pkl").open("wb") as stream:
        pickle.dump(handler.target_model_metadata, stream)
    with (tmp_path / "population.pkl").open("wb") as stream:
        pickle.dump(handler.population, stream)
    config = handler.configs.model_dump()
    config["target"]["target_folder"] = str(target_dir)
    config["target"]["data_path"] = str(tmp_path / "population.pkl")
    config["audit"]["output_dir"] = str(tmp_path / "audit")
    return config, tmp_path / "audit.yaml"


@pytest.mark.parametrize("field", ["optimizer", "criterion"])
@pytest.mark.parametrize("params", ["omitted", None, {}, "explicit"])
def test_metadata_parameter_defaults_through_yaml(saved_audit_inputs, field, params):
    config, config_path = saved_audit_inputs
    metadata_path = Path(config["target"]["target_folder"]) / "model_metadata.pkl"
    with metadata_path.open("rb") as stream:
        metadata = pickle.load(stream).model_dump()
    if params == "omitted":
        metadata[field].pop("params")
        params = {}
    else:
        if params == "explicit":
            params = {"lr": 0.1} if field == "optimizer" else {"label_smoothing": 0.1}
        metadata[field]["params"] = params
    validated = MIAMetaDataSchema.model_validate(metadata)
    with metadata_path.open("wb") as stream:
        pickle.dump(validated, stream)
    config["audit"]["attack_list"] = [{"attack": "lira", "num_shadow_models": 4,
                                       "training_data_fraction": 1 / 3, "online": True,
                                       "robustness": {"n_runs": 2}}]
    config_path.write_text(yaml.safe_dump(config))
    audit = LeakPro(TinyInputHandler, str(config_path))
    original = audit.handler.target_model_metadata.model_dump()
    results = audit.run_audit()
    assert len(results) == 2
    assert getattr(audit.handler.target_model_metadata, field).params == params
    assert audit.handler.target_model_metadata.model_dump() == original
    if params:
        if field == "optimizer":
            assert audit.handler.get_optimizer(audit.handler.target_model).defaults["lr"] == params["lr"]
        else:
            assert audit.handler.get_criterion().label_smoothing == params["label_smoothing"]


@pytest.mark.parametrize("name", ["lira", "base", "rmia"])
def test_yaml_runs_actual_attacks_with_mixed_training_sizes(saved_audit_inputs, mocker, caplog, name):
    config, config_path = saved_audit_inputs
    attack_config = {"num_shadow_models": 4, "training_data_fraction": 0.5, "online": True}
    config["audit"]["random_seed"] = 11
    config["audit"]["attack_list"] = [
        {"attack": "unknown"},
        {"attack": "lira", **attack_config, "robustness": None},
        {"attack": name, **attack_config, "robustness": {}},
        {"attack": "base", **attack_config},
    ]
    config_path.write_text(yaml.safe_dump(config))
    training = mocker.spy(TinyInputHandler, "train")
    report = mocker.patch("leakpro.leakpro.ReportHandler")
    audit = LeakPro(TinyInputHandler, str(config_path))
    results = audit.run_audit(create_pdf=True)
    assert len(results) == 6
    assert len({result.id for result in results}) == 6
    assert all(result.result.id == result.id for result in results)
    attack = audit.attack_scheduler.attacks[1]
    assert isinstance(attack, AttackFactoryMIA.attack_classes[name])
    assert training.call_count == 4  # All entries reuse the attack's ordinary pool.
    assert "training sizes differ" in caplog.text
    assert audit.handler.configs.audit.attack_list == config["audit"]["attack_list"]
    directory = Path(config["audit"]["output_dir"]) / "robustness" / results[1].metadata["robustness"]["experiment_id"]
    saved = json.loads((directory / "summary.json").read_text())
    assert results[1].robustness == saved["rotated_target_summary"]
    assert saved["setup"]["targets"] == [0, 3, 1, 2]
    assert saved["setup"]["references"] == [[-1, 2, 3], [-1, 0, 1], [-1, 2, 3], [-1, 0, 1]]
    assert all(run["target_training_size"] == 30 for run in saved["runs"])
    assert all(run["reference_training_sizes"] == [20, 30, 30] for run in saved["runs"])
    report.assert_called_once_with(results=results, report_dir=audit.report_dir)
    report.return_value.create_report.assert_called_once_with()
    report_dir = Path(config["audit"]["output_dir"]) / "report-check"
    MIAResult.create_results(results, save_dir=str(report_dir))
    assert all((report_dir / result.id).is_dir() for result in results)


def test_seeded_rmia_experiments_keep_native_ids_and_saved_reports(saved_audit_inputs, mocker):
    config, config_path = saved_audit_inputs
    config["audit"]["attack_list"] = [
        {"attack": "rmia", "num_shadow_models": 4, "online": True, "gamma": 1.0,
         "robustness": {"n_runs": 3, "seed": seed}} for seed in (17, 29)
    ]
    config_path.write_text(yaml.safe_dump(config))
    results = LeakPro(TinyInputHandler, str(config_path)).run_audit()
    assert len(results) == len({result.id for result in results}) == 6
    first = {item.metadata["robustness"]["target"]: item for item in results[:3]}
    second = {item.metadata["robustness"]["target"]: item for item in results[3:]}
    assert -1 in first and -1 not in second
    for target in first.keys() & second.keys():
        np.testing.assert_array_equal(first[target].signal_values, second[target].signal_values)
    assert [r.metadata["robustness"]["target"] for r in results[:3]] != [
        r.metadata["robustness"]["target"] for r in results[3:]]
    for result in results:
        assert result.id == result.result.id == f"RMIA-{hash_config(result.metadata)[:16]}"
    root = Path(config["audit"]["output_dir"])
    loaded = [MIAResult.load(str(path)) for path in root.glob("robustness/*/run_*/data_objects/*.json")]
    assert {result.id for result in loaded} == {result.id for result in results}
    plot = mocker.spy(MIAResult, "create_roc_plot")
    MIAResult.create_results(loaded, save_dir=str(root / "report-check"))
    overlays = [call for call in plot.call_args_list if "rotation_summary" in call.kwargs]
    assert len(overlays) == 2
    for call in overlays:
        group = call.args[0]
        assert len(group) == 3
        assert len({item.metadata["robustness"]["experiment_id"] for item in group}) == 1
    assert all((root / "report-check" / result.id).is_dir() for result in results)


@pytest.mark.parametrize("seed", [7, None])
def test_yaml_seed_override_and_output_directory(handler, tmp_path, seed):
    handler.configs.audit.attack_list = [{"attack": "lira", "num_shadow_models": 4,
                                         "robustness": {"n_runs": 2, "seed": seed}}]
    output_dir = tmp_path / "custom"
    scheduler = AttackScheduler(handler, output_dir=str(output_dir))
    records = scheduler.run_attacks()
    attack = scheduler.attacks[0]
    assert scheduler.robustness_options[attack]["seed"] is seed
    assert len(records) == 2
    assert all(entry["attack_type"] == "lira" and entry["attack_object"] is attack for entry in records)
    directory = output_dir / "robustness" / records[0]["result_object"].metadata["robustness"]["experiment_id"]
    assert (directory / "summary.json").is_file()


@pytest.mark.parametrize("options", [
    False, {"n_runs": True}, {"n_runs": 1}, {"n_runs": 6},
    {"n_runs": 2, "seed": -1}, {"n_runs": 2, "typo": 4},
])
def test_invalid_yaml_robustness_rejected_before_training(handler, mocker, options):
    handler.configs.audit.attack_list = [{"attack": "lira", "num_shadow_models": 4,
                                         "robustness": options}]
    training = mocker.spy(TinyHandler, "train")
    with pytest.raises(ValueError):
        AttackScheduler(handler, output_dir=handler.configs.audit.output_dir)
    assert training.call_count == 0


def test_robustness_rejects_non_mia_before_factory_initialization(handler, mocker):
    handler.configs.audit.attack_list = [{"attack": "lira", "num_shadow_models": 4,
                                         "training_data_fraction": 1 / 3,
                                         "robustness": {"n_runs": 2}}]
    handler.configs.audit.attack_type = "minv"
    initialize = mocker.spy(AttackScheduler, "_initialize_factory")
    with pytest.raises(ValueError, match="only supported for MIA"):
        AttackScheduler(handler, output_dir=handler.configs.audit.output_dir)
    assert initialize.call_count == 0


def test_global_optuna_flag_tunes_only_ordinary_entries(handler, mocker):
    handler.configs.audit.attack_list = [
        {"attack": "rmia", "num_shadow_models": 4, "online": True, "robustness": {"n_runs": 3}},
        {"attack": "rmia", "num_shadow_models": 4, "online": True},
    ]
    scheduler = AttackScheduler(handler, output_dir=handler.configs.audit.output_dir)
    robust, ordinary = scheduler.attacks
    # Exercise both active-tuner branches without running a numerical Optuna study.
    robust.optuna_params = ordinary.optuna_params = 1
    robust_tuning = mocker.patch.object(robust, "run_with_optuna")
    ordinary_tuning = mocker.patch.object(
        ordinary, "run_with_optuna", return_value=SimpleNamespace(best_params={"temperature": 3.0}))
    training = mocker.spy(TinyHandler, "train")
    expected = scheduler.run_attacks(use_optuna=False)
    robust_config = robust.configs.model_dump()
    actual = scheduler.run_attacks(use_optuna=True)
    robust_tuning.assert_not_called()
    ordinary_tuning.assert_called_once_with()
    assert robust.configs.model_dump() == robust_config
    assert actual[-1]["result_object"].metadata["temperature"] == 3.0
    assert len(actual) == len(expected) == 4
    assert training.call_count == 4
    for before, after in zip(expected[:3], actual[:3]):
        assert before["result_object"].id == after["result_object"].id
        np.testing.assert_array_equal(before["result_object"].signal_values, after["result_object"].signal_values)


def test_summary_includes_first_run_and_uses_upper_roc_ties():
    original = MIAResult.from_full_scores([0, 0, 1, 1], [4, 3, 2, 1], result_name="test")
    rotated = MIAResult.from_full_scores([1, 1, 0, 1, 0], [5, 4, 3, 2, 1], result_name="test")
    summary = summarize_robustness([original, rotated, rotated])
    assert summary["roc_auc"]["mean"] == pytest.approx((original.roc_auc + 2 * rotated.roc_auc) / 3)
    assert summary["roc_auc"]["n_total"] == 3
    assert summary["roc"]["fpr"][:2] == [0.0, 0.0]
    np.testing.assert_allclose(summary["roc"]["mean_tpr"][:2], [0.0, 4 / 9])
    np.testing.assert_allclose(np.asarray(summary["roc"]["pointwise_tpr_percentiles_2_5_97_5"])[:, 1],
                               [1 / 30, 2 / 3])


@pytest.mark.parametrize("all_tied", [False, True])
def test_summary_includes_roc_origin_for_tied_scores(all_tied):
    tied = MIAResult.from_full_scores([1, 0, 1, 0], [2, 2, 1, 1], result_name="tied")
    other = tied if all_tied else MIAResult.from_full_scores([1, 0, 1, 0], [4, 3, 2, 1], result_name="untied")
    summary = summarize_robustness([tied, other])["roc"]
    assert summary["fpr"][0] == 0.0
    assert summary["mean_tpr"][0] == 0.0
    upper = np.searchsorted(summary["fpr"], 0.0, side="right") - 1
    assert summary["mean_tpr"][upper] == pytest.approx(0.0 if all_tied else 0.25)
    expected = [0.0, 0.0] if all_tied else [0.0125, 0.4875]
    np.testing.assert_allclose(np.array(summary["pointwise_tpr_percentiles_2_5_97_5"])[:, upper], expected)


@pytest.mark.parametrize("name", ["lira", "base", "rmia"])
@pytest.mark.parametrize("online", [False, True])
def test_rotated_scores_match_independent_equations(handler, name, online):
    # Noncontiguous records expose positional membership and cached-row mistakes.
    records = np.random.default_rng(45).permutation(60)[:40]
    handler.train_indices, handler.test_indices = records[:20], records[20:]
    handler.target_model_metadata.train_indices = handler.train_indices.tolist()
    handler.target_model_metadata.test_indices = handler.test_indices.tolist()
    handler.target_model = TinyModel()
    handler.train(handler.get_dataloader(handler.train_indices), handler.target_model, handler.get_criterion(),
                  handler.get_optimizer(handler.target_model), 1)
    config = {"num_shadow_models": 4, "online": online}
    if name == "lira":
        config["var_calculation"] = "carlini"
    if name == "rmia":
        config["gamma"] = 1.0
    attack = AttackFactoryMIA.create_attack(name, config, handler)
    runs = attack.run_robustness(n_runs=3, seed=29)
    pool = ShadowModelHandler()
    ids = [-1, *map(int, attack.shadow_model_indices)]
    logits = [pool.load_logits(name=f"target_{pool.target_model_hash}")]
    logits.extend(pool.load_logits(indx=index) for index in attack.shadow_model_indices)
    labels = handler.get_labels(records)
    training = [handler.train_indices, *[meta.train_indices for meta in pool.get_shadow_model_metadata(attack.shadow_model_indices)]]
    membership = np.array([np.isin(records, indices) for indices in training])
    values = (np.array([attack.signal(raw, labels) for raw in logits]) if name == "lira" else
              np.array([softmax_logits(raw, attack.temperature)[np.arange(len(labels)), labels] for raw in logits]))
    for result in runs:
        target = ids.index(result.metadata["robustness"]["target"])
        references = expected_reference_rows(training, target)
        out = ~membership[references]
        kept = np.ones(len(records), dtype=bool)
        if name == "lira":
            kept = out.any(axis=0) & ((~out).any(axis=0) if online else True)
            reference_values, reference_out = values[references][:, kept], out[:, kept]
            out_std = np.std(reference_values[reference_out])
            in_std = np.std(reference_values[~reference_out]) if online else None
            expected = []
            for column, target_value in enumerate(values[target, kept]):
                v, mask = reference_values[:, column], reference_out[:, column]
                score = -norm.logpdf(target_value, v[mask].mean(), out_std + 1e-30)
                if online:
                    score += norm.logpdf(target_value, v[~mask].mean(), in_std + 1e-30)
                expected.append(score)
            expected = np.array(expected)
        elif name == "base":
            log_conf = np.log(values + attack.epsilon)
            ref_conf = log_conf[references]
            threshold = (logsumexp(ref_conf, axis=0) - np.log(len(references)) if online else
                         logsumexp(np.where(out, ref_conf, -np.inf), axis=0) - np.log(out.sum(axis=0)))
            expected = log_conf[target] - (threshold if online else attack.offline_scale_factor * threshold)
        else:
            z_logits = [attack.logits_z_theta, *attack.logits_z_shadow_models]
            z_prob = np.array([softmax_logits(raw, attack.temperature)[np.arange(len(attack.z_labels)), attack.z_labels]
                               for raw in z_logits])
            p_z = z_prob[references].mean(axis=0)
            p_x = values[references].mean(axis=0) if online else np.nanmean(np.where(out, values[references], np.nan), axis=0)
            if not online:
                p_z = 0.5 * ((attack.offline_a + 1) * p_z + (1 - attack.offline_a))
                p_x = 0.5 * ((attack.offline_a + 1) * p_x + (1 - attack.offline_a))
            ratio_z = z_prob[target] / (p_z + attack.epsilon)
            ratio_x = values[target] / (p_x + attack.epsilon)
            expected = (ratio_x[:, None] / ratio_z[None] > attack.gamma).mean(axis=1)
        expected_by_record = dict(zip(records[kept], expected))
        np.testing.assert_allclose(result.signal_values.ravel(), [expected_by_record[index] for index in result.audit_indices])


@pytest.mark.parametrize("criterion", [nn.MSELoss(), nn.L1Loss(), nn.HuberLoss()])
def test_rmia_regression_uses_reference_only_scale_without_changing_ordinary_state(handler, criterion):
    handler.population.targets = torch.randn(60, 2)
    handler._criterion = criterion
    handler.target_model_metadata.criterion = LossConfig(name=type(criterion).__name__.lower())
    handler.target_model = TinyModel()
    handler.train(handler.get_dataloader(handler.train_indices), handler.target_model, criterion,
                  handler.get_optimizer(handler.target_model), 1)
    # Audit and auxiliary records can overlap, producing ratios near 1.
    # Keep those comparisons away from the cutoff in this scale test.
    attack = AttackFactoryMIA.create_attack("rmia", {"num_shadow_models": 4, "online": True, "gamma": 1.1}, handler)
    runs = attack.run_robustness(n_runs=3, seed=18)
    ordinary_scale = attack._scale
    ids = [-1, *map(int, attack.shadow_model_indices)]
    audit_values = np.array([attack.logits_theta, *attack.logits_shadow_models])
    z_values = np.array([attack.logits_z_theta, *attack.logits_z_shadow_models])
    training = [handler.train_indices, *(meta.train_indices for meta in
                ShadowModelHandler().get_shadow_model_metadata(attack.shadow_model_indices))]
    for result in runs:
        target = ids.index(result.metadata["robustness"]["target"])
        references = expected_reference_rows(training, target)
        z_residual = z_values[references] - attack.z_labels
        if isinstance(criterion, nn.MSELoss):
            scale = (z_residual ** 2).mean()
            energy = lambda residual: np.mean(residual ** 2, axis=1) / (2 * scale)
        elif isinstance(criterion, nn.L1Loss):
            scale = np.abs(z_residual).mean()
            energy = lambda residual: np.mean(np.abs(residual), axis=1) / scale
        else:
            huber = lambda r: np.where(np.abs(r) <= 1, r ** 2 / 2, np.abs(r) - 0.5)
            scale = huber(z_residual).mean()
            energy = lambda residual: huber(residual).mean(axis=1) / scale
        p_x = np.array([np.exp(-energy(raw - attack.ground_truth)) for raw in audit_values])
        p_z = np.array([np.exp(-energy(raw - attack.z_labels)) for raw in z_values])
        ratio_x = p_x[target] / (p_x[references].mean(axis=0) + attack.epsilon)
        ratio_z = p_z[target] / (p_z[references].mean(axis=0) + attack.epsilon)
        expected = (ratio_x[:, None] / ratio_z[None] > attack.gamma).mean(axis=1)
        by_record = dict(zip(attack.audit_dataset["data"], expected))
        np.testing.assert_allclose(result.signal_values.ravel(), [by_record[index] for index in result.audit_indices])
    assert attack._scale == ordinary_scale
    np.testing.assert_allclose(original_rotation(runs).signal_values, attack.run_attack().signal_values)


@pytest.mark.parametrize("name", ["lira", "base", "rmia"])
def test_failed_rotation_removes_stale_summary_and_preserves_handler(handler, monkeypatch, name):
    attack = AttackFactoryMIA.create_attack(name, {"num_shadow_models": 4, "online": True}, handler)
    runs = attack.run_robustness(n_runs=3, seed=7)
    directory = Path(handler.configs.audit.output_dir) / "robustness" / runs[0].metadata["robustness"]["experiment_id"]
    summary_path = directory / "summary.json"
    assert summary_path.exists()
    scorer = "_compute_scores" if name == "rmia" else "_score_signals"
    score = getattr(attack, scorer)
    calls = []

    def fail_second(*args):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("injected rotation failure")
        return score(*args)

    monkeypatch.setattr(attack, scorer, fail_second)
    original_model = handler.target_model
    with pytest.raises(RuntimeError, match="injected rotation failure"):
        attack.run_robustness(n_runs=3, seed=7)
    assert not summary_path.exists()
    assert handler.target_model is original_model
    assert ShadowModelHandler().handler is handler
    np.testing.assert_array_equal(handler.train_indices, np.arange(20))


def test_rmia_preserves_float64_strict_threshold(handler):
    attack = AttackFactoryMIA.create_attack("rmia", {"online": True, "num_shadow_models": 2, "gamma": 1.0}, handler)
    attack.epsilon = 0
    scores = attack._compute_scores(np.array([[1 + 1e-12, 1.0, 1 - 1e-12]]), np.ones((2, 3)), None, np.array([[1.0]]))
    np.testing.assert_array_equal(scores, [1.0, 0.0, 0.0])


@pytest.mark.parametrize("overrides", [
    {"epochs": 2}, {"batch_size": 5}, {"init_params": {"num_classes": 3}},
    {"optimizer": {"name": "sgd", "params": {"lr": 0.2}}},
    {"criterion": {"name": "crossentropyloss", "params": {"label_smoothing": 0.1}}},
])
def test_training_setup_still_matches_before_training(handler, mocker, overrides):
    handler.configs.shadow_model = ShadowModelConfig(**overrides)
    handler.configs.audit.attack_list = [{"attack": "lira", "num_shadow_models": 4, "robustness": {"n_runs": 2}}]
    training = mocker.spy(TinyHandler, "train")
    with pytest.raises(ValueError, match="same target and shadow training setup"):
        AttackScheduler(handler, output_dir=handler.configs.audit.output_dir)
    assert training.call_count == 0



def test_lira_new_target_in_existing_output_uses_current_cached_logits(handler):
    config = {"num_shadow_models": 4, "online": True}
    first = AttackFactoryMIA.create_attack("lira", config, handler)
    first.run_robustness(n_runs=2, seed=17)
    old_hash = hash_model(handler.target_model)
    with torch.no_grad():
        handler.target_model.linear.weight.copy_(torch.tensor([[4., -3.], [-4., 3.]]))
        handler.target_model.linear.bias.copy_(torch.tensor([2., -2.]))
    assert hash_model(handler.target_model) != old_hash
    second = AttackFactoryMIA.create_attack("lira", config, handler)
    second.run_robustness(n_runs=2, seed=17)
    pool = ShadowModelHandler()
    current_logits = pool.load_logits(name=f"target_{pool.target_model_hash}")
    expected = second.signal(current_logits, handler.get_labels(second.audit_dataset["data"]))
    np.testing.assert_allclose(second._unfiltered_signals[0], expected)


def test_seeded_rmia_follows_ordinary_lifecycle_with_same_native_pool(handler, mocker):
    config = {"num_shadow_models": 4, "online": True, "gamma": 1.0}
    used = AttackFactoryMIA.create_attack("rmia", config, handler)
    used.prepare_attack()
    used.run_attack()
    training = mocker.spy(TinyHandler, "train")
    before = used.run_robustness(n_runs=3, seed=17)
    fresh = AttackFactoryMIA.create_attack("rmia", config, handler)
    after = fresh.run_robustness(n_runs=3, seed=17)
    assert training.call_count == 0
    np.testing.assert_array_equal(used.shadow_model_indices, fresh.shadow_model_indices)
    np.testing.assert_array_equal(used.attack_data_indices, fresh.attack_data_indices)
    for old, new in zip(before, after):
        assert old.metadata["robustness"] == new.metadata["robustness"]
        np.testing.assert_array_equal(old.signal_values, new.signal_values)


@pytest.mark.parametrize("name", ["lira", "base", "rmia"])
def test_preparation_failure_keeps_prior_artifacts(handler, monkeypatch, name):
    attack = AttackFactoryMIA.create_attack(name, {"num_shadow_models": 4, "online": True}, handler)
    runs = attack.run_robustness(n_runs=3, seed=17)
    directory = Path(handler.configs.audit.output_dir) / "robustness" / runs[0].metadata["robustness"]["experiment_id"]
    completed = directory / "summary.json"
    prior_summary = completed.read_bytes()

    def fail(*args, **kwargs):
        raise RuntimeError("injected preparation failure")

    monkeypatch.setattr(attack, "prepare_attack", fail)
    with pytest.raises(RuntimeError, match="injected preparation failure"):
        attack.run_robustness(n_runs=2, seed=19)
    assert completed.read_bytes() == prior_summary


def test_rmia_rejects_fraction_that_selects_no_z_points(handler):
    attack = AttackFactoryMIA.create_attack(
        "rmia", {"num_shadow_models": 2, "online": True, "attack_data_fraction": 0.01}, handler)
    with pytest.raises(ValueError, match="Dataset indices are empty"):
        attack.run_robustness(n_runs=2, seed=17)


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("from_disk", [False, True])
def test_report_handles_tied_results_without_plot_mocks(tmp_path, mixed, from_disk):
    tied = MIAResult.from_full_scores([1, 1, 0, 0], [0.5] * 4, result_name="tied")
    results = [tied]
    if mixed:
        results.append(MIAResult.from_full_scores([1, 1, 0, 0], [4, 3, 2, 1], result_name="scored"))
    with plt.rc_context({"figure.figsize": (2.5, 2.5)}):
        if from_disk:
            saved_dir = tmp_path / "saved"
            for index, result in enumerate(results):
                result.save(attack_obj=SimpleNamespace(attack_id=result.id), output_dir=str(saved_dir))
                results[index] = MIAResult.load(str(saved_dir / "data_objects" / f"{result.id}.json"))
        tex = MIAResult.create_results(results, save_dir=str(tmp_path))
    assert tex.count("N/A") >= 4
    assert (tmp_path / "ROC.png").is_file()
    assert (tmp_path / tied.id / "ROC.png").is_file()


def test_offline_rmia_preserves_float32_marginal_at_strict_threshold(handler):
    attack = AttackFactoryMIA.create_attack("rmia", {"num_shadow_models": 3}, handler)
    target = np.array([[0.3, 0.7]], dtype=np.float32)
    references = np.array([[0.13, 0.2], [0.36, 0.3], [0.61, 0.4]], dtype=np.float32)
    out = np.array([[True, True], [True, False], [True, True]])
    # Reference the original masked-mean arithmetic, including its float32 rounding.
    original_mean = np.nanmean(np.where(out, references, np.nan), axis=0)
    original_ratio = target / (0.5 * ((attack.offline_a + 1) * original_mean + (1 - attack.offline_a)) + attack.epsilon)
    promoted_mean = np.where(out, references, 0).sum(axis=0) / out.sum(axis=0)
    promoted_ratio = target / (0.5 * ((attack.offline_a + 1) * promoted_mean + (1 - attack.offline_a)) + attack.epsilon)
    attack.gamma = float((float(original_ratio[0, 0]) + promoted_ratio[0, 0]) / 2)
    expected = (original_ratio.astype(np.float64).ravel() > attack.gamma).astype(float)
    actual = attack._compute_scores(target, references, out, np.array([[1.0]]))
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="OUT reference model"):
        attack._compute_scores(target, references, np.zeros_like(out), np.array([[1.0]]))


def test_changed_population_uses_fresh_audit_directory(handler, mocker, tmp_path):
    from copy import copy
    from leakpro.signals.signal import ModelLogits

    settings = {"num_shadow_models": 4, "online": True}
    first = AttackFactoryMIA.create_attack("base", settings, handler)
    first_result, _ = first.run_robustness(n_runs=2, seed=7)
    pool = ShadowModelHandler()
    cache_key = f"target_{pool.target_model_hash}"
    old_logits = pool.load_logits(name=cache_key)
    other = copy(handler)
    other.population = TinyHandler.UserDataset(handler.population.data + 3, handler.population.targets)
    other.configs = handler.configs.model_copy(deep=True)
    other.configs.audit.output_dir = str(tmp_path / "revised-audit")
    train = mocker.spy(TinyHandler, "train")
    second = AttackFactoryMIA.create_attack("base", settings, other)
    second_result, _ = second.run_robustness(n_runs=2, seed=7)
    for audit, result in ((handler, first_result), (other, second_result)):
        experiment_id = result.metadata["robustness"]["experiment_id"]
        directory = Path(audit.configs.audit.output_dir) / "robustness" / experiment_id
        assert (directory / "summary.json").is_file()
    expected = ModelLogits()([second.target_model], other, second.audit_dataset["data"])[0]
    actual = ShadowModelHandler().load_logits(name=cache_key)
    np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(old_logits, actual)
    assert train.call_count == 4
    third = AttackFactoryMIA.create_attack("base", settings, other)
    third.run_robustness(n_runs=2, seed=7)
    assert train.call_count == 4
    with pytest.raises(ValueError, match="different audit"):
        first.run_robustness(n_runs=2, seed=7)


def test_shadow_sampling_change_builds_then_reuses_new_pool(handler, mocker):
    settings = {"num_shadow_models": 4, "online": True}
    first = AttackFactoryMIA.create_attack("base", settings, handler)
    first.run_robustness(n_runs=2, seed=7)
    train = mocker.spy(TinyHandler, "train")
    handler.configs.shadow_model = ShadowModelConfig(sampling_method="random")
    second = AttackFactoryMIA.create_attack("base", settings, handler)
    second.run_robustness(n_runs=2, seed=7)
    assert train.call_count == 4
    assert set(first.shadow_model_indices).isdisjoint(second.shadow_model_indices)
    second.run_robustness(n_runs=2, seed=7)
    assert train.call_count == 4


def test_random_pool_follows_the_handler_sampler(saved_audit_inputs):
    class FirstHalf(TinyInputHandler):
        def sample_shadow_indices(self, shadow_population, data_fraction):
            return np.sort(shadow_population)[:int(len(shadow_population) * data_fraction)]

    class LastHalf(TinyInputHandler):
        def sample_shadow_indices(self, shadow_population, data_fraction):
            return np.sort(shadow_population)[-int(len(shadow_population) * data_fraction):]

    config, path = saved_audit_inputs
    config["shadow_model"] = {"sampling_method": "random"}
    config["audit"]["attack_list"] = [
        {"attack": "base", "online": True, "num_shadow_models": 2, "robustness": {"n_runs": 2, "seed": 17}}
    ]
    path.write_text(yaml.safe_dump(config))

    def splits(handler_type):
        original, _ = LeakPro(handler_type, str(path)).run_audit()
        indices = original.metadata["robustness"]["references"]
        metadata = ShadowModelHandler().get_shadow_model_metadata(indices)
        return indices, [list(model.train_indices) for model in metadata]

    first, lower = splits(FirstHalf)
    assert lower == [list(range(30))] * 2
    replacement, upper = splits(LastHalf)
    assert upper == [list(range(30, 60))] * 2
    assert set(first).isdisjoint(replacement)
    assert splits(LastHalf) == (replacement, upper)


@pytest.mark.parametrize("custom_sampler", [False, True], ids=["built-in", "custom"])
def test_random_pool_matches_fraction_sensitive_sampler(saved_audit_inputs, tmp_path, custom_sampler):
    class FractionSampler(TinyInputHandler):
        def sample_shadow_indices(self, shadow_population, data_fraction):
            records = np.sort(shadow_population)
            count = int(len(records) * data_fraction)
            return records[:count] if data_fraction < 0.405 else records[-count:]

    handler_type = FractionSampler if custom_sampler else TinyInputHandler
    config, path = saved_audit_inputs
    config["shadow_model"] = {"sampling_method": "random"}
    entries = [
        {"attack": "base", "online": True, "num_shadow_models": 2, "training_data_fraction": fraction,
         "robustness": {"n_runs": 2, "seed": 17}}
        for fraction in (0.40, 0.41)
    ]
    config["audit"]["attack_list"] = entries
    path.write_text(yaml.safe_dump(config))
    first, _, changed, _ = LeakPro(handler_type, str(path)).run_audit()

    def splits(result):
        indices = result.metadata["robustness"]["references"]
        return indices, [list(model.train_indices) for model in ShadowModelHandler().get_shadow_model_metadata(indices)]

    first_ids, first_rows = splits(first)
    changed_ids, changed_rows = splits(changed)
    if custom_sampler:
        assert first_rows == [list(range(24))] * 2
        assert changed_rows == [list(range(36, 60))] * 2
        assert set(first_ids).isdisjoint(changed_ids)
    else:
        assert (first_ids, first_rows) == (changed_ids, changed_rows)

    config["audit"].update(output_dir=str(tmp_path / "fresh"), attack_list=entries[1:])
    path.write_text(yaml.safe_dump(config))
    fresh, _ = LeakPro(handler_type, str(path)).run_audit()
    assert splits(fresh)[1] == changed_rows
    np.testing.assert_array_equal(fresh.signal_values, changed.signal_values)


def test_predictions_follow_forward_settings_with_same_weights(saved_audit_inputs):
    config, yaml_path = saved_audit_inputs
    config["audit"]["attack_list"] = [{"attack": "base", "online": True, "num_shadow_models": 2,
                                      "robustness": {"n_runs": 2, "seed": 17}}]
    yaml_path.write_text(yaml.safe_dump(config))
    metadata_file = Path(config["target"]["target_folder"]) / "model_metadata.pkl"
    first = LeakPro(TinyInputHandler, str(yaml_path))
    first.run_audit()
    predictions = ShadowModelHandler().load_logits(name="target")
    metadata = first.handler.target_model_metadata.model_copy(update={"init_params": {"gain": 2.0}})
    metadata_file.write_bytes(pickle.dumps(metadata))
    warm = LeakPro(TinyInputHandler, str(yaml_path)).run_audit()
    np.testing.assert_array_equal(ShadowModelHandler().load_logits(name="target"), 2 * predictions)
    config["audit"]["output_dir"] += "-fresh"
    yaml_path.write_text(yaml.safe_dump(config))
    cold = LeakPro(TinyInputHandler, str(yaml_path)).run_audit()
    for reused, fresh in zip(warm, cold):
        np.testing.assert_array_equal(reused.signal_values, fresh.signal_values)


def test_reused_mixed_rmia_audit_reloads_changed_checkpoint(saved_audit_inputs):
    config, path = saved_audit_inputs
    normal = {"attack": "rmia", "num_shadow_models": 4, "online": True, "gamma": 1.0}
    config["audit"]["attack_list"] = [normal, {**normal, "robustness": {"n_runs": 2, "seed": 17}}]
    path.write_text(yaml.safe_dump(config))
    audit = LeakPro(TinyInputHandler, str(path))
    results = audit.run_audit()
    initial, original = results[0], original_rotation(results)
    np.testing.assert_array_equal(initial.signal_values, original.signal_values)

    pool = ShadowModelHandler()
    checkpoint = Path(pool.storage_path) / "shadow_model_0.pkl"
    weights = torch.load(checkpoint, weights_only=True)
    torch.save({name: torch.zeros_like(value) for name, value in weights.items()}, checkpoint)

    results = audit.run_audit()
    reused, rotated = results[0], original_rotation(results)
    assert np.any(initial.signal_values != rotated.signal_values)
    np.testing.assert_array_equal(reused.signal_values, rotated.signal_values)
    fresh, *_ = LeakPro(TinyInputHandler, str(path)).run_audit()
    np.testing.assert_array_equal(reused.signal_values, fresh.signal_values)


def test_replacing_one_shadow_checkpoint_refreshes_only_its_predictions(handler, mocker):
    settings = {"num_shadow_models": 2, "online": True}
    attack = AttackFactoryMIA.create_attack("base", settings, handler)
    before = attack.run_robustness(n_runs=3, seed=17)
    pool = ShadowModelHandler()
    checkpoint = Path(pool.storage_path) / "shadow_model_0.pkl"
    state = torch.load(checkpoint, weights_only=True)
    torch.save({name: torch.zeros_like(value) for name, value in state.items()}, checkpoint)
    infer = mocker.spy(pool, "cache_logits")
    after = attack.run_robustness(n_runs=3, seed=17)
    assert infer.call_count == 1
    assert np.all(pool.load_logits(indx=0) == 0)
    assert before[0].id != after[0].id
    assert np.any(before[0].signal_values != after[0].signal_values)
    assert attack.run_robustness(n_runs=3, seed=17)[0].id == after[0].id
    assert infer.call_count == 1


@pytest.mark.parametrize("model", ["target", "shadow"])
def test_experiment_distinguishes_checkpoints_with_identical_scores(handler, tmp_path, model):
    handler.population.data[:, 1] = 0
    settings = {"num_shadow_models": 2, "online": True}
    output = tmp_path / "reports"
    attack = AttackFactoryMIA.create_attack("base", settings, handler)
    audit_seed = handler.configs.audit.random_seed
    seed_everything(audit_seed)
    before = attack.run_robustness(n_runs=3, seed=17, output_dir=str(output))
    if model == "target":
        with torch.no_grad():
            handler.target_model.linear.weight[:, 1].add_(1)
        # Separate training output reuses the same model ordinals without sharing metadata.
        handler.configs.audit.output_dir = str(tmp_path / "second-pool")
        attack = AttackFactoryMIA.create_attack("base", settings, handler)
    else:
        pool = ShadowModelHandler()
        checkpoint = Path(pool.storage_path) / f"shadow_model_{attack.shadow_model_indices[0]}.pkl"
        weights = torch.load(checkpoint, weights_only=True)
        weights["linear.weight"][:, 1].add_(1)
        torch.save(weights, checkpoint)

    seed_everything(audit_seed)
    after = attack.run_robustness(n_runs=3, seed=17, output_dir=str(output))
    for original, updated in zip(before, after):
        np.testing.assert_array_equal(original.signal_values, updated.signal_values)
        assert original.id != updated.id
    assert len(list((output / "robustness").glob("*/summary.json"))) == 2


@pytest.mark.parametrize("target_params,compatible", [({}, True), ({"batch_size": None}, False)])
def test_loader_default_batch_size_through_yaml(saved_audit_inputs, target_params, compatible):
    config, path = saved_audit_inputs
    metadata_path = Path(config["target"]["target_folder"]) / "model_metadata.pkl"
    metadata = pickle.loads(metadata_path.read_bytes())
    metadata.data_loader.params = target_params
    metadata_path.write_bytes(pickle.dumps(metadata))
    config["shadow_model"] = {"batch_size": 1}
    config["audit"]["attack_list"] = [{"attack": "base", "num_shadow_models": 2, "online": True,
                                      "robustness": {"n_runs": 2}}]
    path.write_text(yaml.safe_dump(config))
    if compatible:
        assert len(LeakPro(TinyInputHandler, str(path)).run_audit()) == 2
    else:
        with pytest.raises(ValueError, match="same target and shadow training setup"):
            LeakPro(TinyInputHandler, str(path))


def original_rotation(results):
    return next(result for result in results if result.metadata.get("robustness", {}).get("target") == -1)


def expected_reference_rows(training, target):
    complement = set(range(60)) - set(training[target])
    return [row for row, split in enumerate(training) if row != target and set(split) != complement]


def test_default_draw_keeps_both_targets_but_excludes_their_partners():
    population = np.arange(10, 16)
    # Two pairs in nonadjacent slots. Seed 11 draws all shadows and omits the original.
    training_indices = [[10, 11], [10, 11, 12], [10, 12, 14], [13, 14, 15], [11, 13, 15]]
    plan = RobustnessConfig(seed=11).reference_models(population, training_indices)
    assert list(plan) == [1, 4, 2, 3]
    assert {target: refs.tolist() for target, refs in plan.items()} == {
        1: [0, 2, 4], 4: [0, 1, 3], 2: [0, 1, 3], 3: [0, 2, 4]
    }

    # Complementary-looking audit rows alone do not make two full splits a pair.
    overlapping = [[10, 11, 14], [12, 13, 14], [10, 15]]
    refs = RobustnessConfig(seed=17).reference_models(population, overlapping)
    np.testing.assert_array_equal(refs[0], [1, 2])


def test_robustness_plot_shows_all_runs_and_pointwise_percentiles(tmp_path, mocker):
    results = [MIAResult.from_full_scores([1, 0, 1, 0], scores, result_name="LiRA")
               for scores in ([4, 3, 2, 1], [3, 4, 1, 2])]
    summary = summarize_robustness(results)
    plot = mocker.spy(plt, "plot")
    with plt.rc_context({"figure.figsize": (2.5, 2.5)}):
        MIAResult.create_roc_plot(results, str(tmp_path), rotation_summary=summary)
    assert (tmp_path / "ROC.png").is_file()
    gray = [call for call in plot.call_args_list if call.kwargs.get("color") == "gray"]
    assert len(gray) == 2
    for call, result in zip(gray, results):
        assert 0 < call.kwargs["alpha"] < 1
        np.testing.assert_array_equal(call.args[0], result.fpr)
        np.testing.assert_array_equal(call.args[1], result.tpr)
    red = [call for call in plot.call_args_list if call.kwargs.get("color") == "#c62828"]
    assert [call.kwargs["label"] for call in red] == ["2.5th percentile", "97.5th percentile"]
    for call, values in zip(red, summary["roc"]["pointwise_tpr_percentiles_2_5_97_5"]):
        np.testing.assert_array_equal(call.args[0], summary["roc"]["fpr"])
        np.testing.assert_array_equal(call.args[1], values)


def test_roc_does_not_move_vertical_jump_into_preceding_segment():
    first = MIAResult.from_full_scores([0, 0, 1, 1, 0, 0], [6, 5, 4, 3, 2, 1], result_name="first")
    second = MIAResult.from_full_scores([0, 1, 0, 1], [4, 3, 2, 1], result_name="second")
    roc = summarize_robustness([first, second])["roc"]
    assert roc["mean_tpr"][roc["fpr"].index(0.25)] == 0
    at_jump = np.asarray(roc["fpr"]) == 0.5
    np.testing.assert_allclose(np.asarray(roc["mean_tpr"])[at_jump], [0, 0.75])
    assert trapezoid(roc["mean_tpr"], roc["fpr"]) == pytest.approx((first.roc_auc + second.roc_auc) / 2)
