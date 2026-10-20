#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Integration checks for target rotation and fixed reference-pool reuse."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from leakpro import LeakPro
from leakpro.attacks.attack_scheduler import AttackScheduler
from leakpro.attacks.mia_attacks.attack_factory_mia import AttackFactoryMIA
from leakpro.attacks.robustness import Robustness
from leakpro.attacks.utils.distillation_model_handler import DistillationModelHandler
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.schemas import EvalOutput, LeakProConfig, LossConfig, MIAMetaDataSchema, ShadowModelConfig, TrainingOutput
from leakpro.utils.save_load import hash_model


class TinyModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.linear = nn.Linear(2, num_classes)

    def forward(self, inputs):
        return self.linear(inputs)


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


@pytest.fixture
def handler(tmp_path, monkeypatch):
    monkeypatch.setenv("LEAKPRO_DEVICE", "cpu")
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
    DistillationModelHandler.delete_instance()
    AttackFactoryMIA.shadow_model_handler = None
    AttackFactoryMIA.distillation_model_handler = None


@pytest.mark.parametrize("name,extra", [
    ("lira", {"online": True}), ("rmia", {"online": True}), ("base", {"online": True}), ("population", {}),
])
def test_real_attacks_rotate_target_and_reuse_pool(handler, monkeypatch, mocker, name, extra):
    if name == "population":
        handler.target_model = TinyModel(num_classes=1)
        handler.population.targets = handler.population.targets.float().unsqueeze(1)
        handler._criterion = nn.BCEWithLogitsLoss()
        handler.target_model_metadata.init_params = {"num_classes": 1}
        handler.target_model_metadata.criterion = LossConfig(name="bcewithlogitsloss")
        handler.train(handler.get_dataloader(handler.train_indices), handler.target_model, handler.get_criterion(),
                      handler.get_optimizer(handler.target_model), 1)
    config = extra if name == "population" else {"num_shadow_models": 4, "training_data_fraction": 1 / 3, **extra}
    captured = []
    create = AttackFactoryMIA.create_attack

    def capture(cls, *args, **kwargs):
        attack = create(*args, **kwargs)
        captured.append(attack)
        return attack

    monkeypatch.setattr(AttackFactoryMIA, "create_attack", classmethod(capture))
    training = mocker.spy(TinyHandler, "train")
    original_model = handler.target_model
    original_hash = hash_model(original_model)
    original_config = handler.configs.model_dump()
    wrapper = Robustness(handler, name, config, n_runs=3, num_shadow_models=4, seed=123)
    runs = wrapper.run()

    assert training.call_count == 4
    assert [entry["metadata"]["target"] for entry in runs] == wrapper.target_indices
    assert wrapper.target_indices[0] == -1
    assert len(set(wrapper.target_indices)) == 3
    assert len({hash_model(attack.handler.target_model) for attack in captured}) == 3
    for attack, entry in zip(captured, runs):
        metadata = entry["metadata"]
        assert metadata["target"] not in metadata["references"]
        assert len(metadata["references"]) == 4
        assert set(metadata["references"]) | {metadata["target"]} == {-1, *wrapper._shadow_indices}
        if metadata["target"] != -1:
            assert -1 in metadata["references"]
        expected_train = wrapper._metadata[metadata["target"]].train_indices
        assert set(attack.handler.train_indices) == set(expected_train)
        assert attack.handler.target_model_metadata.train_result == wrapper._metadata[metadata["target"]].train_result
        assert attack.handler.target_model_metadata.test_result == wrapper._metadata[metadata["target"]].test_result
        assert hash_model(attack.handler.target_model) == metadata["target_hash"]
        assert entry["result_object"].true.sum() == metadata["member_count"]
        if name == "lira":
            labels = attack.handler.get_labels(attack.audit_dataset["data"])
            expected = [attack.signal(wrapper._load_logits(index, attack.audit_dataset["data"]), labels)
                        for index in metadata["references"]]
            membership = np.array([np.isin(attack.audit_dataset["data"], wrapper._metadata[index].train_indices)
                                   for index in metadata["references"]])
            eligible = membership.any(axis=0) & (~membership).any(axis=0)
            np.testing.assert_allclose(attack.shadow_models_signals, np.asarray(expected)[:, eligible])
            np.testing.assert_array_equal(entry["result_object"].audit_indices, attack.audit_dataset["data"][eligible])
        assert list((Path(attack.handler.configs.audit.output_dir) / "data_objects").glob("*.json"))
    assert handler.target_model is original_model
    assert hash_model(original_model) == original_hash
    assert handler.configs.model_dump() == original_config
    np.testing.assert_array_equal(handler.train_indices, np.arange(20))
    assert not ShadowModelHandler.is_created()
    assert not DistillationModelHandler.is_created()
    saved = json.loads((wrapper.output_dir / "summary.json").read_text())
    assert saved["rotated_target_summary"]["n_targets"] == 2
    assert saved["runs"][0]["target_hash"] == original_hash


def test_cached_pool_and_seed_reproduce_runs(handler, mocker):
    config = {"num_shadow_models": 4, "training_data_fraction": 1 / 3, "online": True}
    first = Robustness(handler, "lira", config, 3, 4, seed=7)
    first.run()
    training = mocker.spy(TinyHandler, "train")
    second = Robustness(handler, "lira", config, 3, 4, seed=7)
    second.run()
    assert training.call_count == 0
    assert first.target_indices == second.target_indices
    assert first.output_dir == second.output_dir
    for old, new in zip(first.results, second.results):
        np.testing.assert_allclose(old["result_object"].signal_values, new["result_object"].signal_values)
    assert first.summary == second.summary


@pytest.mark.parametrize("fraction,compatible", [(15 / 22, True), (14.9 / 22, False)])
def test_target_training_count_survives_fraction_rounding(handler, mocker, fraction, compatible):
    handler.population.data = handler.population.data[:22]
    handler.population.targets = handler.population.targets[:22]
    handler.population_size = 22
    handler.train_indices = np.arange(15)
    handler.test_indices = np.arange(15, 22)
    handler.target_model_metadata.train_indices = handler.train_indices.tolist()
    handler.target_model_metadata.test_indices = handler.test_indices.tolist()
    handler.target_model_metadata.num_train = 15
    handler.target_model = TinyModel()
    handler.train(handler.get_dataloader(handler.train_indices), handler.target_model, handler.get_criterion(),
                  handler.get_optimizer(handler.target_model), 1)
    training = mocker.spy(TinyHandler, "train")
    wrapper = Robustness(handler, "lira", {"num_shadow_models": 4, "training_data_fraction": fraction,
                                          "online": True}, 3, 4, seed=19)
    if compatible:
        assert len(wrapper.run()) == 3
    else:
        with pytest.raises(ValueError, match="incompatible with the robustness pool"):
            wrapper.run()
    assert training.call_count == 4
    assert all(len(call.args[1].dataset) == 15 for call in training.call_args_list)
    assert all(len(metadata.train_indices) == 15 for metadata in wrapper._metadata.values())


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


def test_documented_workflow_from_saved_leakpro_inputs(saved_audit_inputs):
    config, config_path = saved_audit_inputs
    config["audit"]["attack_list"] = [{"attack": "lira", "num_shadow_models": 4,
                                       "training_data_fraction": 1 / 3, "online": True}]
    config_path.write_text(yaml.safe_dump(config))
    audit = LeakPro(TinyInputHandler, str(config_path))
    existing = ShadowModelHandler()
    old_state = dict(vars(existing))
    runner = Robustness(audit.handler, "lira", {key: value for key, value in config["audit"]["attack_list"][0].items()
                                              if key != "attack"}, 3, 4, seed=11)
    runs = runner.run()
    assert len(runs) == 3
    assert runner._pool_handler.handler.train.__self__ is runner._pool_handler.handler
    assert runner._copy_handler().train.__self__ is not audit.handler
    assert vars(ShadowModelHandler()) == old_state
    assert json.loads((runner.output_dir / "summary.json").read_text())["setup"]["targets"][0] == -1


@pytest.mark.parametrize("name", ["lira", "rmia"])
def test_yaml_robustness_runs_through_normal_audit(saved_audit_inputs, mocker, name):
    config, config_path = saved_audit_inputs
    attack_config = {"num_shadow_models": 4, "training_data_fraction": 1 / 3, "online": True}
    config["audit"]["random_seed"] = 11
    config["audit"]["attack_list"] = [
        {"attack": "unknown"},
        {"attack": "lira", **attack_config, "robustness": None},
        {"attack": name, **attack_config, "robustness": {"n_runs": 3}},
        {"attack": "base", **attack_config},
    ]
    config_path.write_text(yaml.safe_dump(config))
    training = mocker.spy(TinyInputHandler, "train")
    report = mocker.patch("leakpro.leakpro.ReportHandler")
    audit = LeakPro(TinyInputHandler, str(config_path))
    results = audit.run_audit(create_pdf=True)
    assert len(results) == 5
    assert len({result.id for result in results}) == 5
    assert all(result.result.id == result.id for result in results)
    assert audit.attack_scheduler.attack_names == ["lira", name, "base"]
    runner = audit.attack_scheduler.attacks[1]
    assert isinstance(runner, Robustness)
    assert runner.seed == 11
    assert runner.num_shadow_models == 4
    assert all(actual is entry["result_object"] for actual, entry in zip(results[1:4], runner.results))
    assert audit.handler.configs.audit.attack_list == config["audit"]["attack_list"]
    assert training.call_count == 8  # Four ordinary shadows and one reusable four-model robustness pool.
    report.assert_called_once_with(results=results, report_dir=audit.report_dir)
    report.return_value.create_report.assert_called_once_with()
    saved = json.loads((runner.output_dir / "summary.json").read_text())
    assert saved["setup"]["targets"] == [-1, *np.random.default_rng(11).permutation(4)[:2].tolist()]
    assert len(saved["runs"]) == 3
    assert all(entry["target"] not in entry["references"] for entry in saved["runs"])
    assert all(-1 in entry["references"] for entry in saved["runs"][1:])
    assert len(list((Path(config["audit"]["output_dir"]) / "data_objects").glob("*.json"))) == 2
    report_dir = Path(config["audit"]["output_dir"]) / "report-check"
    MIAResult.create_results(results, save_dir=str(report_dir))
    assert all((report_dir / result.id).is_dir() for result in results)


@pytest.mark.parametrize("options", [
    {}, False, {"n_runs": True}, {"n_runs": 1}, {"n_runs": 6},
    {"n_runs": 2, "seed": -1}, {"n_runs": 2, "typo": 4}, {"n_runs": 2, "num_shadow_models": 3},
])
def test_invalid_yaml_robustness_rejected_before_training(handler, mocker, options):
    handler.configs.audit.attack_list = [{"attack": "lira", "num_shadow_models": 4,
                                         "robustness": options}]
    training = mocker.spy(TinyHandler, "train")
    with pytest.raises(ValueError):
        AttackScheduler(handler, output_dir=handler.configs.audit.output_dir)
    assert training.call_count == 0


@pytest.mark.parametrize("seed", [7, None])
def test_yaml_robustness_seed_override_and_output_directory(handler, tmp_path, seed):
    handler.configs.audit.attack_list = [{"attack": "lira", "num_shadow_models": 4,
                                         "training_data_fraction": 1 / 3,
                                         "robustness": {"n_runs": 2, "seed": seed}}]
    output_dir = tmp_path / "custom"
    scheduler = AttackScheduler(handler, output_dir=str(output_dir))
    records = scheduler.run_attacks()
    runner = scheduler.attacks[0]
    assert runner.seed is seed
    assert len(records) == 2
    assert all(entry["attack_type"] == "lira" and entry["attack_object"] is runner for entry in records)
    assert runner.output_dir.is_relative_to(output_dir)
    assert (runner.output_dir / "summary.json").is_file()
    assert not (Path(handler.configs.audit.output_dir) / "robustness").exists()


def test_robustness_rejects_non_mia_and_optuna_before_training(handler, mocker):
    handler.configs.audit.attack_list = [{"attack": "lira", "num_shadow_models": 4,
                                         "robustness": {"n_runs": 2}}]
    handler.configs.audit.attack_type = "minv"
    initialize = mocker.spy(AttackScheduler, "_initialize_factory")
    with pytest.raises(ValueError, match="only supported for MIA"):
        AttackScheduler(handler, output_dir=handler.configs.audit.output_dir)
    assert initialize.call_count == 0
    handler.configs.audit.attack_type = "mia"
    training = mocker.spy(TinyHandler, "train")
    scheduler = AttackScheduler(handler, output_dir=handler.configs.audit.output_dir)
    with pytest.raises(ValueError, match="use_optuna must be False"):
        scheduler.run_attacks(use_optuna=True)
    assert training.call_count == 0


@pytest.mark.parametrize("fail_second", [False, True])
def test_gradient_attack_preserves_caller_model_state(handler, monkeypatch, fail_second):
    model = handler.target_model
    model.train()
    original_hash = hash_model(model)
    hooks = [dict(module._forward_hooks) for module in model.modules()]
    training = [module.training for module in model.modules()]
    gradients = [parameter.grad.clone() for parameter in model.parameters()]
    attack_class = AttackFactoryMIA.attack_classes["yoqo"]
    run_attack = attack_class.run_attack
    calls = []

    def run_then_fail(attack):
        result = run_attack(attack)
        calls.append(result)
        if fail_second and len(calls) == 2:
            raise RuntimeError("injected gradient attack failure")
        return result

    monkeypatch.setattr(attack_class, "run_attack", run_then_fail)
    runner = Robustness(handler, "yoqo", {"num_shadow_models": 4, "training_data_fraction": 1 / 3,
                                          "online": True, "max_iterations": 1}, 2, 4, seed=123)
    if fail_second:
        with pytest.raises(RuntimeError, match="injected gradient attack failure"):
            runner.run()
    else:
        assert len(runner.run()) == 2
    assert len(calls) == 2
    assert handler.target_model is model
    assert hash_model(model) == original_hash
    assert [dict(module._forward_hooks) for module in model.modules()] == hooks
    assert [module.training for module in model.modules()] == training
    for actual, expected in zip(model.parameters(), gradients):
        torch.testing.assert_close(actual.grad, expected, rtol=0, atol=0)


@pytest.mark.parametrize("runs,pool,seed", [(1, 4, 1), (6, 4, 1), (True, 4, 1), (2, 0, 1), (2, 4, -1)])
def test_invalid_schedule_rejected_before_training(handler, mocker, runs, pool, seed):
    training = mocker.spy(TinyHandler, "train")
    with pytest.raises(ValueError):
        Robustness(handler, "population", {}, runs, pool, seed=seed)
    assert training.call_count == 0


def test_training_setup_mismatch_restores_existing_handler(handler, mocker):
    existing = ShadowModelHandler(handler)
    original_state = dict(vars(existing))
    handler.configs.shadow_model = ShadowModelConfig(epochs=2)
    training = mocker.spy(TinyHandler, "train")
    with pytest.raises(ValueError, match="same target and shadow training setup"):
        Robustness(handler, "population", {}, 2, 2, seed=3).run()
    assert training.call_count == 0
    assert vars(ShadowModelHandler()) == original_state


def test_incompatible_attack_population_does_not_train_replacements(handler, mocker):
    training = mocker.spy(TinyHandler, "train")
    config = {"num_shadow_models": 4, "training_data_fraction": 0.5, "online": True}
    with pytest.raises(ValueError, match="incompatible with the robustness pool"):
        Robustness(handler, "lira", config, 3, 4, seed=123).run()
    assert training.call_count == 4
    assert not ShadowModelHandler.is_created()


def test_attack_failure_restores_shared_state(handler, monkeypatch):
    existing = ShadowModelHandler(handler)
    old_state = dict(vars(existing))
    attack_class = AttackFactoryMIA.attack_classes["lira"]
    original_run = attack_class.run_attack
    calls = []

    def fail_second(attack):
        calls.append(attack.handler.target_model)
        if len(calls) == 2:
            raise RuntimeError("injected attack failure")
        return original_run(attack)

    monkeypatch.setattr(attack_class, "run_attack", fail_second)
    with pytest.raises(RuntimeError, match="injected attack failure"):
        Robustness(handler, "lira", {"num_shadow_models": 4, "training_data_fraction": 1 / 3}, 3, 4, seed=6).run()
    assert vars(ShadowModelHandler()) == old_state
    assert handler.target_model is not calls[0]


def test_summary_uses_upper_roc_ties_and_excludes_original():
    original = MIAResult.from_full_scores([0, 0, 1, 1], [4, 3, 2, 1], result_name="test")
    rotated = MIAResult.from_full_scores([1, 1, 0, 1, 0], [5, 4, 3, 2, 1], result_name="test")
    wrapper = object.__new__(Robustness)
    wrapper.results = [{"result_object": result} for result in [original, rotated, rotated]]
    summary = wrapper._summarize()
    assert summary["roc_auc"]["mean"] == rotated.roc_auc
    assert summary["roc"]["fpr"][0] == 0
    assert summary["roc"]["mean_tpr"][0] == pytest.approx(2 / 3)
    np.testing.assert_allclose(summary["roc"]["pointwise_tpr_percentiles_2_5_97_5"],
                               [summary["roc"]["mean_tpr"]] * 2)


@pytest.mark.parametrize("all_tied", [False, True])
def test_summary_includes_roc_origin_for_tied_scores(all_tied):
    tied = MIAResult.from_full_scores([1, 0, 1, 0], [2, 2, 1, 1], result_name="tied")
    other = tied if all_tied else MIAResult.from_full_scores([1, 0, 1, 0], [4, 3, 2, 1], result_name="untied")
    wrapper = object.__new__(Robustness)
    wrapper.results = [{"result_object": result} for result in [tied, tied, other]]
    summary = wrapper._summarize()["roc"]
    assert summary["fpr"][0] == 0.0
    assert summary["mean_tpr"][0] == pytest.approx(0.0 if all_tied else 0.25)
    expected = [0.0, 0.0] if all_tied else [0.0125, 0.4875]
    np.testing.assert_allclose(np.array(summary["pointwise_tpr_percentiles_2_5_97_5"])[:, 0], expected)
