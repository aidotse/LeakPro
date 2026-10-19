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


def test_documented_workflow_from_saved_leakpro_inputs(handler, tmp_path):
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
    config["audit"]["attack_list"] = [{"attack": "lira", "num_shadow_models": 4,
                                       "training_data_fraction": 1 / 3, "online": True}]
    config_path = tmp_path / "audit.yaml"
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
    assert handler.target_model is calls[0]


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
