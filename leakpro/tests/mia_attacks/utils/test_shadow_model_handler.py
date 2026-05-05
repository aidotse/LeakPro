#
# Copyright 2023-2026 AI Sweden
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Test the shadow model handler module."""
import os
import pickle

import numpy as np
from pytest import raises

from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.schemas import EvalOutput, LossConfig, OptimizerConfig, ShadowModelConfig, ShadowModelTrainingSchema, TrainingOutput
from leakpro.tests.constants import get_shadow_model_config
from leakpro.tests.input_handler.image_input_handler import ImageInputHandler

def test_shadow_model_handler_singleton(image_handler:ImageInputHandler) -> None:
    """Test that only one instance gets created."""

    image_handler.configs.shadow_model = get_shadow_model_config()
    if ShadowModelHandler.is_created() == False:
        sm = ShadowModelHandler(image_handler)
        assert ShadowModelHandler.is_created() == True

    image_handler.configs.shadow_model.model_class = "ConvNet_Dummy"
    module_path = image_handler.configs.shadow_model.module_path
    error_str = f"Failed to create model blueprint from ConvNet_Dummy in {module_path}"
    # expect value error as there is no class ConvNetDummy
    with raises(ValueError) as excinfo:
        ShadowModelHandler(image_handler)
    assert str(excinfo.value) == error_str

def test_shadow_model_handler_creation_from_target(image_handler:ImageInputHandler) -> None:
    image_handler.configs.shadow_model = None

    # Test initialization
    if ShadowModelHandler.is_created() == True:
        ShadowModelHandler.delete_instance()
    sm = ShadowModelHandler(image_handler)

    assert sm.epochs == image_handler.target_model_metadata.epochs
    assert sm.init_params == image_handler.target_model_metadata.init_params
    assert sm.model_blueprint == image_handler.target_model.__class__

    optimizer_config = image_handler.target_model_metadata.optimizer.params
    assert sm.optimizer_config == optimizer_config
    loss_config = image_handler.target_model_metadata.criterion.params
    assert sm.loss_config == loss_config

def test_shadow_model_creation_and_loading(image_handler:ImageInputHandler) -> None:
    shadow_config = ShadowModelConfig(**get_shadow_model_config())
    
    image_handler.configs.shadow_model = shadow_config


    # Test initialization
    if ShadowModelHandler.is_created() == True:
        ShadowModelHandler.delete_instance()
    sm = ShadowModelHandler(image_handler)

    assert sm.epochs == image_handler.configs.shadow_model.epochs
    assert sm.init_params == {}
    assert sm.model_blueprint is not None
    assert sm.optimizer_config is not None
    assert sm.loss_config is not None

    # Test creation
    n_models = 2
    training_fraction = 0.5
    online = False

    entries_start = os.listdir(sm.storage_path)
    n_entries_start = len(entries_start)

    indx = sm.create_shadow_models(n_models, image_handler.test_indices, training_fraction, online)[0]
    entries = os.listdir(sm.storage_path)
    n_entries_phase1 = len(entries)
    assert n_entries_phase1 - n_entries_start == 2*n_models
    assert f"metadata_{indx}.pkl" in entries
    assert f"shadow_model_{indx}.pkl" in entries

    # Test loading
    meta= sm._load_metadata(sm.storage_path + f"/metadata_{indx}.pkl")
    assert meta.num_train == training_fraction * len(image_handler.test_indices)
    assert meta.batch_size == image_handler.configs.shadow_model.batch_size
    assert meta.optimizer_params == image_handler.configs.shadow_model.optimizer.params
    assert meta.criterion_params == image_handler.configs.shadow_model.criterion.params
    shadow_model_indices = [indx]
    models, indices = sm.get_shadow_models(shadow_model_indices)
    for model in models:
        assert model.model_obj.__class__.__name__ == "ConvNet"

    # Test index mask
    # check what test data is included in the training data of the shadow models
    mask = sm.get_in_indices_mask(shadow_model_indices, np.array(image_handler.test_indices))

    true_mask_0 = np.array([True if item in meta.train_indices else False for item in image_handler.test_indices])
    assert np.array_equal(mask[:, 0], true_mask_0)


def test_shadow_model_partial_inheritance_from_target(image_handler: ImageInputHandler) -> None:
    """Shadow model should inherit missing fields from target setup."""
    image_handler.configs.shadow_model = ShadowModelConfig(init_params={"dpsgd": False})

    if ShadowModelHandler.is_created() is True:
        ShadowModelHandler.delete_instance()
    sm = ShadowModelHandler(image_handler)

    expected_init_params = image_handler.target_model_metadata.init_params.copy()
    expected_init_params.update({"dpsgd": False})

    assert sm.model_class == image_handler.configs.target.model_class
    assert sm.init_params == expected_init_params
    assert sm.optimizer_config == image_handler.target_model_metadata.optimizer.params
    assert sm.loss_config == image_handler.target_model_metadata.criterion.params
    assert sm.epochs == image_handler.target_model_metadata.epochs


def test_shadow_model_init_params_distinguish_missing_partial_and_empty(image_handler: ImageInputHandler) -> None:
    """Missing, partial, and empty init_params should have distinct inheritance behavior."""
    image_handler.target_model_metadata.init_params = {"width": 16, "dpsgd": True}

    image_handler.configs.shadow_model = ShadowModelConfig()
    if ShadowModelHandler.is_created() is True:
        ShadowModelHandler.delete_instance()
    sm = ShadowModelHandler(image_handler)
    assert sm.init_params == {"width": 16, "dpsgd": True}

    image_handler.configs.shadow_model = ShadowModelConfig(init_params={"dpsgd": False})
    if ShadowModelHandler.is_created() is True:
        ShadowModelHandler.delete_instance()
    sm = ShadowModelHandler(image_handler)
    assert sm.init_params == {"width": 16, "dpsgd": False}

    image_handler.configs.shadow_model = ShadowModelConfig(init_params={})
    if ShadowModelHandler.is_created() is True:
        ShadowModelHandler.delete_instance()
    sm = ShadowModelHandler(image_handler)
    assert sm.init_params == {}


def test_shadow_model_partial_override_optimizer_only(image_handler: ImageInputHandler) -> None:
    """Provided optimizer should override while other fields still inherit."""
    image_handler.configs.shadow_model = ShadowModelConfig(
        optimizer=OptimizerConfig(name="adam", params={"lr": 0.002})
    )

    if ShadowModelHandler.is_created() is True:
        ShadowModelHandler.delete_instance()
    sm = ShadowModelHandler(image_handler)

    assert sm.model_class == image_handler.configs.target.model_class
    assert sm.optimizer_class.__name__.lower() == "adam"
    assert sm.optimizer_config["lr"] == 0.002
    assert sm.loss_config == image_handler.target_model_metadata.criterion.params
    assert sm.epochs == image_handler.target_model_metadata.epochs


def test_shadow_model_model_class_without_module_path_uses_target_module(image_handler: ImageInputHandler) -> None:
    """model_class without module_path should resolve against the target module path."""
    image_handler.configs.shadow_model = ShadowModelConfig(model_class="ConvNet_Dummy")

    if ShadowModelHandler.is_created() is True:
        ShadowModelHandler.delete_instance()

    expected_path = image_handler.configs.target.module_path
    with raises(ValueError) as excinfo:
        ShadowModelHandler(image_handler)

    assert str(excinfo.value) == f"Failed to create model blueprint from ConvNet_Dummy in {expected_path}"


def test_shadow_model_invalid_optimizer_error_uses_optimizer_name(image_handler: ImageInputHandler) -> None:
    """Invalid optimizer errors should not depend on optimizer params containing a name key."""
    image_handler.configs.shadow_model = ShadowModelConfig(optimizer=OptimizerConfig(name="not_an_optimizer"))

    if ShadowModelHandler.is_created() is True:
        ShadowModelHandler.delete_instance()

    with raises(ValueError) as excinfo:
        ShadowModelHandler(image_handler)

    assert str(excinfo.value) == "Failed to create optimizer from not_an_optimizer"


def test_shadow_model_invalid_criterion_error_uses_criterion_name(image_handler: ImageInputHandler) -> None:
    """Invalid criterion errors should not reference a nonexistent criterion_config attribute."""
    image_handler.configs.shadow_model = ShadowModelConfig(criterion=LossConfig(name="not_a_loss"))

    if ShadowModelHandler.is_created() is True:
        ShadowModelHandler.delete_instance()

    with raises(ValueError) as excinfo:
        ShadowModelHandler(image_handler)

    assert str(excinfo.value) == "Failed to create criterion from not_a_loss"


def test_shadow_model_filter_requires_full_config_match(image_handler: ImageInputHandler, tmp_path) -> None:
    """Reuse should require the full effective training configuration, not just class and size."""
    image_handler.configs.shadow_model = ShadowModelConfig(
        init_params={"dpsgd": False},
        optimizer=OptimizerConfig(name="adam", params={"lr": 0.002}),
        batch_size=16,
    )

    if ShadowModelHandler.is_created() is True:
        ShadowModelHandler.delete_instance()
    sm = ShadowModelHandler(image_handler)
    sm.storage_path = str(tmp_path)

    metadata_kwargs = {
        "init_params": sm.init_params,
        "train_indices": [0, 1, 2, 3, 4],
        "num_train": 5,
        "optimizer": sm.optimizer_name,
        "optimizer_params": sm.optimizer_config,
        "criterion": sm.criterion_name,
        "criterion_params": sm.loss_config,
        "epochs": sm.epochs,
        "batch_size": sm.batch_size,
        "train_result": EvalOutput(accuracy=0.0, loss=0.0),
        "test_result": EvalOutput(accuracy=0.0, loss=0.0),
        "online": True,
        "model_class": sm.model_class,
        "model_module_path": sm.model_path,
        "target_model_hash": sm.target_model_hash,
    }
    matching_metadata = ShadowModelTrainingSchema(**metadata_kwargs)
    stale_metadata = ShadowModelTrainingSchema(
        **{
            **metadata_kwargs,
            "optimizer_params": {"lr": 0.5},
        }
    )

    with open(tmp_path / "metadata_0.pkl", "wb") as file:
        pickle.dump(matching_metadata, file)
    with open(tmp_path / "metadata_1.pkl", "wb") as file:
        pickle.dump(stale_metadata, file)

    all_indices, filtered_indices = sm._filter(data_size=5, online=True)

    assert sorted(all_indices) == [0, 1]
    assert filtered_indices == [0]


def test_shadow_model_creation_uses_shadow_batch_size(
    image_handler: ImageInputHandler,
    monkeypatch,
    tmp_path,
) -> None:
    """Shadow training dataloader should honor the configured shadow-model batch size."""
    shadow_config = ShadowModelConfig(**get_shadow_model_config())
    shadow_config.batch_size = 7
    image_handler.configs.shadow_model = shadow_config

    if ShadowModelHandler.is_created() is True:
        ShadowModelHandler.delete_instance()
    sm = ShadowModelHandler(image_handler)
    sm.storage_path = str(tmp_path / "attack_objects")
    os.makedirs(sm.storage_path, exist_ok=True)
    sm.attack_cache_folder_path = str(tmp_path / "attack_cache")
    os.makedirs(sm.attack_cache_folder_path, exist_ok=True)

    observed = {}

    def fake_train(data_loader, model, criterion, optimizer, epochs):
        observed["batch_size"] = data_loader.batch_size
        return TrainingOutput(model=model, metrics=EvalOutput(accuracy=0.0, loss=0.0))

    monkeypatch.setattr(image_handler, "train", fake_train)
    monkeypatch.setattr(image_handler, "eval", lambda *args, **kwargs: EvalOutput(accuracy=0.0, loss=0.0))
    monkeypatch.setattr(sm, "cache_logits", lambda *args, **kwargs: None)

    sm.create_shadow_models(num_models=1, shadow_population=image_handler.test_indices, training_fraction=0.5, online=False)

    assert observed["batch_size"] == shadow_config.batch_size
