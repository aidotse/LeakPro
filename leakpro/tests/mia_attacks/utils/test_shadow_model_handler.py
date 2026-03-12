"""Test the shadow model handler module."""
import os

import numpy as np
from pytest import raises

from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.tests.constants import get_shadow_model_config
from leakpro.tests.input_handler.image_input_handler import ImageInputHandler
from leakpro.schemas import OptimizerConfig, ShadowModelConfig

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
