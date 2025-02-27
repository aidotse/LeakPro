"""Test the shadow model handler module."""
import os

import numpy as np
from pytest import raises

from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.tests.constants import get_shadow_model_config
from leakpro.tests.input_handler.image_input_handler import ImageInputHandler
from leakpro.schemas import ShadowModelConfig

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

    assert sm.batch_size == image_handler.target_model_metadata.batch_size
    assert sm.epochs == image_handler.target_model_metadata.epochs
    assert sm.init_params == image_handler.target_model_metadata.init_params
    assert sm.model_blueprint == image_handler.target_model.__class__

    optimizer_config = image_handler.target_model_metadata.optimizer.model_copy().model_dump(exclude={"name"})
    assert sm.optimizer_config == optimizer_config
    loss_config = image_handler.target_model_metadata.loss.model_copy().model_dump(exclude={"name"})
    assert sm.loss_config == loss_config

def test_shadow_model_creation_and_loading(image_handler:ImageInputHandler) -> None:
    shadow_config = ShadowModelConfig(**get_shadow_model_config())
    
    image_handler.configs.shadow_model = shadow_config


    # Test initialization
    if ShadowModelHandler.is_created() == True:
        ShadowModelHandler.delete_instance()
    sm = ShadowModelHandler(image_handler)

    assert sm.batch_size == image_handler.configs.shadow_model.batch_size
    assert sm.epochs == image_handler.configs.shadow_model.epochs
    assert sm.init_params == {}
    assert sm.model_blueprint is not None
    assert sm.optimizer_config is not None
    assert sm.loss_config is not None

    # Test creation
    n_models = 1
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


    indx2 = sm.create_shadow_models(n_models, image_handler.test_indices, training_fraction, not online)[0]
    entries = os.listdir(sm.storage_path)
    assert len(entries) - n_entries_phase1 == 2*n_models
    assert f"metadata_{indx}.pkl" in entries
    assert f"shadow_model_{indx}.pkl" in entries
    assert f"metadata_{indx2}.pkl" in entries
    assert f"shadow_model_{indx2}.pkl" in entries

    # Test loading
    meta_0 = sm._load_metadata(sm.storage_path + f"/metadata_{indx}.pkl")
    meta_1 = sm._load_metadata(sm.storage_path + f"/metadata_{indx2}.pkl")
    assert meta_0.online == online
    assert meta_1.online == (not online)
    assert meta_1.num_train == meta_0.num_train

    shadow_model_indices = [indx, indx2]
    models, indices = sm.get_shadow_models(shadow_model_indices)
    for model in models:
        assert model.model_obj.__class__.__name__ == "ConvNet"

    # Test index mask
    # check what test data is included in the training data of the shadow models
    mask = sm.get_in_indices_mask(shadow_model_indices, np.array(image_handler.test_indices))

    true_mask_0 = np.array([True if item in meta_0.train_indices else False for item in image_handler.test_indices])
    assert np.array_equal(mask[:, 0], true_mask_0)

    true_mask_1 = np.array([True if item in meta_1.train_indices else False for item in image_handler.test_indices])
    assert np.array_equal(mask[:, 1], true_mask_1)
