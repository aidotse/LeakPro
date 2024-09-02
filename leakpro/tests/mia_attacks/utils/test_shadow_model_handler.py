"""Test the shadow model handler module."""
import os
import numpy as np

from pytest import raises
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.user_inputs.cifar10_input_handler import Cifar10InputHandler
from leakpro.tests.constants import shadow_model_config

def test_shadow_model_handler_singleton(image_handler:Cifar10InputHandler) -> None:
    """Test that only one instance gets created."""
    
    image_handler.configs.shadow_model = shadow_model_config
    if ShadowModelHandler.is_created() == False:
        sm = ShadowModelHandler(image_handler)
        assert ShadowModelHandler.is_created() == True
    
    with raises(ValueError) as excinfo:
        ShadowModelHandler(image_handler)
    assert str(excinfo.value) == "Singleton already created with specific parameters."

def test_shadow_model_creation_and_loading(image_handler:Cifar10InputHandler) -> None:
    image_handler.configs.shadow_model = shadow_model_config
    
    # Test initialization
    if ShadowModelHandler.is_created() == False:
        sm = ShadowModelHandler(image_handler)
    else:
        sm = ShadowModelHandler()
    
    assert sm.batch_size == shadow_model_config.batch_size
    assert sm.epochs == shadow_model_config.epochs
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
    

    indx2 = sm.create_shadow_models(n_models, image_handler.test_indices, training_fraction, ~online)[0]
    entries = os.listdir(sm.storage_path)
    assert len(entries) - n_entries_phase1 == 2*n_models
    assert f"metadata_{indx}.pkl" in entries
    assert f"shadow_model_{indx}.pkl" in entries
    assert f"metadata_{indx2}.pkl" in entries
    assert f"shadow_model_{indx2}.pkl" in entries
    
    # Test loading
    meta_0 = sm._load_metadata(sm.storage_path + f"/metadata_{indx}.pkl")
    meta_1 = sm._load_metadata(sm.storage_path + f"/metadata_{indx2}.pkl")
    assert meta_0["online"] == online
    assert meta_1["online"] == ~online
    assert meta_1["num_train"] == meta_0["num_train"]
    
    shadow_model_indices = [indx, indx2]
    models, indices = sm.get_shadow_models(shadow_model_indices)
    for model in models:
        assert model.model_obj.__class__.__name__ == "ConvNet"
    
    # Test index mask
    # check what test data is included in the training data of the shadow models
    mask = sm.get_in_indices_mask(shadow_model_indices, np.array(image_handler.test_indices))
    
    true_mask_0 = np.array([True if item in meta_0["train_indices"] else False for item in image_handler.test_indices])
    assert np.array_equal(mask[:, 0], true_mask_0)
    
    true_mask_1 = np.array([True if item in meta_1["train_indices"] else False for item in image_handler.test_indices])
    assert np.array_equal(mask[:, 1], true_mask_1)
