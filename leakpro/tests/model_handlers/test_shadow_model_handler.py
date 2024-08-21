"""Test the shadow model handler module."""
import os
import shutil
from dotmap import DotMap
import numpy as np

from pytest import raises
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.user_inputs.cifar10_input_handler import Cifar10InputHandler

def _setup_shadow_test() -> DotMap:
    """Setup the shadow model test."""
    shadow_model_config = DotMap()
    shadow_model_config.module_path = "./leakpro/tests/input_handler/image_utils.py"
    shadow_model_config.model_class = "ConvNet"
    shadow_model_config.storage_path = "./leakpro/tests/model_handlers/model_handler_output"
    shadow_model_config.batch_size = 32
    shadow_model_config.epochs = 1
    shadow_model_config.optimizer = {"name": "sgd", "lr": 0.001}
    shadow_model_config.loss = {"name": "crossentropyloss"}
    return shadow_model_config


def test_shadow_model_handler_singleton(image_handler:Cifar10InputHandler) -> None:
    """Test that only one instance gets created."""
    shadow_model_config = _setup_shadow_test()
    image_handler.configs.shadow_model = shadow_model_config
    sm = ShadowModelHandler(image_handler)
    
    with raises(ValueError) as excinfo:
        ShadowModelHandler(image_handler)
    assert str(excinfo.value) == "Singleton already created with specific parameters."
    
    # delete the singleton to not get error in the next tests
    del sm 

def test_shadow_model_creation_and_loading(image_handler:Cifar10InputHandler) -> None:
    shadow_model_config = _setup_shadow_test()
    image_handler.configs.shadow_model = shadow_model_config
    
    # Test initialization
    sm = ShadowModelHandler(image_handler)
    
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
    
    entries = os.listdir(sm.storage_path)
    assert len(entries) == 0
    
    sm.create_shadow_models(n_models, image_handler.test_indices, training_fraction, online)
    
    entries = os.listdir(sm.storage_path)
    assert len(entries) == 2*n_models
    assert "metadata_0.pkl" in entries
    assert "shadow_model_0.pkl" in entries
    
    sm.create_shadow_models(n_models, image_handler.test_indices, training_fraction, ~online)
    trained_models = 2
    entries = os.listdir(sm.storage_path)
    assert len(entries) == 2*trained_models
    assert "metadata_0.pkl" in entries
    assert "shadow_model_0.pkl" in entries
    assert "metadata_1.pkl" in entries
    assert "shadow_model_1.pkl" in entries
    
    # Test loading
    meta_0 = sm._load_metadata(sm.storage_path + "/metadata_0.pkl")
    meta_1 = sm._load_metadata(sm.storage_path + "/metadata_1.pkl")
    assert meta_0["online"] == online
    assert meta_1["online"] == ~online
    assert meta_1["num_train"] == meta_0["num_train"]
    
    shadow_model_indices = [0,1]
    models, indices = sm.get_shadow_models(shadow_model_indices)
    for model in models:
        assert model.model_obj.__class__.__name__ == "ConvNet"
    
    # Test index mask
    mask = sm.get_in_indices_mask(shadow_model_indices, np.array(image_handler.test_indices))
    
    true_mask_0 = np.array([True if item in meta_0["train_indices"] else False for item in image_handler.test_indices])
    assert np.array_equal(mask[:, 0], true_mask_0)
    
    true_mask_1 = np.array([True if item in meta_1["train_indices"] else False for item in image_handler.test_indices])
    assert np.array_equal(mask[:, 1], true_mask_1)
    
    # remove created files
    shutil.rmtree(sm.storage_path)
    del sm
    