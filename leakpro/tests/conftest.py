"""Test for the image input handler."""
import logging
import os
from typing import Generator

import pytest
from dotmap import DotMap

from leakpro.tests.input_handler.image_utils import setup_image_test
from leakpro.user_inputs.cifar10_input_handler import Cifar10InputHandler
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.tests.constants import STORAGE_PATH

@pytest.fixture()
def image_handler() -> Generator[Cifar10InputHandler, None, None]:
    """Fixture for the image input handler to be shared between many tests."""

    config = DotMap()
    config.target = setup_image_test()

    # Create dummy logger
    logger = logging.getLogger("dummy")
    logger.addHandler(logging.NullHandler())

    handler = Cifar10InputHandler(config, logger)

    yield handler

    # Clean up folder when tests are done
    if os.path.exists(config.target.data_path):
        os.remove(config.target.data_path) # remove data
    if os.path.exists(config.target.module_path):
        os.remove(config.target.module_path)
    if os.path.exists(config.target.trained_model_path):
        os.remove(config.target.trained_model_metadata_path) # remove target metadata

@pytest.fixture()
def offline_shadow_model_handler(image_handler) -> Generator[ShadowModelHandler, None, None]:
    """Fixture for the shadow model handler to be shared between many tests."""

    # Add shadow model configs
    image_handler.configs.shadow_model = DotMap()
    image_handler.configs.shadow_model.storage_path = STORAGE_PATH + "/shadow_models"
    image_handler.configs.shadow_model.model_class = "ConvNet"
    image_handler.configs.shadow_model.optimizer = {"name": "sgd", "lr": 0.001}
    image_handler.configs.shadow_model.loss = {"name": "crossentropyloss", "init_params": {}}
    

    n_models = 3
    training_fraction = 0.5
    online = False

    shadow_handler = ShadowModelHandler(image_handler)
    shadow_handler.create_shadow_models(n_models, image_handler.test_indices, training_fraction, online)
    
    yield shadow_handler

    # Clean up folder tests are done
    if os.path.exists(STORAGE_PATH + "/shadow_models"):
        os.remove(STORAGE_PATH + "/shadow_models")