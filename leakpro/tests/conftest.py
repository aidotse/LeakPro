"""Test for the image input handler."""
import logging
import os
import pickle
import shutil
import copy
import random 

from typing import Generator
from torch import randn_like, save

import pytest
from dotmap import DotMap

from leakpro.tests.input_handler.image_utils import setup_image_test
from leakpro.user_inputs.cifar10_input_handler import Cifar10InputHandler
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.tests.constants import STORAGE_PATH


@pytest.fixture(scope="session")
def manage_storage_directory():
    """Fixture to create and remove the storage directory."""
    
    # Setup: Create the folder at the start of the test session
    os.makedirs(STORAGE_PATH, exist_ok=True)
    
    # Yield control back to the test session
    yield
    
    # Teardown: Remove the folder and its contents at the end of the session
    if os.path.exists(STORAGE_PATH):
        shutil.rmtree(STORAGE_PATH)

@pytest.fixture()
def image_handler(manage_storage_directory) -> Generator[Cifar10InputHandler, None, None]:
    """Fixture for the image input handler to be shared between many tests."""

    config = DotMap()
    config.target = setup_image_test()

    # Create dummy logger
    logger = logging.getLogger("dummy")
    logger.addHandler(logging.NullHandler())

    handler = Cifar10InputHandler(config, logger)

    # Yield control back to the test session
    yield handler

@pytest.fixture()
def create_shadow_models(image_handler) -> Generator[ShadowModelHandler, None, None]:
    """Fixture for the shadow model handler to be shared between many tests."""

    model_storage_name = "shadow_model"
    metadata_storage_name = "metadata"
        
    # Add shadow model configs
    model = image_handler.target_model

    # Create three models for offline and three models for online
    n_models = 3
    for online in [True, False]:
        
        # Make a funky split of the data where online choose from the train data and offline only from the test data
        if online:
            indices = random.randint(1, len(image_handler.train_indices), len(image_handler.train_indices)//2)
        else:
            indices = random.randint(1, len(image_handler.test_indices), len(image_handler.test_indices)//2)
        
        meta_data = {}
        meta_data["init_params"] = {}
        meta_data["train_indices"] = indices
        meta_data["num_train"] = len(indices)
        meta_data["optimizer"] = image_handler.configs.target.optimizer.name.lower()
        meta_data["criterion"] = image_handler.configs.target.loss.name.lower()
        meta_data["batch_size"] = image_handler.configs.target.batch_size
        meta_data["epochs"] = image_handler.configs.target.epochs
        meta_data["online"] = online
        
        for i in range(n_models):
            shadow_model = copy.deepcopy(model)
            for param in shadow_model.parameters():
                param.data = randn_like(param)  # Fills each parameter tensor with random numbers
            os.makedirs(STORAGE_PATH + f"/{model_storage_name}/model_{i}.pkl", exist_ok=True)
            with open(STORAGE_PATH + f"/{model_storage_name}/model_{i}.pkl", "wb") as f:
                save(shadow_model.state_dict(), f)
            with open(STORAGE_PATH + f"/{metadata_storage_name}_{i}.pkl", "wb") as f:
                pickle.dump(meta_data, f)

    # Yield control back to the test session
    yield