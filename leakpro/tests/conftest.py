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
