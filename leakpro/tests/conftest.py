"""Test for the image input handler."""
import logging
import os
import shutil

from typing import Generator

import pytest
from dotmap import DotMap

from input_handler.image_utils import setup_image_test
from input_handler.cifar10_input_handler import Cifar10InputHandler
from constants import STORAGE_PATH


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
