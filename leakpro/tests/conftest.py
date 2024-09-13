"""Test for the image input handler."""
import os
import shutil

import sys
print (sys.path)

from typing import Generator

import pytest
from dotmap import DotMap

from leakpro.tests.input_handler.image_utils import setup_image_test
from leakpro.tests.input_handler.cifar10_input_handler import Cifar10InputHandler
from leakpro.tests.constants import STORAGE_PATH, get_audit_config


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
    config.audit = get_audit_config()

    handler = Cifar10InputHandler(config)

    # Yield control back to the test session
    yield handler
