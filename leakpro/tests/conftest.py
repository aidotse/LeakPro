"""Test for the image input handler."""
import os
import shutil

import pytest
import yaml
from dotmap import DotMap

from leakpro import LeakPro
from leakpro.tests.constants import STORAGE_PATH, get_audit_config
from leakpro.tests.input_handler.image_input_handler import ImageInputHandler
from leakpro.tests.input_handler.image_utils import setup_image_test
from leakpro.tests.input_handler.tabular_input_handler import TabularInputHandler
from leakpro.tests.input_handler.tabular_utils import setup_tabular_test


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

@pytest.fixture
def image_handler(manage_storage_directory) -> ImageInputHandler:
    """Fixture for the image input handler to be shared between many tests."""

    config = DotMap()
    config.target = setup_image_test()
    config.audit = get_audit_config()
    config.audit.data_modality = "image"
    #save config to file
    config_path = f"{STORAGE_PATH}/image_test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config.toDict(), f)

    leakpro = LeakPro(ImageInputHandler, config_path)
    handler = leakpro.handler
    handler.configs = DotMap(handler.configs)

    # Yield control back to the test session
    return handler

@pytest.fixture
def tabular_handler(manage_storage_directory) -> TabularInputHandler:
    """Fixture for the image input handler to be shared between many tests."""

    config = DotMap()
    config.target = setup_tabular_test()
    config.audit = get_audit_config()
    config.audit.data_modality = "tabular"
    #save config to file
    config_path = f"{STORAGE_PATH}/tabular_test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config.toDict(), f)

    leakpro = LeakPro(TabularInputHandler, config_path)
    handler = leakpro.handler
    handler.configs = DotMap(handler.configs)

    # Yield control back to the test session
    return handler
