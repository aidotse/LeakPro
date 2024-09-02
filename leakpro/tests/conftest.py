"""Test for the image input handler."""
import logging
import os
from typing import Generator

import pytest
from dotmap import DotMap

from leakpro.tests.input_handler.image_utils import setup_image_test
from leakpro.user_inputs.cifar10_input_handler import Cifar10InputHandler


@pytest.fixture
def image_handler() -> Generator[Cifar10InputHandler, None, None]:
    """Fixture for the image input handler to be shared between many tests."""

    config = DotMap()
    config.target = setup_image_test()

    # Create dummy logger
    logger = logging.getLogger("dummy")
    logger.addHandler(logging.NullHandler())

    handler = Cifar10InputHandler(config, logger)

    yield handler

    # Clean up folder tests are done
    os.remove(config.target.data_path) # remove data
    os.remove(config.target.trained_model_path) # remove target model
    os.remove(config.target.trained_model_metadata_path) # remove target metadata
