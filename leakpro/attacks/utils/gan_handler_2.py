"""Module for handling GANs."""
from torch.nn import Module

from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger

from .generator_handler import GeneratorHandler


class GANHandler(GeneratorHandler):
    """Handler for training and managing GANs."""

    def __init__(self: Self, handler: AbstractInputHandler) -> None:
        """Initialize the GANHandler class."""
        super().__init__(handler, caller="gan")
        self._setup_discriminator_configs()

    def _setup_discriminator_configs(self) -> None:
        """Load discriminator-specific configurations (e.g., discriminator path, params)."""
        self.discriminator_path = self.handler.configs.get("discriminator", {}).get("module_path")
        self.discriminator_class = self.handler.configs.get("discriminator", {}).get("model_class")
        self.disc_init_params = self.handler.configs.get("discriminator", {}).get("init_params", {})

        if self.discriminator_path and self.discriminator_class:
            self.discriminator_blueprint = self._import_model_from_path(self.discriminator_path, self.discriminator_class)
        else:
            raise ValueError("Discriminator path and class must be specified in the config.")

    def get_discriminator(self) -> Module:
        """Instantiate and return a discriminator model."""
        return self.discriminator_blueprint(**self.disc_init_params)

    def train(self) -> None:
        """Train the GAN model (generator and discriminator)."""
        logger.info("Training GAN...")
        # GAN-specific training logic would be implemented here.

    def generate_samples(self) -> None:
        """Generate samples using the trained generator."""
        logger.info("Generating samples using GAN...")
        # GAN-specific sample generation logic would be implemented here.
