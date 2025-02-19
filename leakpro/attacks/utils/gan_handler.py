"""Module for handling GANs."""
import os

import torch
from torch.nn import Module

from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger

from .generator_handler import GeneratorHandler


class GANHandler(GeneratorHandler):
    """Handler for training and managing GANs."""

    def __init__(self: Self, handler: AbstractInputHandler) -> None:
        """Initialize the GANHandler class."""
        super().__init__(handler, caller="gan_handler")
        self._setup_discriminator_configs()

    def _setup_discriminator_configs(self) -> None:
        """Load discriminator-specific configurations (e.g., discriminator path, params)."""
        # TODO: Fix this, not correct. We want to set the configs to default to the current attack, as done previsouly. More general this way.
        logger.info("Setting up discriminator configurations")
        self.discriminator_path = self.handler.configs.get("plgmi", {}).get("discriminator", {}).get("module_path")
        self.discriminator_class = self.handler.configs.get("plgmi", {}).get("discriminator", {}).get("model_class")
        self.disc_init_params = self.handler.configs.get("plgmi", {}).get("discriminator", {}).get("init_params", {})
        self.discriminator_checkpoint = self.handler.configs.get("plgmi", {}).get("discriminator", {}).get("checkpoint_path", None)
        logger.info(f"Discriminator path: {self.discriminator_path}, Discriminator class: {self.discriminator_class}")
        # Check that discriminator class is provided, else raise an error
        if self.discriminator_path and self.discriminator_class:
            self.discriminator_blueprint = self._import_model_from_path(self.discriminator_path, self.discriminator_class)
        else:
            raise ValueError("Discriminator path and class must be specified in the config.")

    def get_discriminator(self) -> Module:
        """Instantiate and return a discriminator model."""
        discriminator = self.discriminator_blueprint(**self.disc_init_params)
        if self.discriminator_checkpoint and os.path.exists(self.discriminator_checkpoint):
            discriminator.load_state_dict(torch.load(self.discriminator_checkpoint))
        return discriminator

    def train(self) -> None:
        """Train the GAN model (generator and discriminator)."""
        logger.info("Training GAN...")
        # GAN-specific training logic would be implemented here.
        self.handler.train_gan(self)

    def sample_from_generator(self, gen: Module, n_classes: int, batch_size: int, device: torch.device, dim_z: int) -> tuple:
        """Sample random z and y from the generator."""
        z = torch.empty(batch_size, dim_z, dtype=torch.float32, device=device).normal_()
        y = torch.randint(0, n_classes, (batch_size,)).to(device)
        return gen(z, y), y, z

    def save_discriminator(self, discriminator: Module, path: str) -> None:
        """Save the discriminator model."""
        torch.save(discriminator.state_dict(), path)