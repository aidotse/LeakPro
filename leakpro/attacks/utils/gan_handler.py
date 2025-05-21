"""Module for handling GANs."""
import os

import torch
from torch.nn import Module

from leakpro.input_handler.minv_handler import MINVHandler
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger

from .generator_handler import GeneratorHandler


class GANHandler(GeneratorHandler):
    """Handler for training and managing GANs."""

    def __init__(self: Self, handler: MINVHandler, configs: dict) -> None:
        """Initialize the GANHandler class."""
        logger.info("Initializing GANHandler...")

        super().__init__(handler, configs=configs, caller="gan_handler")
        self._setup_discriminator_configs(configs.discriminator)

    def _setup_discriminator_configs(self: Self, configs : dict) -> None:
        """Load discriminator-specific configurations (e.g., discriminator path, params)."""
        logger.info("Setting up discriminator configurations")
        self.discriminator_path = configs.module_path
        self.discriminator_class = configs.model_class
        self.disc_init_params = configs.init_params
        self.discriminator_checkpoint = configs.checkpoint_path
        logger.info(f"Discriminator path: {self.discriminator_path}, Discriminator class: {self.discriminator_class}")
        # Check that discriminator class is provided, else raise an error
        if self.discriminator_path and self.discriminator_class: # TODO: Is this check needed?
            self.discriminator_blueprint = self._import_model_from_path(self.discriminator_path, self.discriminator_class)
        else:
            raise ValueError("Discriminator path and class must be specified in the config.")

    def load_discriminator(self) -> Module:
        """Instantiate and return a discriminator model."""
        self.discriminator = self.discriminator_blueprint(**self.disc_init_params)
        if self.discriminator_checkpoint and os.path.exists(self.discriminator_checkpoint):
            logger.info(f"Loading discriminator model from {self.discriminator_checkpoint}")
            self.discriminator.load_state_dict(torch.load(self.discriminator_checkpoint))
        return self.discriminator

    def get_discriminator(self) -> Module:
        """Return the discriminator model."""
        if not hasattr(self, "discriminator"):
            self.discriminator = self.load_discriminator()
        return self.discriminator

    def train(self) -> None:
        """Train the GAN model (generator and discriminator)."""
        logger.info("Training GAN...")
        # GAN-specific training logic would be implemented here.
        self.handler.train_gan(self)

    def sample_from_generator(self,
                                batch_size: int = None,
                                label: int = None,
                                z: torch.tensor = None) -> tuple:
        """Samples data from a given generator model.

        Args:
            batch_size (int): The number of samples to generate.
            label (int): The optional class label to generate samples for, otherwise random.
            z (torch.tensor): The latent vector to generate samples from, otherwise random.

        Returns:
        -------
            tuple: A tuple containing the generated samples, the class labels, and the latent vectors.

        """
        if batch_size is None:
            batch_size = 1

        if z is not None:
            z = z.unsqueeze(0).expand(batch_size, -1).to(self.device)
        else:
            z = torch.empty(batch_size, self.dim_z, dtype=torch.float32, device=self.device).normal_()

        if label is not None:
            y = torch.tensor([label] * batch_size).to(self.device)
        else:
            y = torch.randint(0, self.num_classes, (batch_size,)).to(self.device)
        return self.generator(z, y), y, z

    def save_discriminator(self, discriminator: Module, path: str) -> None:
        """Save the discriminator model."""
        torch.save(discriminator.state_dict(), path)
