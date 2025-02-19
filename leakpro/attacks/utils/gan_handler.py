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

    def __init__(self: Self, handler: AbstractInputHandler, configs: dict) -> None:
        """Initialize the GANHandler class."""
        super().__init__(handler, configs=configs, caller="gan_handler")
        self._setup_discriminator_configs(configs)

    def _setup_discriminator_configs(self: Self, configs : dict) -> None:
        """Load discriminator-specific configurations (e.g., discriminator path, params)."""
        logger.info("Setting up discriminator configurations")
        self.discriminator_path = configs.get("discriminator", {}).get("module_path")
        self.discriminator_class = configs.get("discriminator", {}).get("model_class")
        self.disc_init_params = configs.get("discriminator", {}).get("init_params", {})
        self.discriminator_checkpoint = configs.get("discriminator", {}).get("checkpoint_path", None)
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

    def sample_from_generator(self,
                                gen: Module,
                                n_classes: int,
                                batch_size: int,
                                device: torch.device,
                                dim_z: int,
                                label: int = None) -> tuple:
        """Samples data from a given generator model.

        Args:
            gen (Module): The generator model to sample from.
            n_classes (int): The number of classes for the conditional generation.
            batch_size (int): The number of samples to generate.
            device (torch.device): The device to perform the computation on.
            dim_z (int): The dimensionality of the latent space.
            label (int): The optional class label to generate samples for.

        Returns:
            tuple: A tuple containing the generated samples, the class labels, and the latent vectors.

        """

        z = torch.empty(batch_size, dim_z, dtype=torch.float32, device=device).normal_()
        if label is not None:
            y = torch.tensor([label] * batch_size).to(device)
        else:
            y = torch.randint(0, n_classes, (batch_size,)).to(device)
        return gen(z, y), y, z

    def save_discriminator(self, discriminator: Module, path: str) -> None:
        """Save the discriminator model."""
        torch.save(discriminator.state_dict(), path)