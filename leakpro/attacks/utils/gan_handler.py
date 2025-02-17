"""Module for handling GANs."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from leakpro.attacks.utils.model_handler import GeneratorHandler
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.signals.signal_extractor import PytorchModel
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class GANHandler(GeneratorHandler):
    """Class for handling GANs."""

    def __init__(self:Self, handler: AbstractInputHandler) -> None:
        """Initialize the GAN handler.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.

        """
        super().__init__(handler)

    def create_gan(self:Self, public_dataset:DataLoader) -> tuple:
        """Create the GAN models.

        Returns
        -------
            tuple: The Generator and Discriminator models.

        """
        logger.info("Creating the GAN models")



        return self.generator, self.discriminator


    def get_generator(self:Self) -> PytorchModel:
        """Get the Generator model.

        Returns
        -------
            PytorchModel: The Generator model.

        """
        logger.info("Getting the Generator model")

        return PytorchModel(self.generator.model)

