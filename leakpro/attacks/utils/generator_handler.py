"""Module for handling Generators."""


import numpy as np
import torch

from leakpro.attacks.utils.model_handler import ModelHandler
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.signals.signal_extractor import PytorchModel
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class GeneratorHandler(ModelHandler):
    """Base class for handling generative models like GANs, diffusion models etc..."""

    def __init__(self:Self, handler: AbstractInputHandler) -> None:
        """Initialize the Generator handler.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.

        """
        super().__init__(handler)

        self.generator = self._get_generator()

    def _get_generator(self:Self) -> PytorchModel:
        """Get the Generator model.

        Returns
        -------
            PytorchModel: The Generator model.

        """
        logger.info("Getting the Generator model")
        return PytorchModel(self.model_blueprint, self.handler.get_criterion())

    def save_generator(self:Self, path:str) -> None:
        """Save the Generator model to a file.

        Args:
        ----
            path (str): The path to save the Generator model.

        """
        logger.info("Saving the Generator model")
        torch.save(self.generator.model.state_dict(), path)

    def load_generator(self:Self, path:str) -> None:
        """Load the Generator model from a file.

        Args:
        ----
            path (str): The path to load the Generator model.

        """
        logger.info("Loading the Generator model")
        self.generator.model.load_state_dict(torch.load(path))

    def sample_data(self:Self, num_samples:int) -> np.ndarray:
        """Sample data from the Generator model.

        Args:
        ----
            num_samples (int): The number of samples to generate.

        Returns:
        -------
            np.ndarray: The generated data.

        """
        logger.info("Sampling data from the Generator model")
        self.generator.model.eval()
        with torch.no_grad():
            return self.generator.model(num_samples).device().numpy()
