"""Module for handling generators."""
import os

import torch
from torch.nn import Module

from leakpro.input_handler.minv_handler import MINVHandler
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class GeneratorHandler():
    """Base class for generator models like GANs, diffusion models, etc."""

    def __init__(self: Self, handler: MINVHandler, configs: dict, caller: str = "generator") -> None:
        """Initialize the GeneratorHandler base class."""
        self.handler = handler
        self._setup_generator_configs(configs)
        self.trained_bool = False
        logger.info(f"Initialized GeneratorHandler with caller: {caller}")


    def _setup_generator_configs(self: Self, configs: dict) -> None:
        """Load generator-specific configurations (e.g., generator path, params)."""
        logger.info("Setting up generator configurations")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator_path = configs.get("generator", {}).get("module_path")
        self.generator_class = configs.get("generator", {}).get("model_class")
        self.gen_init_params = configs.get("generator", {}).get("init_params", {})
        self.dim_z = self.gen_init_params.get("dim_z", 128)
        self.num_classes = self.gen_init_params.get("num_classes", 0)
        self.generator_checkpoint = configs.get("generator", {}).get("checkpoint_path", None)
        logger.info(f"Generator path: {self.generator_path}, Generator class: {self.generator_class}")
        if self.generator_path and self.generator_class:
            self.generator_blueprint = self._import_model_from_path(self.generator_path, self.generator_class)
        else:
            raise ValueError("Generator path and class must be specified in the config.")

    def load_generator(self) -> Module:
        """Instantiate and return a generator model."""
        logger.info("Getting generator model with init params: %s", self.gen_init_params)
        self.generator = self.generator_blueprint(**self.gen_init_params)
        if self.generator_checkpoint and os.path.exists(self.generator_checkpoint):
            self.generator.load_state_dict(torch.load(self.generator_checkpoint))
            logger.info(f"Loaded generator model from {self.generator_checkpoint}")
            self.trained_bool = True
        return self.generator

    def get_generator(self) -> Module:
        """Return the generator model."""
        if not hasattr(self, "generator"):
            self.load_generator()
        return self.generator

    def train(self) -> None:
        """Abstract training method for generator models. To be implemented by subclasses."""
        if self.trained_bool:
            logger.info("Generator has already been trained, skipping training.")
            return
        raise NotImplementedError("Subclasses must implement the train method")

    def sample_from_generator(self) -> None:
        """Abstract sample generation method for generator models. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the generate_samples method")

    '''
    def get_public_data(self, batch_size: int) -> DataLoader:
        """Return data loader for the public dataset."""
        # Get public dataloader
        self.public_path = self.handler.configs.get("public_dataset", {}).get("public_data_path")
        # Load pickle file
        try:
            with open(self.public_path, "rb") as f:
                self.public_dataset = joblib.load(f)
            logger.info(f"Loaded public data from {self.public_path}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find the public data at {self.public_path}") from e
        return DataLoader(self.public_dataset, batch_size = batch_size, shuffle=False)

    def get_private_data(self, batch_size: int) -> DataLoader: # TODO: Should be implemented in another class
        """Return data loader for the private dataset."""
        # Get private dataloader
        self.private_path = self.handler.configs.get("target", {}).get("data_path")
        # Load pickle file
        try:
            with open(self.private_path, "rb") as f:
                self.private_dataset = joblib.load(f)
            logger.info(f"Loaded private data from {self.private_path}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find the private data at {self.private_path}") from e
        return DataLoader(self.private_dataset, batch_size = batch_size, shuffle=False)
    '''

    def save_generator(self, generator: Module, path: str) -> None:
        """Save the generator model."""
        torch.save(generator.state_dict(), path)
        logger.info(f"Saved generator model to {path}")
