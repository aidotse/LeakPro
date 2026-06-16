"""Module for handling generators."""
import os

import torch
from torch.nn import Module

from leakpro.input_handler.minv_handler import MINVHandler
from leakpro.input_handler.user_imports import get_class_from_module, import_module_from_file
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class GeneratorHandler():
    """Base class for generator models like GANs, diffusion models, etc."""

    def __init__(self: Self, handler: MINVHandler, configs: dict, caller: str = "generator") -> None:
        """Initialize the GeneratorHandler base class."""
        self.handler = handler
        self._setup_generator_configs(configs.generator)
        self.trained_bool = False
        logger.info(f"Initialized GeneratorHandler with caller: {caller}")


    def _setup_generator_configs(self: Self, configs: dict) -> None:
        """Load generator-specific configurations (e.g., generator path, params)."""
        logger.info("Setting up generator configurations")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator_path = configs.module_path
        self.generator_class = configs.model_class
        self.gen_init_params = configs.init_params
        self.dim_z = self.gen_init_params.get("dim_z", 128)
        self.num_classes = self.gen_init_params.get("num_classes", 0)
        self.generator_checkpoint = configs.checkpoint_path
        logger.info(f"Generator path: {self.generator_path}, Generator class: {self.generator_class}")
        if self.generator_path and self.generator_class: # TODO: Is this check needed?
            self.generator_blueprint = self._import_model_from_path(self.generator_path, self.generator_class)
        else:
            raise ValueError("Generator path and class must be specified in the config.")

    def load_generator(self) -> Module:
        """Instantiate and return a generator model."""
        logger.info("Getting generator model with init params: %s", self.gen_init_params)

        self.generator = self.generator_blueprint(**self.gen_init_params)
        if self.generator_checkpoint and os.path.exists(self.generator_checkpoint):
            try:
                if self.generator_class == "CustomCTGAN":
                    self.generator = self.generator_blueprint.load(self.generator_checkpoint)
                else:
                    self.generator.load_state_dict(torch.load(self.generator_checkpoint))
                logger.info(f"Loaded generator model from {self.generator_checkpoint}")
                self.trained_bool = True
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Could not load generator checkpoint %s; it will be retrained. Error: %s",
                    self.generator_checkpoint,
                    e,
                )
                self._quarantine_generator_checkpoint()
        return self.generator

    def _quarantine_generator_checkpoint(self) -> None:
        """Move an unreadable checkpoint aside so the next run can retrain."""
        if not self.generator_checkpoint or not os.path.exists(self.generator_checkpoint):
            return

        corrupt_path = f"{self.generator_checkpoint}.corrupt"
        counter = 1
        while os.path.exists(corrupt_path):
            corrupt_path = f"{self.generator_checkpoint}.corrupt.{counter}"
            counter += 1

        try:
            os.replace(self.generator_checkpoint, corrupt_path)
            logger.warning("Moved unreadable generator checkpoint to %s", corrupt_path)
        except OSError as move_error:
            logger.warning(
                "Could not move unreadable generator checkpoint %s: %s",
                self.generator_checkpoint,
                move_error,
            )

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

    def _import_model_from_path(self:Self, module_path:str, model_class:str)->None:
        """Import the model from the given path.

        Args:
        ----
            module_path (str): The path to the module.
            model_class (str): The name of the model class.

        """
        try:
            module = import_module_from_file(module_path)
            return get_class_from_module(module, model_class)
        except Exception as e:
            raise ValueError(f"Failed to create model blueprint from {model_class} in {module_path}") from e

    def save_generator(self, generator: Module, path: str) -> None:
        """Save the generator model."""
        torch.save(generator.state_dict(), path)
        logger.info(f"Saved generator model to {path}")
