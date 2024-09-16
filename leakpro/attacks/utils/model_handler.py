"""Abstract class for the model handler."""

import os
from copy import deepcopy

import joblib
from torch import load
from torch.nn import Module

from leakpro.user_inputs.abstract_input_handler import AbstractInputHandler
from leakpro.user_inputs.user_imports import (
    get_class_from_module,
    get_criterion_mapping,
    get_optimizer_mapping,
    import_module_from_file,
)
from leakpro.utils.import_helper import Self, Tuple
from leakpro.utils.logger import logger


class ModelHandler():
    """Class to handle models used in attacks."""

    def __init__(
        self:Self,
        handler: AbstractInputHandler,
        caller:str=None
    )->None:
        """Initialize the ModelHandler class."""

        self.handler = handler

        caller_configs =  handler.configs.get(caller, None)
        if caller_configs is None:
            caller_configs = {}

        self.model_path = caller_configs.get("module_path", None)
        self.model_class = caller_configs.get("model_class", None)

        self.use_target_model_setup = self.model_path is None or self.model_class is None

        # Get the model blueprint
        if self.use_target_model_setup:
            self.model_blueprint = handler.target_model_blueprint
        else:
            self.model_blueprint = self._import_model_from_path(self.model_path, self.model_class)

        # Pick either target config or caller config
        setup_config = deepcopy(handler.target_model_metadata) if self.use_target_model_setup else caller_configs

        self.init_params = setup_config.get("init_params", {})

        # Get optimizer class
        self.optimizer_config = setup_config["optimizer"]
        optimizer_name = self.optimizer_config.pop("name") # pop to only have input parameters left
        self.optimizer_class = self._get_optimizer_class(optimizer_name)

        # Get criterion class
        self.loss_config = setup_config["loss"]
        criterion_class = self.loss_config.pop("name") # pop to only have input parameters left
        self.criterion_class = self._get_criterion_class(criterion_class)

        self.batch_size = setup_config.get("batch_size", 32)
        assert self.batch_size > 0, "Batch size must be greater than 0"

        self.epochs = setup_config.get("epochs", 40)
        assert self.epochs > 0, "Epochs must be greater than 0"

        # Set the storage paths for objects created by the handler
        storage_path = handler.configs["audit"].get("output_dir", None)
        if storage_path is not None:
            self.storage_path = f"{storage_path}/attack_objects/{caller}"
            if not os.path.exists(self.storage_path):
                # Create the folder
                os.makedirs(self.storage_path)
                logger.info(f"Created folder {self.storage_path}")
        else:
            raise ValueError("Storage path not provided")

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

    def _get_optimizer_class(self:Self, optimizer_name:str) -> None:
        """Get the optimizer class based on the optimizer name.

        Args:
        ----
            optimizer_name (str): The name of the optimizer.

        """
        try:
            return get_optimizer_mapping()[optimizer_name]
        except Exception as e:
            raise ValueError(f"Failed to create optimizer from {self.optimizer_config['name']}") from e

    def _get_criterion_class(self:Self, criterion_name:str)->None:
        """Get the criterion class based on the criterion name.

        Args:
        ----
            criterion_name (str): The name of the criterion.

        """
        try:
            return get_criterion_mapping()[criterion_name]
        except Exception as e:
            raise ValueError(f"Failed to create criterion from {self.criterion_config['name']}") from e

    def _get_model_criterion_optimizer(self:Self) -> Tuple[Module, Module, Module]:
        """Get the model, criterion, and optimizer from the handler or config."""

        # Set up shadow model from config file
        if self.use_target_model_setup:
            model, criterion, optimizer = self.handler.get_target_replica()
        else:
            model = self.model_blueprint(**self.init_params)
            optimizer = self.optimizer_class(model.parameters(), **self.optimizer_config)
            criterion = self.criterion_class(**self.loss_config)

        return model, criterion, optimizer

    def get_criterion(self:Self) -> Module:
        """Get the criterion from the handler or config."""

        return self.criterion_class(**self.loss_config)

    def _load_model(self:Self, model_path:str) -> Tuple[Module, Module]:
        """Load a model from a path.

        Args:
        ----
            model_path (str): The path to the saved model.

        Returns:
        -------
            Module: The loaded shadow model.
            Module: The loaded criterion.

        """
        # First create the blueprint to inject the weights
        try:
            model = self.model_blueprint(**self.init_params)  # noqa: E501
            criterion = self.handler.get_criterion() if self.use_target_model_setup else self.criterion_class(**self.loss_config)
        except Exception as e:
            raise ValueError("Failed to create model from blueprint") from e

        # Then load the weights and insert them into the model
        try:
            with open(model_path, "rb") as f:
                model.load_state_dict(load(f))
                logger.info(f"Loaded model from {model_path}")
            return model, criterion
        except FileNotFoundError as e:
            raise ValueError(f"Model file not found at {model_path}") from e

    def _load_metadata(self:Self, metadata_path:str) -> dict:
        """Load metadata from a saved state.

        Args:
        ----
            metadata_path (str): The path to the saved metadata.

        Returns:
        -------
            dict: The loaded metadata.

        """
        try:
            with open(metadata_path, "rb") as f:
                return joblib.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Metadata at {metadata_path} not found") from e
