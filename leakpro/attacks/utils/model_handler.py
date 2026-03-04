"""Abstract class for the model handler."""

import os

import joblib
from torch import load
from torch.nn import Module

from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.input_handler.user_imports import (
    get_class_from_module,
    get_criterion_mapping,
    get_optimizer_mapping,
    import_module_from_file,
)
from leakpro.utils.import_helper import Self, Tuple
from leakpro.utils.logger import logger
from leakpro.utils.save_load import hash_model


class ModelHandler():
    """Class to handle models used in attacks."""

    def __init__(
        self:Self,
        handler: MIAHandler,
        caller:str=None
    )->None:
        """Initialize the ModelHandler class."""

        self.handler = handler

        caller_configs = getattr(handler.configs, caller) if caller is not None else None
        self.use_target_model_setup = caller_configs is None
        target_setup = handler.target_model_metadata

        # get the bluepring for the model
        if self.use_target_model_setup:
            self.model_class = handler.target_model_blueprint.__name__
            self.model_blueprint = handler.target_model_blueprint
        else:
            # Allow partial shadow model config by inheriting from the target setup.
            self.model_path = caller_configs.module_path or handler.configs.target.module_path
            self.model_class = caller_configs.model_class or handler.configs.target.model_class
            try:
                self.model_blueprint = self._import_model_from_path(self.model_path, self.model_class)
            except Exception as e:
                raise ValueError(f"Failed to create model blueprint from {self.model_class} in {self.model_path}") from e

        if self.use_target_model_setup:
            self.init_params = target_setup.init_params
            optimizer_name = target_setup.optimizer.name
            self.optimizer_config = target_setup.optimizer.params
            criterion_name = target_setup.criterion.name
            self.loss_config = target_setup.criterion.params
            self.epochs = target_setup.epochs
        else:
            # Inherit defaults from target and apply caller overrides when present.
            self.init_params = target_setup.init_params.copy()
            self.init_params.update(caller_configs.init_params or {})
            optimizer_cfg = caller_configs.optimizer or target_setup.optimizer
            criterion_cfg = caller_configs.criterion or target_setup.criterion
            optimizer_name = optimizer_cfg.name
            self.optimizer_config = optimizer_cfg.params
            criterion_name = criterion_cfg.name
            self.loss_config = criterion_cfg.params
            self.epochs = caller_configs.epochs if caller_configs.epochs is not None else target_setup.epochs

        # Get optimizer class
        self.optimizer_class = self._get_optimizer_class(optimizer_name)

        # Get criterion class
        self.criterion_class = self._get_criterion_class(criterion_name)

        # Set the storage paths for objects created by the handler
        storage_path = handler.configs.audit.output_dir
        if storage_path is not None:
            self.storage_path = f"{storage_path}/attack_objects/{caller}"
            if not os.path.exists(self.storage_path):
                # Create the folder
                os.makedirs(self.storage_path)
                logger.info(f"Created folder {self.storage_path}")
        else:
            raise ValueError("Storage path not provided")

        # Create the hash for the target model
        self.target_model_hash = hash_model(self.handler.target_model)

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
