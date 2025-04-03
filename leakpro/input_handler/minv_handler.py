"""Parent class for user inputs."""

import inspect
import types

import joblib
import numpy as np
import torch
from pydantic import BaseModel
from torch import nn, optim
from torch.utils.data import DataLoader

from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.input_handler.user_imports import get_class_from_module, import_module_from_file
from leakpro.schemas import DataLoaderConfig, LossConfig, MIAMetaDataSchema, OptimizerConfig
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class MINVHandler:
    """Parent class for user inputs."""

    def __init__(self:Self, configs: dict, user_input_handler:AbstractInputHandler) -> None:
        self.configs = configs
        self._load_model_class()
        self._load_target_metadata()
        self._load_trained_target_model()
        self._load_public_data()
        self._load_private_data()
        self._load_criterion()
        self._load_dataloader_params()

        # Attach methods to Handler explicitly defined in AbstractInputHandler from user_input_handler
        for name, _ in inspect.getmembers(AbstractInputHandler, predicate=inspect.isfunction):
            if hasattr(user_input_handler, name) and not name.startswith("__"):
                attr = getattr(user_input_handler, name)
                if callable(attr):
                    attr = types.MethodType(attr, self) # ensure to properly bind methods to handler
                setattr(self, name, attr)

        # Save the Data-creation class to allow for creation of datasets with the same properties
        self.UserDataset = user_input_handler.UserDataset
    def _load_public_data(self: Self) -> None:
        """Load the public dataset."""
        public_data_path = self.configs.target.public_data_path
        try:
            with open(public_data_path, "rb") as f:
                self.public_dataset = joblib.load(f)
            logger.info(f"Loaded public data from {public_data_path}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find the public data at {public_data_path}") from e

    def _load_private_data(self: Self) -> None:
        """Load the private dataset."""
        private_data_path = self.configs.target.data_path
        try:
            with open(private_data_path, "rb") as f:
                self.private_dataset = joblib.load(f)
            logger.info(f"Loaded private data from {private_data_path}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find the private data at {private_data_path}") from e

    def _load_model_class(self: Self) -> None:
        """Get the model class blueprint from the target module."""
        model_class = self.configs.target.model_class
        if model_class is None:
            raise ValueError("model_class not found in configs.")

        module_path=self.configs.target.module_path
        if module_path is None:
            raise ValueError("module_path not found in configs.")

        try:
            target_module = import_module_from_file(module_path)
            self.target_model_blueprint = get_class_from_module(target_module, model_class)
            logger.info(f"Target model blueprint created from {model_class} in {module_path}.")
        except Exception as e:
            raise ValueError(f"Failed to create the target model blueprint from {model_class} in {module_path}") from e

    def _load_target_metadata(self: Self) -> None:
        """Get the target model metadata from the trained model metadata file."""
        target_model_metadata_path = self.configs.target.target_folder
        if target_model_metadata_path is None:
            raise ValueError("Trained model metadata path not found in configs.")
        try:
            self.target_model_metadata_path = f"{target_model_metadata_path}/model_metadata.pkl"
            with open(self.target_model_metadata_path, "rb") as f:
                target_model_metadata = joblib.load(f)

                # check if the metadata is a schema or a dict, initate a schema
                if not isinstance(target_model_metadata, BaseModel):
                    self.target_model_metadata = MIAMetaDataSchema(**target_model_metadata)
                else:
                    self.target_model_metadata = target_model_metadata

                self.train_indices = self.target_model_metadata.train_indices
                self.test_indices = self.target_model_metadata.test_indices

            logger.info(f"Loaded target model metadata from {self.target_model_metadata_path}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find the target model metadata at {self.target_model_metadata_path}") from e

    def _load_trained_target_model(self:Self) -> None:
        """Get the trained target model."""
        model_path = self.configs.target.target_folder
        if model_path is None:
            raise ValueError("Trained model path not found in configs.")
        self.model_path = f"{model_path}/target_model.pkl"
        init_params = self.target_model_metadata.init_params
        try:
            with open(self.model_path, "rb") as f:
                self.target_model = self.target_model_blueprint(**init_params)
                self.target_model.load_state_dict(torch.load(f))
            logger.info(f"Loaded target model from {model_path}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find the trained target model at {model_path}") from e

    def _load_criterion(self:Self) -> None:
        """Get the criterion for the target model."""

        criterion_config = self.target_model_metadata.criterion
        if not isinstance(criterion_config, LossConfig):
            raise ValueError("Criterion is not a valid schema.")

        # create dict of losses as {name (lower-case): Class}
        loss_classes = {
            cls.__name__.lower(): cls for cls in vars(nn.modules.loss).values()
            if isinstance(cls, type) and issubclass(cls, nn.Module)
        }

        loss_cls = loss_classes.get(criterion_config.name.lower())  # Retrieve correct case-sensitive name
        if loss_cls is None:
            raise ValueError(f"Criterion {criterion_config.name} not found in torch.nn.modules.loss")

        # Retrieve only valid __init__ parameters
        valid_params = inspect.signature(loss_cls.__init__).parameters
        filtered_params = {k: v for k, v in criterion_config.params.items() if k in valid_params}

        self._criterion = loss_cls(**filtered_params)  # Instantiate the loss function

    def _load_dataloader_params(self:Self) -> None:
        dataloader_config = self.target_model_metadata.data_loader
        if not isinstance(dataloader_config, DataLoaderConfig):
            raise ValueError("Dataloader is not a valid schema.")
        self.dataloader_config = dataloader_config

    #------------------------------------------------
    # get-set methods
    #------------------------------------------------
    def get_target_model_blueprint(self:Self) -> torch.nn.Module:
        """Get the target model blueprint."""
        return self._target_model_blueprint

    def set_target_model_blueprint(self:Self, value:torch.nn.Module) -> None:
        """Set the target model blueprint."""
        self._target_model_blueprint = value

    def get_target_model(self:Self) -> torch.nn.Module:
        """Get the trained target model wrapped as PyTorchModel."""
        return self._target_model

    def set_target_model(self:Self, model:torch.nn.Module) -> None:
        """Set the trained target model."""
        self._target_model = model

    def get_target_model_metadata(self:Self) -> dict:
        """Get the metadata of the target model."""
        return self._target_model_metadata

    def set_target_model_metadata(self:Self, metadata:dict) -> None:
        """Set the metadata of the target model."""
        self._target_model_metadata = metadata

    def get_train_indices(self:Self) -> np.ndarray:
        """Get the training indices of the target model."""
        return self.train_indices

    def get_test_indices(self:Self) -> np.ndarray:
        """Get the testing indices of the target model."""
        return self.test_indices

    def get_criterion(self:Self) -> nn.modules.loss._Loss:
        """Get the criterion for the target model."""
        return self._criterion

    def get_optimizer(self:Self, model:torch.nn.Module) -> None:
        """Get an instance of same optimizer as used in target model."""

        optimizer_config = self.target_model_metadata.optimizer
        if not isinstance(optimizer_config, OptimizerConfig):
            raise ValueError("Optimizer is not a valid schema.")

        # Get all optimizer class names from torch.optim both case sensitive and lower-case
        optimizer_classes = {
            cls.__name__.lower(): cls for cls in vars(optim).values()
            if isinstance(cls, type) and issubclass(cls, optim.Optimizer)
        }
        optimizer_cls = optimizer_classes.get(optimizer_config.name.lower())

        if optimizer_cls is None:
            raise ValueError(f"Optimizer {self.name} not found in torch.optim")

        # Retrieve only valid __init__ parameters
        valid_params = inspect.signature(optimizer_cls.__init__).parameters
        filtered_params = {k: v for k, v in optimizer_config.params.items() if k in valid_params}

        return optimizer_cls(model.parameters(), **filtered_params)

    def get_public_dataloader(self:Self, batch_size: int) -> DataLoader:
        """Return the public dataset dataloader."""
        return DataLoader(self.public_dataset, batch_size = batch_size, shuffle=False)

    def get_private_dataloader(self:Self, batch_size: int) -> DataLoader:
        """Return the private dataset dataloader."""
        return DataLoader(self.private_dataset, batch_size = batch_size, shuffle=False)

    def get_num_classes(self:Self) -> int:
        """Return the number of classes in the target model."""
        # TODO: Implement this to work with different types of datasets (where the label is not always called y)
        return self.private_dataset.y.unique().shape[0]
