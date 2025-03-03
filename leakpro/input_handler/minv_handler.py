"""Parent class for user inputs."""

import joblib
import torch
from pydantic import BaseModel
from torch.utils.data import DataLoader

from leakpro.input_handler.user_imports import get_class_from_module, import_module_from_file
from leakpro.schemas import MIAMetaDataSchema
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class MINVHandler:
    """Parent class for user inputs."""

    def __init__(self:Self, configs: dict) -> None:
        self.configs = configs
        self._load_model_class()
        self._load_target_metadata()
        self._load_trained_target_model()
        self._load_public_data()
        self._load_private_data()

    def _load_public_data(self) -> None:
        """Load the public dataset."""
        public_data_path = self.configs.target.public_data_path
        try:
            with open(public_data_path, "rb") as f:
                self.public_dataset = joblib.load(f)
            logger.info(f"Loaded public data from {public_data_path}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find the public data at {public_data_path}") from e

    def _load_private_data(self) -> None:
        """Load the private dataset."""
        private_data_path = self.configs.target.data_path
        try:
            with open(private_data_path, "rb") as f:
                self.private_dataset = joblib.load(f)
            logger.info(f"Loaded private data from {private_data_path}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find the private data at {private_data_path}") from e

    def _load_model_class(self:Self) -> None:
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


    def _load_target_metadata(self:Self) -> None:
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

    def get_public_dataloader(self:Self, batch_size: int) -> DataLoader:
        """Return the public dataset dataloader."""
        return DataLoader(self.public_dataset, batch_size = batch_size, shuffle=False)

    def get_private_dataloader(self:Self, batch_size: int) -> DataLoader:
        """Return the private dataset dataloader."""
        return DataLoader(self.private_dataset, batch_size = batch_size, shuffle=False)
