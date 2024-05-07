"""Module for handling shadow models."""

import logging
import os
import pickle
import re

import joblib
import numpy as np
from torch import cuda, device, load, nn, optim, save
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from leakpro.import_helper import Self, Tuple
from leakpro.model import PytorchModel
from leakpro.utils.input_handler import get_class_from_module, import_module_from_file
from leakpro.user_code.parent_template import CodeHandler


def singleton(cls):  # noqa: ANN001, ANN201
    """Decorator to create a singleton with initialization parameters."""
    instances = {}
    params = {}

    def get_instance(*args, **kwargs):  # noqa: ANN003, ANN002, ANN202
        if cls not in instances:
            # Store the initialization parameters when the singleton is first created
            params[cls] = (args, kwargs)
            instances[cls] = cls(*args, **kwargs)  # Create the singleton instance
        elif args or kwargs:
            # Raise an error if trying to reinitialize with different parameters
            raise ValueError("Singleton already created with specific parameters.")
        return instances[cls]

    return get_instance

@singleton
class ShadowModelHandler():
    """A class handling the creation, training, and loading of shadow models."""


    def __init__(self:Self, handler: CodeHandler, config:dict, logger:logging.Logger) -> None:
        """Initialize the ShadowModelHandler.

        Args:
        ----
            target_model (Module): The target model.
            target_config (dict): The configuration of the target model.
            config (dict): The configuration of the ShadowModelHandler.
            logger (logging.Logger): The logger object for logging.

        """
        self.handler = handler

        self.logger = logger

        self.storage_path = config["storage_path"]
        # Check if the folder does not exist
        if not os.path.exists(self.storage_path):
            # Create the folder
            os.makedirs(self.storage_path)
            self.logger.info(f"Created folder {self.storage_path}")

        self.model_storage_name = "shadow_model"
        self.metadata_storage_name = "metadata"

    def create_shadow_models(
        self:Self,
        num_models:int,
        dataset_indices: np.ndarray,
        training_fraction:float
    ) -> None:
        """Create and train shadow models based on the blueprint.

        Args:
        ----
            num_models (int): The number of shadow models to create.
            dataset:indices (np.ndarray): The indices of the whole dataset available for training the shadow models.
            training_fraction (float): The fraction of the shadow model indices to use for training.

        Returns:
        -------
            None

        """
        if num_models < 0:
            raise ValueError("Number of models cannot be negative")

        entries = os.listdir(self.storage_path)
        # Define a regex pattern to match files like model_{i}.pkl
        pattern = re.compile(rf"^{self.model_storage_name}_\d+\.pkl$")
        model_files = [f for f in entries if pattern.match(f)]
        num_to_reuse = len(model_files)

        shadow_data_size = int(len(dataset_indices)*training_fraction)

        for i in range(num_to_reuse, num_models):

            shadow_data_indices = np.random.choice(dataset_indices, shadow_data_size, replace=False)

            self.logger.info(f"Created shadow dataset {i} with size {len(shadow_data_indices)}")

            self.logger.info(f"Training shadow model {i}")

            training_results = self.handler.train_shadow_model(shadow_data_indices)
            shadow_model = training_results["model"]
            meta_data = {"metrics": training_results["metrics"], "configuration": training_results["configuration"]}

            self.logger.info(f"Training shadow model {i} complete")
            with open(f"{self.storage_path}/{self.model_storage_name}_{i}.pkl", "wb") as f:
                save(shadow_model.state_dict(), f)
                self.logger.info(f"Saved shadow model {i} to {self.storage_path}")

            self.logger.info(f"Storing metadata for shadow model {i}")

            with open(f"{self.storage_path}/{self.metadata_storage_name}_{i}.pkl", "wb") as f:
                pickle.dump(meta_data, f)

            self.logger.info(f"Metadata for shadow model {i} stored in {self.storage_path}")

    def _load_shadow_model(self:Self, index:int) -> Module:
        """Load a shadow model from a saved state.

        Args:
        ----
            index (int): The index of the shadow model to load.

        Returns:
        -------
            Module: The loaded shadow model.

        """
        if index < 0:
            raise ValueError("Index cannot be negative")
        if index >= len(os.listdir(self.storage_path)):
            raise ValueError("Index out of range")
        shadow_model = self.handler.get_shadow_model_class()(**self.handler.get_shadow_model_init_params())
        with open(f"{self.storage_path}/{self.model_storage_name}_{index}.pkl", "rb") as f:
            shadow_model.load_state_dict(load(f))
            self.logger.info(f"Loaded shadow model {index}")
        
        # TODO put this into the handler. Maybe just make the handler return the Pytorch model and save the PytorchModel directly to disc
        return PytorchModel(shadow_model, self.handler.loss)

    def get_shadow_models(self:Self, num_models:int) -> list:
        """Load the the shadow models."""
        shadow_models = []
        shadow_model_indices = []
        for i in range(num_models):
            self.logger.info(f"Loading shadow model {i}")
            shadow_models.append(self._load_shadow_model(i))
            shadow_model_indices.append(i)
        return shadow_models, shadow_model_indices

    def identify_models_trained_on_samples(self:Self, shadow_model_indices: list[int], sample_indices:set[int]) -> list:
        """Identify the shadow models trained on the provided samples.

        Args:
        ----
            shadow_model_indices (list[int]): The indices of the shadow models.
            sample_indices (set[int]): The indices of the samples.

        Returns:
        -------
            list: The list of shadow models trained on the provided samples.

        """
        if shadow_model_indices is None:
            raise ValueError("Shadow model indices must be provided")
        if sample_indices is None:
            raise ValueError("Sample indices must be provided")

        if isinstance(sample_indices, list):
            sample_indices = set(sample_indices)

        self.logger.info("Identifying shadow models trained on provided samples")
        shadow_model_trained_on_data_index = np.zeros((len(shadow_model_indices), len(sample_indices)), dtype=bool)
        for i in shadow_model_indices:
            with open(f"{self.storage_path}/{self.metadata_storage_name}_{i}.pkl", "rb") as f:
                meta_data = joblib.load(f)
                train_indices = set(meta_data["configuration"]["train_indices"].tolist())

                for j in range(len(sample_indices)):
                    shadow_model_trained_on_data_index[i, j] = sample_indices[j] in train_indices

        return shadow_model_trained_on_data_index
