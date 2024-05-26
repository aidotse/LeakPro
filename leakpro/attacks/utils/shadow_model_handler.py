"""Module for handling shadow models."""

import logging
import os
import pickle
import re

import joblib
import numpy as np
import torch
from torch import Tensor, cuda, device, jit, load, nn, optim, save
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from leakpro.import_helper import Self, Tuple
from leakpro.model import PytorchModel
from leakpro.user_inputs.abstract_input_handler import AbstractInputHandler
from leakpro.utils.input_handler import get_class_from_module, import_module_from_file


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


    def __init__(self:Self, handler: AbstractInputHandler) -> None:
        """Initialize the ShadowModelHandler.

        Args:
        ----
            target_model (Module): The target model.
            target_config (dict): The configuration of the target model.
            config (dict): The configuration of the ShadowModelHandler.
            logger (logging.Logger): The logger object for logging.

        """
        config = handler.configs["shadow_model"]
        self.logger = handler.logger
        self.handler = handler

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
        dataset:Dataset,
        indicies: np.ndarray,
        training_fraction:float=0.1,
        retrain:bool = False
    ) -> None:
        """Create and train shadow models based on the blueprint.

        Args:
        ----
            num_models (int): The number of shadow models to create.
            # dataset:indices (np.ndarray): The indices of the whole dataset available for training the shadow models.
            # training_fraction (float): The fraction of the shadow model indices to use for training.
            dataset (torch.utils.data.Dataset): The full dataset available for training the shadow models.
            indicies (list): The indices to use from the dataset for training the shadow models.
            training_fraction (float): The fraction of the dataset to use for training.
            retrain (bool): Whether to retrain the shadow models or not.

        Returns:
        -------
            None

        """
        if num_models < 0:
            raise ValueError("Number of models cannot be negative")

        if retrain:
            self.logger.info("Retraining shadow models")
            num_to_reuse = 0
        else:
            entries = os.listdir(self.storage_path)
            # Define a regex pattern to match files like model_{i}.pkl
            pattern = re.compile(rf"^{self.model_storage_name}_\d+\.pkl$")
            model_files = [f for f in entries if pattern.match(f)]
            num_to_reuse = len(model_files)

        # Get the size of the dataset
        shadow_data_size = int(len(indicies)*training_fraction)

        for i in range(num_to_reuse, num_models):

            shadow_data_indices = np.random.choice(indicies, shadow_data_size, replace=False)
            shadow_dataset = dataset.subset(shadow_data_indices)
            shadow_train_loader = DataLoader(shadow_dataset, batch_size=self.batch_size, shuffle=True)
            self.logger.info(f"Created shadow dataset {i} with size {len(shadow_dataset)}")

            self.logger.info(f"Training shadow model {i}")

            training_results = self.handler.train_shadow_model(shadow_data_indices)
            shadow_model = training_results["model"]
            meta_data = {"metrics": training_results["metrics"], "configuration": training_results["configuration"]}

            self.logger.info(f"Training shadow model {i} complete")
            with open(f"{self.storage_path}/{self.model_storage_name}_{i}.pkl", "wb") as f:
                save(shadow_model.state_dict(), f)
                self.logger.info(f"Saved shadow model {i} to {self.storage_path}")

            self.logger.info(f"Storing metadata for shadow model {i}")
            meta_data = {}
            meta_data["init_params"] = self.init_params
            meta_data["train_indices"] = shadow_data_indices
            meta_data["num_train"] = len(shadow_data_indices)
            meta_data["optimizer"] = self.optimizer_class.__name__
            meta_data["criterion"] = self.criterion_class.__name__
            meta_data["batch_size"] = self.batch_size
            meta_data["epochs"] = self.epochs
            meta_data["learning_rate"] = self.optimizer_config["lr"]
            meta_data["weight_decay"] = self.optimizer_config.get("weight_decay", 0.0)

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

        shadow_model = self.shadow_model_blueprint(**self.init_params)

        try:
            with open(f"{self.storage_path}/{self.model_storage_name}_{index}.pkl", "rb") as f:
                shadow_model.load_state_dict(load(f))
                self.logger.info(f"Loaded shadow model {index}")
            return PytorchModel(shadow_model, self.criterion_class(**self.loss_config))
        except FileNotFoundError:
            self.logger.error(f"Could not find the shadow model {index}")
            return None

    def get_shadow_models(self:Self, num_models:int) -> Tuple[list, list]:
        """Load the the shadow models."""
        shadow_models = []
        shadow_model_indices = []
        for i in range(num_models):
            self.logger.info(f"Loading shadow model {i}")
            model = self._load_shadow_model(i)
            shadow_models.append(model)
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

    def _load_metadata(self:Self, index:int) -> dict:
        """Load a shadow model from a saved state.

        Args:
        ----
            index (int): The index of the shadow model to load metadata for.

        Returns:
        -------
            Module: The loaded metadata.

        """
        if index < 0:
            raise ValueError("Index cannot be negative")
        if index >= len(os.listdir(self.storage_path)):
            raise ValueError("Index out of range")

        try:
            with open(f"{self.storage_path}/{self.metadata_storage_name}_{index}.pkl", "rb") as f:
                return joblib.load(f)
        except FileNotFoundError:
            self.logger.error(f"Could not find the metadata for shadow model {index}")
            return None

    def get_shadow_model_metadata(self:Self, num_models:int) -> list:
        """Load the the shadow model metadata."""
        metadata = []
        for i in range(num_models):
            self.logger.info(f"Loading metadata {i}")
            metadata.append(self._load_metadata(i))
        return metadata

    def get_in_indices_mask(self:Self, num_models:int, dataset:np.ndarray) -> np.ndarray:
        """Get the mask indicating which indices in the dataset are present in the shadow model training set.

        Args:
        ----
            num_models (int): The number of shadow models.
            dataset (np.ndarray): The dataset.

        Returns:
        -------
            np.ndarray: The mask indicating which indices are present in the shadow model training set.

        """
        # Retrieve metadata for shadow models
        metadata = self.get_shadow_model_metadata(num_models)

        # Extract training indices for each shadow model
        models_in_indices = [data["train_indices"] for data in metadata]

        # Convert to numpy array for easier manipulation
        models_in_indices = np.asarray(models_in_indices)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_indices_tensor = torch.from_numpy(models_in_indices).to(device=device)
        dataset_tensor = torch.from_numpy(dataset).to(device=device)
        indice_masks_tensor = torch.zeros((len(dataset), len(models_in_indices)), dtype=torch.bool, device=device)

        return torch_indice_in_shadowmodel_training_set(indice_masks_tensor,\
                                                        dataset_tensor, model_indices_tensor).cpu().numpy()

@jit.script
def torch_indice_in_shadowmodel_training_set(in_tensor:Tensor, dataset:Tensor, model_indices:Tensor) -> Tensor:
    """Check if an audit indice is present in the shadow model training set.

    Args:
    ----
        in_tensor (Tensor): Tensor to store the mask(s) ( audit dataset x num models )
        dataset (Tensor): The tensor containing all audit indices to check. ( audit dataset )
        model_indices (Tensor): The tensor of indices for the shadow model training set. ( num models x train dataset)

    Returns:
    -------
        in_tensor (Tensor): The mask(s) indicating if the audit indices is present in each shadow model training set.

    """
    for i in range(in_tensor.shape[1]):
        in_tensor[:, i] = torch.isin(dataset, model_indices[i, :])
    return in_tensor
