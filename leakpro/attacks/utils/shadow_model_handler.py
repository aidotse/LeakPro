"""Module for handling shadow models."""

import os
import pickle
import re

import numpy as np
import torch
from torch import Tensor, jit, save
from torch.nn import Module

from leakpro.attacks.utils.model_handler import ModelHandler
from leakpro.import_helper import Self, Tuple
from leakpro.signal_extractor import PytorchModel
from leakpro.user_inputs.abstract_input_handler import AbstractInputHandler


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
class ShadowModelHandler(ModelHandler):
    """A class handling the creation, training, and loading of shadow models."""

    def __init__(self:Self, handler: AbstractInputHandler) -> None:  # noqa: PLR0912
        """Initialize the ShadowModelHandler.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.

        """
        super().__init__(handler)
        self.configs = handler.configs["shadow_model"]

        # Read from the config file
        module_path = self.configs.get("module_path", None)
        model_class = self.configs.get("model_class", None)
        self.optimizer_config = self.configs.get("optimizer", None)
        self.loss_config = self.configs.get("loss", None)
        self.batch_size = self.configs.get("batch_size", 32)
        self.epochs = self.configs.get("epochs", 40)
        self.storage_path = self.configs.get("storage_path")

        if module_path is None or model_class is None:
            self.model_blueprint = None
            self.criterion_class = None
            self.optimizer_class = None
        else:
            self.init_params = self.configs.get("init_params", {})
            self._import_model_from_path(module_path, model_class)

            # Read the optimizer for shadow models if it has been provided
            if self.optimizer_config is None:
                raise ValueError("Optimizer configuration not found in configs.")
            optimizer_name = self.optimizer_config.pop("name") # pop to only have input parameters left
            self._get_optimizer_class(optimizer_name)

            # Read the loss function for shadow models if it has been provided
            if self.loss_config is None:
                raise ValueError("Loss configuration not found in configs.")
            criterion_class = self.loss_config.pop("name") # pop to only have input parameters left
            self._get_criterion_class(criterion_class)

        # Create the shadow model storage folder
        if self.storage_path is None:
            raise ValueError("Storage path for shadow models not provided")
        # Check if the folder does not exist
        if not os.path.exists(self.storage_path):
            # Create the folder
            os.makedirs(self.storage_path)
            self.logger.info(f"Created folder {self.storage_path}")

        # Set up the names of the shadow model
        self.model_storage_name = "shadow_model"
        self.metadata_storage_name = "metadata"

    def _filter(self:Self, data_size:int, online:bool)->list[int]:
        # Get the metadata for the shadow models
        entries = os.listdir(self.storage_path)
        pattern = re.compile(rf"^{self.metadata_storage_name}_\d+\.pkl$")
        files = [f for f in entries if pattern.match(f)]
        # Extract the index of the metadata
        all_indices = [int(re.search(r"\d+", f).group()) for f in files]
        # Filter out indices to only keep the ones with the same data size
        filtered_indices = []
        for i in all_indices:
            metadata = self._load_shadow_metadata(i)
            if metadata["num_train"] == data_size and metadata["online"] == online:
                filtered_indices.append(i)
        return all_indices, filtered_indices

    def create_shadow_models(
        self:Self,
        num_models:int,
        shadow_population: np.ndarray,
        training_fraction:float=0.1,
        online:bool=False
    ) -> list[int]:
        """Create and train shadow models based on the blueprint.

        Args:
        ----
            num_models (int): The number of shadow models to create.
            shadow_population (list): The indices in population eligible for training the shadow models.
            training_fraction (float): The fraction of the shadow population to use for training of a shadow model.
            online (bool): Whether the shadow models are created using an online or offline dataset.

        Returns:
        -------
            None

        """
        if num_models < 0:
            raise ValueError("Number of models cannot be negative")

        # Get the size of the dataset
        data_size = int(len(shadow_population)*training_fraction)
        all_indices, filtered_indices = self._filter(data_size, online)

        # Create a list of indices to use for the new shadow models
        n_existing_models = len(filtered_indices)

        if n_existing_models >= num_models:
            self.logger.info("Number of existing models exceeds or equals the number of models to create")
            return filtered_indices[:num_models]

        indices_to_use = []
        next_index = max(all_indices) + 1 if all_indices else 0
        while len(indices_to_use) < (num_models-n_existing_models):
            indices_to_use.append(next_index)
            next_index += 1

        for i in indices_to_use:
            # Get dataloader
            data_indices = np.random.choice(shadow_population, data_size, replace=False)
            data_loader = self.handler.get_dataloader(data_indices, self.batch_size)

            # Get shadow model blueprint
            model, criterion, optimizer = self._get_model_criterion_optimizer()

            # Train shadow model
            self.logger.info(f"Training shadow model {i} on {len(data_loader)* data_loader.batch_size} points")
            training_results = self.handler.train(data_loader, model, criterion, optimizer, self.epochs)
            # Read out results
            shadow_model = training_results["model"]
            train_acc = training_results["metrics"]["accuracy"]
            train_loss = training_results["metrics"]["loss"]

            self.logger.info(f"Training shadow model {i} complete")
            with open(f"{self.storage_path}/{self.model_storage_name}_{i}.pkl", "wb") as f:
                save(shadow_model.state_dict(), f)
                self.logger.info(f"Saved shadow model {i} to {self.storage_path}")

            self.logger.info(f"Storing metadata for shadow model {i}")
            meta_data = {}
            meta_data["init_params"] = self.init_params
            meta_data["train_indices"] = data_indices
            meta_data["num_train"] = len(data_indices)
            meta_data["optimizer"] = optimizer.__class__.__name__
            meta_data["criterion"] = criterion.__class__.__name__
            meta_data["batch_size"] = self.batch_size
            meta_data["epochs"] = self.epochs
            meta_data["train_acc"] = train_acc
            meta_data["train_loss"] = train_loss
            meta_data["online"] = online

            with open(f"{self.storage_path}/{self.metadata_storage_name}_{i}.pkl", "wb") as f:
                pickle.dump(meta_data, f)

            self.logger.info(f"Metadata for shadow model {i} stored in {self.storage_path}")
        return filtered_indices + indices_to_use

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

        model_path = f"{self.storage_path}/{self.model_storage_name}_{index}.pkl"
        shadow_model, criterion = self._load_model(model_path)
        return PytorchModel(shadow_model, criterion)

    def get_shadow_models(self:Self, num_models:list[int]) -> Tuple[list, list]:
        """Load the the shadow models."""
        shadow_models = []
        shadow_model_indices = []
        for i in num_models:
            self.logger.info(f"Loading shadow model {i}")
            model = self._load_shadow_model(i)
            shadow_models.append(model)
            shadow_model_indices.append(i)
        return shadow_models, shadow_model_indices

    def _load_shadow_metadata(self:Self, index:int) -> dict:
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
        metadata_path = f"{self.storage_path}/{self.metadata_storage_name}_{index}.pkl"
        return self._load_metadata(metadata_path)

    def get_shadow_model_metadata(self:Self, model_indices:list[int]) -> list:
        """Load the the shadow model metadata."""
        metadata = []
        if model_indices is int:
            model_indices = range(model_indices)
        for i in model_indices:
            self.logger.info(f"Loading metadata {i}")
            metadata.append(self._load_shadow_metadata(i))
        return metadata

    def get_in_indices_mask(self:Self, shadow_model_indices:list[int], dataset:np.ndarray) -> np.ndarray:
        """Get the mask indicating which indices in the dataset are present in the shadow model training set.

        Args:
        ----
            shadow_model_indices (list[int]): The number of shadow models.
            dataset (np.ndarray): The dataset.

        Returns:
        -------
            np.ndarray: The mask indicating which indices are present in the shadow model training set.

        """
        # Retrieve metadata for shadow models
        metadata = self.get_shadow_model_metadata(shadow_model_indices)

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
