"""Module for handling shadow models."""

import os
import pickle
import re

import numpy as np
import torch
from torch import Tensor, jit, save
from torch.nn import Module

from leakpro.attacks.utils.model_handler import ModelHandler
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.schemas import ShadowModelTrainingSchema, TrainingOutput
from leakpro.signals.signal_extractor import PytorchModel
from leakpro.utils.import_helper import Self, Tuple
from leakpro.utils.logger import logger


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
            # Call init again to update states such as model path, target hash etc
            instances[cls].__init__(*args, **kwargs)
        return instances[cls]

    def is_created() -> bool:
        return cls in instances

    def delete_instance() -> None:
        """Delete the singleton instance (for resetting or testing purposes)."""
        if cls in instances:
            del instances[cls]

    # Attach methods to the class
    get_instance.is_created = is_created
    get_instance.delete_instance = delete_instance

    return get_instance

@singleton
class ShadowModelHandler(ModelHandler):
    """A class handling the creation, training, and loading of shadow models."""

    def __init__(self:Self, handler: MIAHandler) -> None:  # noqa: PLR0912
        """Initialize the ShadowModelHandler.

        Args:
        ----
            handler (MIAHandler): The input handler object.

        """
        caller = "shadow_model"
        super().__init__(handler, caller)

        # Set up the names of the shadow model
        self.model_storage_name = "shadow_model"
        self.metadata_storage_name = "metadata"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _filter(self:Self, data_size:int, online:bool)->list[int]:
        # Get the metadata for the shadow models
        entries = os.listdir(self.storage_path)
        pattern = re.compile(rf"^{self.metadata_storage_name}_\d+\.pkl$")
        files = [f for f in entries if pattern.match(f)]

        # Extract the index of the metadata
        all_indices = [int(re.search(r"\d+", f).group()) for f in files]

        # Setup checks
        # Get target model hash, this tells if the data has changed for the training
        target_model_hash = self.target_model_hash
        # create check list
        filter_checks = [data_size, online, self.model_class, target_model_hash]

        # Filter out indices to only keep the ones that passes the checks
        filtered_indices = []
        for i in all_indices:
            metadata = self._load_shadow_metadata(i)
            assert isinstance(metadata, ShadowModelTrainingSchema), "Shadow Model metadata is not of the correct type"
            meta_check_values = [metadata.num_train, metadata.online, metadata.model_class, metadata.target_model_hash]
            if all(a == b for a, b in zip(filter_checks, meta_check_values)):
                filtered_indices.append(i)

        return all_indices, filtered_indices

    def create_shadow_models(
        self:Self,
        num_models:int,
        shadow_population: list,
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
            logger.info("Number of existing models exceeds or equals the number of models to create")
            return filtered_indices[:num_models]

        indices_to_use = []
        next_index = max(all_indices) + 1 if all_indices else 0
        while len(indices_to_use) < (num_models-n_existing_models):
            indices_to_use.append(next_index)
            next_index += 1

        for i in indices_to_use:
            # Get dataloader
            data_indices = np.random.choice(shadow_population, data_size, replace=False)
            data_loader = self.handler.get_dataloader(data_indices, params=None)

            # Get shadow model blueprint
            model, criterion, optimizer = self._get_model_criterion_optimizer()

            # Train shadow model
            logger.info(f"Training shadow model {i} on {len(data_loader.dataset)} points")
            training_results = self.handler.train(data_loader, model, criterion, optimizer, self.epochs)

            # Read out results
            assert isinstance(training_results, TrainingOutput)
            shadow_model = training_results.model

            # Evaluate shadow model on remaining aux data
            remaining_indices = list(set(shadow_population) - set(data_indices))
            dataset_params = data_loader.dataset.return_params()
            test_loader = self.handler.get_dataloader(remaining_indices, params=dataset_params)
            test_result = self.handler.eval(test_loader, shadow_model, criterion, self.device)

            logger.info(f"Training shadow model {i} complete")
            shadow_model_state_dict = shadow_model.state_dict()
            cleaned_state_dict = {key.replace("_module.", "").replace("module.", ""): value
                    for key, value in shadow_model_state_dict.items()}

            with open(f"{self.storage_path}/{self.model_storage_name}_{i}.pkl", "wb") as f:
                save(cleaned_state_dict, f)
                logger.info(f"Saved shadow model {i} to {self.storage_path}")

            logger.info(f"Storing metadata for shadow model {i}")
            meta_data = ShadowModelTrainingSchema(
                init_params=self.init_params,
                train_indices = data_indices,
                num_train = len(data_indices),
                optimizer = optimizer.__class__.__name__,
                criterion = criterion.__class__.__name__,
                epochs = self.epochs,
                train_result = training_results.metrics,
                test_result = test_result,
                online = online,
                model_class = self.model_class,
                target_model_hash= self.target_model_hash
            )

            logger.info(f"Metadata for shadow model {i}:\n{meta_data}")
            with open(f"{self.storage_path}/{self.metadata_storage_name}_{i}.pkl", "wb") as f:
                pickle.dump(meta_data, f)

            logger.info(f"Metadata for shadow model {i} stored in {self.storage_path}")
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
            logger.info(f"Loading shadow model {i}")
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
            logger.info(f"Loading metadata {i}")
            metadata.append(self._load_shadow_metadata(i))
        return metadata

    def get_in_indices_mask(self:Self, shadow_model_indices:list[int], dataset_indices:np.ndarray) -> np.ndarray:
        """Get the mask indicating which indices in the dataset are present in the shadow model training set.

        Args:
        ----
            shadow_model_indices (list[int]): The number of shadow models.
            dataset_indices (np.ndarray): The dataset indices to consider.

        Returns:
        -------
            np.ndarray: The mask indicating which indices are present in the shadow model training set.

        """
        # Retrieve metadata for shadow models
        metadata = self.get_shadow_model_metadata(shadow_model_indices)

        # Extract training indices for each shadow model
        models_in_indices = [data.train_indices for data in metadata]

        # Convert to numpy array for easier manipulation
        models_in_indices = np.asarray(models_in_indices)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_indices_tensor = torch.from_numpy(models_in_indices).to(device=device)
        dataset_tensor = torch.from_numpy(dataset_indices).to(device=device)
        indice_masks_tensor = torch.zeros((len(dataset_indices), len(models_in_indices)), dtype=torch.bool, device=device)

        return _torch_indice_in_shadowmodel_training_set(indice_masks_tensor,\
                                                        dataset_tensor, model_indices_tensor).cpu().numpy()

@jit.script
def _torch_indice_in_shadowmodel_training_set(in_tensor:Tensor, dataset:Tensor, model_indices:Tensor) -> Tensor:
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
