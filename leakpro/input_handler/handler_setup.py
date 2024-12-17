"""Parent class for user inputs."""

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from leakpro.input_handler.user_imports import get_class_from_module, import_module_from_file
from leakpro.utils.import_helper import Any, Self, Tuple
from leakpro.utils.logger import logger


# must be called after initialization
def setup(self:Self) -> None:
    """Set up the code handler by retrieving the model class, target metadata, trained target model, and population."""
    self._load_model_class()
    self._load_target_metadata()
    self._load_trained_target_model()
    self._load_population()

def _load_population(self:Self) -> None:
    """Default implementation of the population loading."""
    try:
        with open(self.configs["target"]["data_path"], "rb") as file:

            self.population = joblib.load(file)
            self.population_size = len(self.population)

            if not _is_indexable(self.population):
                raise ValueError("Population dataset is not indexable.")
            logger.info(f"Loaded population dataset from {self.configs['target']['data_path']}")
        logger.info(f"Loaded population dataset from {self.configs['target']['data_path']}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find the population dataset at {self.configs['target']['data_path']}") from e

def _load_model_class(self:Self) -> None:
    """Get the model class blueprint from the target module."""
    model_class=self.configs["target"].get("model_class", None)
    if model_class is None:
        raise ValueError("model_class not found in configs.")

    module_path=self.configs["target"].get("module_path", None)
    if module_path is None:
        raise ValueError("module_path not found in configs.")

    try:
        target_module = import_module_from_file(module_path)
        self.target_model_blueprint = get_class_from_module(target_module, model_class)
        logger.info(f"Target model blueprint created from {model_class} in {module_path}.")
    except Exception as e:
        raise ValueError(f"Failed to create the target model blueprint from {model_class} in {module_path}") from e

def _validate_target_metadata(self:Self) -> None:
    """Validate the target model metadata."""
    if "train_indices" not in self.target_model_metadata:
        raise ValueError("train_indices not found in target model metadata.")

    if "test_indices" not in self.target_model_metadata:
        raise ValueError("test_indices not found in target model metadata.")

def _load_target_metadata(self:Self) -> None:
    """Get the target model metadata from the trained model metadata file."""
    target_model_metadata_path = self.configs["target"].get("target_folder", None)
    if target_model_metadata_path is None:
        raise ValueError("Trained model metadata path not found in configs.")
    try:
        self.target_model_metadata_path = f"{target_model_metadata_path}/model_metadata.pkl"
        with open(self.target_model_metadata_path, "rb") as f:
            self.target_model_metadata = joblib.load(f)
            self._validate_target_metadata()

            self.train_indices = self.target_model_metadata["train_indices"]
            self.test_indices = self.target_model_metadata["test_indices"]

            if len(self.train_indices) == 0:
                raise ValueError("Train indices are empty.")
            if len(self.test_indices) == 0:
                raise ValueError("Test indices are empty.")

        logger.info(f"Loaded target model metadata from {self.target_model_metadata_path}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find the target model metadata at {self.target_model_metadata_path}") from e

def _load_trained_target_model(self:Self) -> None:
    """Get the trained target model."""
    model_path = self.configs["target"].get("target_folder", None)
    if model_path is None:
        raise ValueError("Trained model path not found in configs.")
    self.model_path = f"{model_path}/target_model.pkl"
    init_params = self.target_model_metadata.get("init_params", {})
    try:
        with open(self.model_path, "rb") as f:
            self.target_model = self.target_model_blueprint(**init_params)
            self.target_model.load_state_dict(torch.load(f))
        logger.info(f"Loaded target model from {model_path}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find the trained target model at {model_path}") from e

#------------------------------------------------
# Methods related to population dataset
#------------------------------------------------
def _validate_indices(self:Self, dataset_indices: np.ndarray) -> None:
    if self.population is None:
        raise ValueError("Population dataset is not loaded.")

    if len(dataset_indices) == 0:
        raise ValueError("Dataset indices are empty.")

    if len(dataset_indices) > self.population_size:
        raise ValueError("Dataset indices are greater than the population size.")

    if len(dataset_indices) != len(np.unique(dataset_indices)):
        raise ValueError("Dataset indices contain duplicates.")

    if not np.all(dataset_indices < len(self.population)):
        raise ValueError("Dataset indices contain values greater than the population size.")

    if not np.all(dataset_indices >= 0):
        raise ValueError("Dataset indices contain negative values.")

    if not np.all(np.isfinite(dataset_indices)):
        raise ValueError("Dataset indices contain non-finite values.")

    if not np.issubdtype(dataset_indices.dtype, np.integer):
        raise ValueError("Dataset indices are not integers.")

def _is_indexable(obj:Any) -> bool:
    """Check if an object is indexable using NumPy array indexing."""

    # Check for common indexable types
    if hasattr(obj, "__getitem__"):
        return True
    raise ValueError("Object is not indexable.")

def get_dataset(self:Self, dataset_indices: np.ndarray) -> np.ndarray:
    """Get the dataset from the population."""

    if isinstance(dataset_indices, np.ndarray) is False:
        dataset_indices = np.array(dataset_indices, ndmin=1)

    self._validate_indices(dataset_indices)

    return self.population.subset(dataset_indices)

def get_dataloader(self: Self, dataset_indices: np.ndarray, batch_size: int = 32) -> DataLoader:
    """Default implementation of the dataloader."""
    dataset = self.get_dataset(dataset_indices)
    collate_fn = self.population.collate_fn if hasattr(self.population, "collate_fn") else None
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

def get_labels(self:Self, dataset_indices: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """Get the labels for given indices in the population."""
    dataset = self.get_dataset(dataset_indices)
    collate_fn = self.population.collate_fn if hasattr(self.population, "collate_fn") else None
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize an empty list to store the labels
    all_labels = []

    # Iterate over the DataLoader to extract the labels
    for _, labels in dataloader:
        all_labels.append(labels.numpy())  # Convert labels from tensors to NumPy arrays

    return np.concatenate(all_labels)

#------------------------------------------------
# Methods related to target model
#------------------------------------------------
def get_target_replica(self:Self) -> Tuple[torch.nn.Module, nn.modules.loss._Loss, torch.optim.Optimizer]:
    """Get an instance of a model created from the target model."""
    init_params = self.target_model_metadata.get("init_params", {})
    try:
        model_replica = self.target_model_blueprint(**init_params)
        return model_replica, self.get_criterion(), self.get_optimizer(model_replica)
    except Exception as e:
        raise ValueError("Failed to create an instance of the target model.") from e

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

def set_population_size(self:Self, size:int) -> None:
    """Set the size of the population."""
    self._population_size = size

def get_population_size(self:Self) -> int:
    """Get the size of the population."""
    return self._population_size

def set_train_indices(self:Self, indices:np.ndarray) -> None:
    """Set the training indices of the target model."""
    self._train_indices = indices

def get_train_indices(self:Self) -> np.ndarray:
    """Get the training indices of the target model."""
    return self._train_indices

def set_test_indices(self:Self, indices:np.ndarray) -> None:
    """Set the testing indices of the target model."""
    self._test_indices = indices

def get_test_indices(self:Self) -> np.ndarray:
    """Get the testing indices of the target model."""
    return self._test_indices
