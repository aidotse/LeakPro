"""Module that contains the dataset class and functions for preparing the dataset for training the target models."""

import logging
import os
import pickle

import joblib
import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.utils.data import Dataset
from torchvision import transforms

from leakpro.import_helper import List, Self


class GeneralDataset(Dataset):
    """Dataset class for general data."""

    def __init__(self:Self, data:np.ndarray, label:np.ndarray, transforms:torch.nn.Module=None) -> None:
        """data_list: A list of GeneralData instances."""
        self.x = data # Convert to tensor and specify the data type
        self.y = label  # Assuming labels are for classification
        self.transforms = transforms

    def __len__(self:Self) -> int:
        """Return the length of the dataset."""
        return len(self.y)

    def __getitem__(self:Self, idx:int) -> List[torch.Tensor]:
        """Return the data and label for a single instance indexed by idx."""
        x = self.transforms(self.x[idx]) if self.transforms else self.x[idx]

        # ensure that x is a tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

class InfiniteRepeatDataset(GeneralDataset):
    """Dataset class for infinite repeat data."""

    def __init__(self:Self, x:np.ndarray, y:np.ndarray, transform:torch.nn.Module=None) -> None:
        """Initialize the InfiniteRepeatDataset class.

        Args:
        ----
            x (np.ndarray): The input data.
            y (np.ndarray): The target labels.
            transform (torch.nn.Module, optional): The data transformation module. Defaults to None.

        """
        super().__init__(x, y, transform)

    def __len__(self:Self) -> int:
        """Return the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self:Self, idx:int) -> List[torch.Tensor]:
        """Return the data and label for a single instance indexed by idx."""
        return self.x[idx % len(self.dataset)], self.y[idx % len(self.dataset)]



def get_dataset(dataset_name: str, data_dir: str, logger:logging.Logger) -> GeneralDataset:
    """Get the dataset."""
    path = f"{data_dir}/{dataset_name}"

    if os.path.exists(f"{path}.pkl"):
        with open(f"{path}.pkl", "rb") as file:
            all_data = joblib.load(file)
        logger.info(f"Load data from {path}.pkl")
    elif "adult" in dataset_name:
        column_names = [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "income",
        ]
        df_train = pd.read_csv(f"{path}/{dataset_name}.data", names=column_names)
        df_test = pd.read_csv(
            f"{path}/{dataset_name}.test", names=column_names, header=0
        )
        df_test["income"] = df_test["income"].str.replace(".", "", regex=False)
        df_concatenated = pd.concat([df_train, df_test], axis=0)
        df_replaced = df_concatenated.replace(" ?", np.nan)
        df_clean = df_replaced.dropna()
        x, y = df_clean.iloc[:, :-1], df_clean.iloc[:, -1]

        categorical_features = [col for col in x.columns if x[col].dtype == "object"]
        numerical_features = [
            col for col in x.columns if x[col].dtype in ["int64", "float64"]
        ]

        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        x_categorical = onehot_encoder.fit_transform(x[categorical_features])

        scaler = StandardScaler()
        x_numerical = scaler.fit_transform(x[numerical_features])

        x = np.hstack([x_numerical, x_categorical])

        # label encode the target variable to have the classes 0 and 1
        y = LabelEncoder().fit_transform(y)

        all_data = GeneralDataset(x,y)
        with open(f"{path}.pkl", "wb") as file:
            pickle.dump(all_data, file)
        logger.info(f"Save data to {path}.pkl")
    elif "cifar10" in dataset_name:
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root="./data/cifar10", train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root="./data/cifar10", train=False,download=True, transform=transform)
        x = np.vstack([trainset.data, testset.data])
        y = np.hstack([trainset.targets, testset.targets])

        all_data = GeneralDataset(x, y, transform)

        with open(f"{path}.pkl", "wb") as file:
            pickle.dump(all_data, file)
        logger.info(f"Save data to {path}.pkl")

    return all_data


def get_split(
    all_index: List[int], used_index: List[int], size: int, split_method: str
) -> np.ndarray:
    """Select points based on the splitting methods.

    Args:
    ----
        all_index (list): All the possible dataset index list
        used_index (list): Index list of used points
        size (int): Size of the points needs to be selected
        split_method (str): Splitting (selection) method

    Raises:
    ------
        NotImplementedError: If the splitting the methods isn't implemented
        ValueError: If there aren't enough points to select
    Returns:
        np.ndarray: List of index

    """
    if split_method in "no_overlapping":
        selected_index = np.setdiff1d(all_index, used_index, assume_unique=True)
        if size <= len(selected_index):
            selected_index = np.random.choice(selected_index, size, replace=False)
        else:
            raise ValueError("Not enough remaining data points.")
    elif split_method == "uniform":
        if size <= len(all_index):
            selected_index = np.random.choice(all_index, size, replace=False)
        else:
            raise ValueError("Not enough remaining data points.")
    else:
        raise NotImplementedError(
            f"{split_method} is not implemented. The only supported methods are uniform and no_overlapping."
        )

    return selected_index


def prepare_train_test_datasets(dataset_size: int, configs: dict) -> dict:
    """Prepare the dataset for training the target models when the training data are sampled uniformly from the population.

    Args:
    ----
        dataset_size (int): Size of the whole dataset
        num_datasets (int): Number of datasets we should generate
        configs (dict): Data split configuration

    Returns:
    -------
        dict: Data split information which saves the information of training points index and test points index.

    """
    
    # The index_list will save all the information about the train, test and auit for each target model.
    all_index = np.arange(dataset_size)
    train_size = int(configs["f_train"] * dataset_size)
    test_size = int(configs["f_test"] * dataset_size)

    selected_index = np.random.choice(all_index, train_size + test_size, replace=False)
    train_index, test_index = train_test_split(selected_index, test_size=test_size)
    return {"train_indices": train_index, "test_indices": test_index}


def get_dataset_subset(dataset: Dataset, indices: List[int]) -> Dataset:
    """Get a subset of the dataset.

    Args:
    ----
        dataset (torchvision.datasets): Whole dataset.
        indices (list): List of indices.

    """
    if max(indices) >= len(dataset) or min(indices) < 0:
        raise ValueError("Index out of range")

    data = dataset.x
    targets = dataset.y
    transforms = dataset.transforms
    subset_data = [data[idx] for idx in indices]
    subset_targets = [targets[idx] for idx in indices]

    return dataset.__class__(subset_data, subset_targets, transforms)



def get_dataloader(
    dataset: GeneralDataset,
    batch_size: int,
    loader_type: str = "torch",
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """Get a data loader for the given dataset.

    Args:
    ----
        dataset (GeneralDataset): The dataset to load.
        batch_size (int): The batch size.
        loader_type (str, optional): The type of data loader. Defaults to "torch".
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

    Returns:
    -------
        torch.utils.data.DataLoader: The data loader.

    """
    if loader_type == "torch":
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=16,
        )
    return None
