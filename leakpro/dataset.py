"""Module that contains the dataset class and functions for preparing the dataset for training the target models."""

import numpy as np
import torch
from torch.utils.data import Dataset

from leakpro.import_helper import List, Self


class GeneralDataset(Dataset):
    """Immutable dataset class for general data."""

    def __init__(
        self:Self,
        data:np.ndarray,
        labels:np.ndarray,
        transforms:torch.nn.Module=None,
        task_type:str="classification"
    ) -> None:
        """data_list: A list of GeneralData instances."""
        assert isinstance(data, np.ndarray), "data must be a numpy array"
        assert isinstance(labels, np.ndarray), "labels must be a numpy array"

        self._data = data
        self._labels = labels
        self._task_type = task_type
        self._transforms = transforms

    def __len__(self:Self) -> int:
        """Return the length of the dataset."""
        return len(self._labels)

    def __getitem__(self:Self, idx:int) -> List[torch.Tensor]:
        """Return the data and label for a single instance indexed by idx."""
        x = self._transforms(self._data[idx]) if self._transforms else self._data[idx]

        # ensure that x is a tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        if self._task_type == "classification":
            y = torch.tensor(self._labels[idx], dtype=torch.long)  # Use torch.long for classification labels
        elif self._task_type == "regression":
            y = torch.tensor(self._labels[idx], dtype=torch.float32)  # Use torch.float32 for regression labels

        return x, y

    def subset(self:Self, indices:np.ndarray) -> Self:
        """Create a subset of the current dataset."""
        # As the data is immutable, just pass the indices
        if isinstance(indices, list):
            indices = np.array(indices)

        if np.max(indices) >= len(self._data) or np.min(indices) < 0:
            raise ValueError("Index out of range")
        return GeneralDataset(self._data[indices], self._labels[indices], self._transforms, self._task_type)

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
