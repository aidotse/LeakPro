import os
import pickle
from typing import List

import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.utils.data import Dataset
from torchvision import transforms


class GeneralDataset(Dataset):
    def __init__(self, data:np.ndarray, label:np.ndarray, transforms=None):
        """data_list: A list of GeneralData instances.
        """
        self.X = data # Convert to tensor and specify the data type
        self.y = label  # Assuming labels are for classification
        self.transforms = transforms

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        """Returns the data and label for a single instance indexed by idx.
        """
        if self.transforms:
            X = self.transforms(self.X[idx])
        else:
            X = self.X[idx]

        # ensure that X is a tensor
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        y = torch.tensor(self.y[idx], dtype=torch.long)
        return X, y

class TabularDataset(Dataset):
    """Tabular dataset."""

    def __init__(self, X, y):
        """Initializes instance of class TabularDataset.

        Args:
        ----
            X (str): features
            y (str): target

        """
        super().__init__(
            data_dict={"X": X, "y": y},
            default_input="X",
            default_output="y",
        )

    def __len__(self):
        return len(self.data_dict["y"])

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        X = np.float32(self.data_dict["X"][idx])
        y = np.float32(self.data_dict["y"][idx])
        return [X, y]


class InfiniteRepeatDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]



def get_dataset(dataset_name: str, data_dir: str):
    path = f"{data_dir}/{dataset_name}"

    if os.path.exists(f"{path}.pkl"):
        with open(f"{path}.pkl", "rb") as file:
            all_data = pickle.load(file)
        print(f"Load data from {path}.pkl")
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
        df = pd.concat([df_train, df_test], axis=0)
        df = df.replace(" ?", np.nan)
        df = df.dropna()
        X, y = df.iloc[:, :-1], df.iloc[:, -1]

        categorical_features = [col for col in X.columns if X[col].dtype == "object"]
        numerical_features = [
            col for col in X.columns if X[col].dtype in ["int64", "float64"]
        ]

        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_categorical = onehot_encoder.fit_transform(X[categorical_features])

        scaler = StandardScaler()
        X_numerical = scaler.fit_transform(X[numerical_features])

        X = np.hstack([X_numerical, X_categorical])

        # label encode the target variable to have the classes 0 and 1
        y = LabelEncoder().fit_transform(y)

        all_data = GeneralDataset(X,y)
        with open(f"{path}.pkl", "wb") as file:
            pickle.dump(all_data, file)
        print(f"Save data to {path}.pkl")
    elif "cifar10" in dataset_name:
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root="./data/cifar10", train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root="./data/cifar10", train=False,download=True, transform=transform)
        X = np.vstack([trainset.data, testset.data])
        y = np.hstack([trainset.targets, testset.targets])

        all_data = GeneralDataset(X, y, transform)

        with open(f"{path}.pkl", "wb") as file:
            pickle.dump(all_data, file)
        print(f"Save data to {path}.pkl")

    return all_data


def get_split(
    all_index: List[int], used_index: List[int], size: int, split_method: str
):
    """Select points based on the splitting methods

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


def prepare_train_test_datasets(dataset_size: int, configs: dict):
    """Prepare the dataset for training the target models when the training data are sampled uniformly from the distribution (pool of all possible data).

    Args:
    ----
        dataset_size (int): Size of the whole dataset
        num_datasets (int): Number of datasets we should generate
        configs (dict): Data split configuration

    Returns:
    -------
        dict: Data split information which saves the information of training points index and test points index for all target models.

    """
    # The index_list will save all the information about the train, test and auit for each target model.
    all_index = np.arange(dataset_size)
    train_size = int(configs["f_train"] * dataset_size)
    test_size = int(configs["f_test"] * dataset_size)

    selected_index = np.random.choice(all_index, train_size + test_size, replace=False)
    train_index, test_index = train_test_split(selected_index, test_size=test_size)
    dataset_train_test = {"train_indices": train_index, "test_indices": test_index}
    return dataset_train_test


def get_dataset_subset(dataset: Dataset, indices: List[int]):
    """Get a subset of the dataset.

    Args:
    ----
        dataset (torchvision.datasets): Whole dataset.
        index (list): List of index.

    """
    assert max(indices) < len(dataset) and min(indices) >= 0, "Index out of range"

    data = dataset.X
    targets = dataset.y
    transforms = dataset.transforms
    subset_data = [data[idx] for idx in indices]
    subset_targets = [targets[idx] for idx in indices]

    new_dataset = dataset.__class__(subset_data, subset_targets, transforms)

    return new_dataset


def get_dataloader(
    dataset: GeneralDataset,
    batch_size: int,
    loader_type="torch",
    shuffle: bool = True,
):
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
