import torch
import pickle 
import os
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from ast import List

from itertools import product
from typing import Dict, Union

import numpy as np

########################################################################################################################
# DATASET CLASS
########################################################################################################################


class Dataset:
    """
    Wrapper around a dictionary-like formatted dataset, with functions to run preprocessing, to define default
    input/output features, and to split a dataset easily.
    """

    def __init__(
        self,
        data_dict: dict,
        default_input: str,
        default_output: str,
        default_group: str = None,
        preproc_fn_dict: dict = None,
        preprocessed: bool = False,
    ):
        """Constructor

        Args:
            data_dict: Contains the dataset as a dict.
            default_input: The key of the data_dict that should be used by default to get the input of a model.
            default_output: The key of the data_dict that should be used by default to get the expected output
                of a model.
            default_group: The key of the data_dict that shouuld be used by default to get the group of the data points.
                This is to contruct class dependent threshold.
            preproc_fn_dict: Contains optional preprocessing functions for each feature.
            preprocessed: Indicates if the preprocessing of preproc_fn_dict has already been applied.
        """

        # Store parameters
        self.data_dict = data_dict
        self.default_input = default_input
        self.default_output = default_output
        self.default_group = default_group
        self.preproc_fn_dict = preproc_fn_dict

        # Store splits names and features names
        self.splits = list(self.data_dict)
        self.features = list(self.data_dict[self.splits[0]])

        # If preprocessing functions were passed as parameters, execute them
        if not preprocessed and preproc_fn_dict is not None:
            self.preprocess()
    

    def preprocess(self):
        """
        Preprocessing function, executed by the constructor, based on the preproc_fn_dict attribute.
        """
        for (split, feature) in product(self.splits, self.features):
            if feature in list(self.preproc_fn_dict):
                fn = self.preproc_fn_dict[feature]
                self.data_dict[split][feature] = fn(self.data_dict[split][feature])

    def get_feature(self, split_name: str, feature_name: str, indices: list = None):
        """Returns a specific feature from samples of a specific split.

        Args:
            split_name: Name of the split.
            feature_name: Name of the feature.
            indices: Optional list of indices. If not specified, the entire subset is returned.

        Returns:
            The requested feature, from samples of the requested split.
        """

        # Two placeholders can be used to trigger either the default input or the default output, as specified during
        # object creation
        if feature_name == "<default_input>":
            feature_name = self.default_input
        elif feature_name == "<default_output>":
            feature_name = self.default_output
        elif feature_name == "<default_group>":
            feature_name = self.default_group

        # If 'indices' is not specified, returns the entire array. Else just return those indices
        if indices is None:
            return self.data_dict[split_name][feature_name]
        else:
            return self.data_dict[split_name][feature_name][indices]

    def subdivide(
        self,
        num_splits: int,
        split_names: list = None,
        method: str = "independent",
        split_size: Union[int, Dict[str, int]] = None,
        delete_original: bool = False,
        in_place: bool = True,
        return_results: bool = False,
    ):
        """Subdivides the splits contained in split_names into sub-splits, e.g. for shadow model training.

        Args:
            num_splits: Number of sub-splits per original split.
            split_names: The splits to subdivide (e.g. train and test). By default, includes all splits.
            method: Either independent or random. If method is independent, then the sub-splits are a partition of the
                original split (i.e. they contain the entire split without repetition). If method is random, then each
                sub-split is a random subset of the original split (i.e. some samples might be missing or repeated). If
                method is hybrid, then each sub-split is a random subset of the original split, with the guarantee that
                the 1st one is not overlapping with the others.
            split_size: If method is random, this is the size of one split (ignored if method is independent). Can
                either be an integer, or a dictionary of integer (one per split).
            delete_original: Indicates if the original split should be deleted.
            in_place: Indicates if the new splits should be included in the parent object or not
            return_results: Indicates if the new splits should be returned or not

        Returns:
            If in_place, a list of new Dataset objects, with the sub-splits. Otherwise, nothing, as the results are
            stored in self.data_dict.
        """

        # By default, includes all splits.
        if split_names is None:
            split_names = self.splits

        # List of results if in_place is False
        new_datasets_dict = [{} for _ in range(num_splits)]

        for split in split_names:

            if split_size is not None:
                parsed_split_size = (
                    split_size if isinstance(split_size, int) else split_size[split]
                )

            # If method is random, then each sub-split is a random subset of the original split.
            if method == "random":
                assert (
                    split_size is not None
                ), 'Argument split_size is required when method is "random" or "hybrid"'
                indices = np.random.randint(
                    self.data_dict[split][self.features[0]].shape[0],
                    size=(num_splits, parsed_split_size),
                )

            # If method is independent, then the sub-splits are a partition of the original split.
            elif method == "independent":
                indices = np.arange(self.data_dict[split][self.features[0]].shape[0])
                np.random.shuffle(indices)
                indices = np.array_split(indices, num_splits)

            # If method is hybrid, then each sub-split is a random subset of the original split, with the guarantee that
            # the 1st one is not overlapping with the others
            elif method == "hybrid":
                assert (
                    split_size is not None
                ), 'Argument split_size is required when method is "random" or "hybrid"'
                available_indices = np.arange(
                    self.data_dict[split][self.features[0]].shape[0]
                )
                indices_a = np.random.choice(
                    available_indices, size=(1, parsed_split_size), replace=False
                )
                available_indices = np.setdiff1d(available_indices, indices_a.flatten())
                indices_b = np.random.choice(
                    available_indices,
                    size=(num_splits - 1, parsed_split_size),
                    replace=True,
                )
                indices = np.concatenate((indices_a, indices_b))

            else:
                raise ValueError(f'Split method "{method}" does not exist.')

            for split_n in range(num_splits):
                # Fill the dictionary if in_place is True
                if in_place:
                    self.data_dict[f"{split}{split_n:03d}"] = {}
                    for feature in self.features:
                        self.data_dict[f"{split}{split_n:03d}"][
                            feature
                        ] = self.data_dict[split][feature][indices[split_n]]
                # Create new dictionaries if return_results is True
                if return_results:
                    new_datasets_dict[split_n][f"{split}"] = {}
                    for feature in self.features:
                        new_datasets_dict[split_n][f"{split}"][
                            feature
                        ] = self.data_dict[split][feature][indices[split_n]]

            # delete_original indicates if the original split should be deleted.
            if delete_original:
                del self.data_dict[split]

        # Update the list of splits
        self.splits = list(self.data_dict)

        # Return new datasets if return_results is True
        if return_results:
            return [
                Dataset(
                    data_dict=new_datasets_dict[i],
                    default_input=self.default_input,
                    default_output=self.default_output,
                    default_group=self.default_group,
                    preproc_fn_dict=self.preproc_fn_dict,
                    preprocessed=True,
                )
                for i in range(num_splits)
            ]

    def __str__(self):
        """
        Returns a string describing the dataset.
        """
        txt = [
            f'{" DATASET OBJECT ":=^48}',
            f"Splits            = {self.splits}",
            f"Features          = {self.features}",
            f"Default features  = {self.default_input} --> {self.default_output}",
            "=" * 48,
        ]
        return "\n".join(txt)

class TabularDataset(Dataset):
    """Tabular dataset."""

    def __init__(self, X, y):
        """Initializes instance of class TabularDataset.
        Args:
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
    elif os.path.exists(f"{path}/{dataset_name}.data"):
        column_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                "hours-per-week", "native-country", "income"]
        df_train = pd.read_csv(f"{path}/{dataset_name}.data", names=column_names)
        df_test = pd.read_csv(f"{path}/{dataset_name}.test", names=column_names, header=0)
        df_test['income'] = df_test['income'].str.replace('.', '', regex=False)
        df = pd.concat([df_train, df_test], axis=0)
        df = df.replace(' ?', np.nan)
        df = df.dropna()
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        
        categorical_features = [col for col in X.columns if X[col].dtype == 'object']
        numerical_features = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
        
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_categorical = onehot_encoder.fit_transform(X[categorical_features])
        
        scaler = StandardScaler()
        X_numerical = scaler.fit_transform(X[numerical_features])

        X = np.hstack([X_numerical, X_categorical])

        # label encode the target variable to have the classes 0 and 1
        y = LabelEncoder().fit_transform(y)
        

        all_data = TabularDataset(X, y)
        with open(f"{path}.pkl", "wb") as file:
            pickle.dump(all_data, file)
        print(f"Save data to {path}.pkl")
    else:
        raise NotImplementedError(f"{dataset_name} is not implemented")

    print(f"the whole dataset size: {len(all_data)}")
    return all_data

def get_split(all_index: List(int), used_index: List(int), size: int, split_method: str):
    """Select points based on the splitting methods

    Args:
        all_index (list): All the possible dataset index list
        used_index (list): Index list of used points
        size (int): Size of the points needs to be selected
        split_method (str): Splitting (selection) method

    Raises:
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
        dataset_size (int): Size of the whole dataset
        num_datasets (int): Number of datasets we should generate
        configs (dict): Data split configuration

    Returns:
        dict: Data split information which saves the information of training points index and test points index for all target models.
    """

    # The index_list will save all the information about the train, test and auit for each target model.
    index_list = []
    all_index = np.arange(dataset_size)
    train_size = int(configs["f_train"] * dataset_size)
    test_size = int(configs["f_test"] * dataset_size)
    
    selected_index = np.random.choice(all_index, train_size + test_size, replace=False)
    train_index, test_index = train_test_split(selected_index, test_size=test_size)
    dataset_train_test = {"train_indices": train_index, "test_indices": test_index}
    return dataset_train_test

def get_dataset_subset(dataset: Dataset, indices: List(int)):
    """Get a subset of the dataset.

    Args:
        dataset (torchvision.datasets): Whole dataset.
        index (list): List of index.
    """
    assert max(indices) < len(dataset) and min(indices) >= 0, "Index out of range"
    
    # Initialize new dataset (this might need to be adjusted based on the specific dataset class)
    data = dataset.data_dict["X"]
    targets = dataset.data_dict["y"]
    subset_data = [data[idx] for idx in indices]
    subset_targets = [targets[idx] for idx in indices]
    
    new_dataset = dataset.__class__(subset_data, subset_targets)
    
    return new_dataset

    
    
def get_dataloader(
    dataset: TabularDataset,
    batch_size: int,
    loader_type="torch",
    shuffle: bool = True,
):
    if loader_type == "torch":
        repeated_data = InfiniteRepeatDataset(dataset)
        return torch.utils.data.DataLoader(
            repeated_data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=16,
        )
