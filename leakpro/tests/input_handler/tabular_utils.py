import os
import pickle

import numpy as np
import torch.nn.functional as F  # noqa: N812
import pandas as pd
import random
from dotmap import DotMap
from sklearn.preprocessing import OneHotEncoder

from torch import from_numpy, tensor, save
from torch.nn import Module
from torch.utils.data import TensorDataset

from leakpro.tests.constants import STORAGE_PATH, get_tabular_handler_config

class MLP(Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.init_params = {"input_size": input_size,
                            "hidden_size": hidden_size,
                            "num_classes": num_classes}
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class DatasetWithSubset(TensorDataset):
    """Dataset with a subset method."""

    def __init__(self, x:tensor, y:tensor, dec_to_onehot:dict, one_hot_encoded:bool=True):
        self.x = x
        self.y = y
        
        # create dictionary to map categorical columns to number of classes
        self.dec_to_onehot = dec_to_onehot
        self.one_hot_encoded = one_hot_encoded
        
    def subset(self, indices):
        return DatasetWithSubset(self.x[indices], 
                                 self.y[indices], 
                                 self.dec_to_onehot, 
                                 self.one_hot_encoded)
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def setup_tabular_test()->None:
    """Setup for the input handler test."""

    config = DotMap()
    parameters = get_tabular_handler_config()
    # Set up the mock image dataset and add path to config
    dataset_path = create_mock_tabular_dataset()

    # ensure mock dataset is correct
    assert os.path.exists(dataset_path)

    with open(dataset_path, "rb") as data_file:
        dataset = pickle.load(data_file)  # noqa: S301

    # Ensure the dataset has the correct size and shape
    assert len(dataset) == parameters.data_points

    del dataset
    config.data_path = dataset_path

    # Set up the model and add path to config
    config.module_path = "./leakpro/tests/input_handler/tabular_utils.py"
    config.model_class = "MLP"
    config.target_folder = parameters.target_folder

    model_path, metadata_path = create_mock_model_and_metadata()

    # ensure mock dataset is correct
    assert os.path.exists(model_path)
    assert os.path.exists(metadata_path)

    return config

def create_mock_tabular_dataset() -> str:
    """Creates a mock tabular dataset with random images."""
    parameters = get_tabular_handler_config()
    
    # Constants to create a mock tabular dataset
    n_points = parameters.data_points
    n_continuous = parameters.n_continuous
    n_categorical = parameters.n_categorical
    num_classes = parameters.num_classes
    dataset_name = "tabular_handler_dataset.pkl"

    continuous_data = np.random.randn(n_points, n_continuous)
    
    categorical_data = []
    for _ in range(n_categorical):
        classes = np.random.randint(0, 10)
        categorical_data.append([random.choice(range(classes)) for _ in range(n_points)])
    categorical_data = np.array(categorical_data).T  # Transpose to align rows with n_points
    
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    categorical_one_hot = one_hot_encoder.fit_transform(categorical_data)
    combined_data = np.hstack([continuous_data, categorical_one_hot])
    
    dec_to_onehot = {}
    for i in range(n_continuous):
        dec_to_onehot[i] = i # Continuous features are identity-mapped
    
    for i in range(n_continuous, combined_data.shape[1]):
        dec_to_onehot[i] = one_hot_encoder.categories_[i - n_continuous]

    one_hot_encoded = True
    
    data = from_numpy(data).float()
    label = from_numpy(np.random.randint(0, num_classes, n_points)).float()
    
    dataset = DatasetWithSubset(data, label, dec_to_onehot, one_hot_encoded)

    # Save the dataset to a .pkg file
    pkg_file_path = STORAGE_PATH + "/" + dataset_name
    with open(pkg_file_path, "wb") as pkg_file:
        pickle.dump(dataset, pkg_file)

    return pkg_file_path, data.shape[1]

def create_mock_model_and_metadata(input_size:int) -> str:
    """Creates a mock model and saves it to a file."""
    parameters = get_tabular_handler_config()
    # Create a mock model
    model = MLP()
    model_path = parameters.target_folder + "/target_model.pkl"
    with open(model_path, "wb") as f:
        save(model.state_dict(), f)

    # Create metadata
    metadata = {
        "init_params": {"input_size": input_size,
                        "hidden_size": 64,
                        "output_size": parameters.num_classes},
        "train_indices": np.arange(parameters.train_data_points).tolist(),
        "test_indices": np.arange(parameters.train_data_points,
                                  parameters.train_data_points + parameters.test_data_points).tolist(),
        "num_train": parameters.data_points,
        "optimizer": {
            "name": parameters.optimizer,
            "lr": parameters.learning_rate,
        },
        "loss": {"name": parameters.loss},
        "batch_size": parameters.batch_size,
        "epochs": parameters.epochs,
    }
    metadata_path = parameters.target_folder + "/model_metadata.pkl"

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    return model_path, metadata_path
