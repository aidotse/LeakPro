import os
import pickle
import random

import numpy as np
from dotmap import DotMap
from sklearn.preprocessing import OneHotEncoder
from torch import from_numpy, save, optim, nn
from torch.utils.data import DataLoader
from torch.nn import Linear, Module, ReLU

from leakpro.tests.constants import STORAGE_PATH, get_tabular_handler_config
from leakpro.schemas import MIAMetaDataSchema, EvalOutput, TrainingOutput
from leakpro.tests.input_handler.tabular_input_handler import TabularInputHandler
from leakpro import LeakPro

class MLP(Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc1 = Linear(input_size, hidden_size)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def setup_tabular_test()->None:
    """Setup for the input handler test."""

    config = DotMap()
    parameters = get_tabular_handler_config()
    # Set up the mock image dataset and add path to config
    dataset_path, one_hot_encoded_cols = create_mock_tabular_dataset()

    # ensure mock dataset is correct
    assert os.path.exists(dataset_path)

    with open(dataset_path, "rb") as data_file:
        dataset = pickle.load(data_file)  # noqa: S301

    # Ensure the dataset has the correct size and shape
    assert len(dataset) == parameters.data_points

    config.data_path = dataset_path

    # Set up the model and add path to config
    config.module_path = "./leakpro/tests/input_handler/tabular_utils.py"
    config.model_class = "MLP"
    config.target_folder = parameters.target_folder

    model_path, metadata_path = create_mock_model_and_metadata(input_size=one_hot_encoded_cols, dataset=dataset)

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
        classes = np.random.randint(2, 10)
        categorical_data.append([random.choice(range(classes)) for _ in range(n_points)])
    categorical_data = np.array(categorical_data).T  # Transpose to align rows with n_points

    one_hot_encoder = OneHotEncoder(sparse_output=False)
    categorical_one_hot = one_hot_encoder.fit_transform(categorical_data)
    combined_data = np.hstack([continuous_data, categorical_one_hot])

    dec_to_onehot = {}
    for i in range(n_continuous):
        dec_to_onehot[i] = [i] # Continuous features are identity-mapped

    n_cols = n_continuous
    for i in range(n_continuous, n_continuous + n_categorical):
        dec_to_onehot[i] = list(one_hot_encoder.categories_[i - n_continuous] + n_cols)
        n_cols += len(dec_to_onehot[i])

    one_hot_encoded = True

    data = from_numpy(combined_data).float()
    label = from_numpy(np.random.randint(0, num_classes, n_points)).float()

    params = {"dec_to_onehot": dec_to_onehot, "one_hot_encoded": one_hot_encoded}
    dataset = TabularInputHandler.UserDataset(data, label, **params)

    # Save the dataset to a .pkg file
    pkg_file_path = STORAGE_PATH + "/" + dataset_name
    with open(pkg_file_path, "wb") as pkg_file:
        pickle.dump(dataset, pkg_file)

    return pkg_file_path, data.shape[1]

def create_mock_model_and_metadata(input_size:int, dataset) -> str:
    """Creates a mock model and saves it to a file."""
    parameters = get_tabular_handler_config()

    if not os.path.exists(parameters.target_folder):
        os.makedirs(parameters.target_folder)

    # Create a mock model
    model = MLP(input_size=input_size, hidden_size=64, num_classes=parameters.num_classes)
    model_path = parameters.target_folder + "/target_model.pkl"
    with open(model_path, "wb") as f:
        save(model.state_dict(), f)

    # Create metadata
    train_result = TrainingOutput(model= model, metrics = EvalOutput(accuracy=0.9, loss=0.1))
    test_result = EvalOutput(accuracy=0.8, loss=0.2)
    optimizer = optim.SGD(model.parameters(), lr=parameters.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(dataset, batch_size=parameters.batch_size, shuffle=False)
    train_indices = np.arange(parameters.train_data_points).tolist()
    test_indices = np.arange(parameters.train_data_points,parameters.train_data_points + parameters.test_data_points).tolist()
    dataset_name = "Tabular"
    meta_data = LeakPro.make_mia_metadata(train_result=train_result,
                                      optimizer=optimizer,
                                      loss_fn=criterion,
                                      dataloader=train_loader,
                                      test_result=test_result,
                                      epochs=parameters.epochs,
                                      train_indices=train_indices,
                                      test_indices=test_indices,
                                      dataset_name=dataset_name)
    
    metadata_path = parameters.target_folder + "/model_metadata.pkl"

    with open(metadata_path, "wb") as f:
        pickle.dump(meta_data, f)

    return model_path, metadata_path
