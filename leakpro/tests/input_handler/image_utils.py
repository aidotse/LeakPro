"""Utility functions for the image handler tests."""
import os
import pickle

import numpy as np
import torch.nn.functional as F  # noqa: N812
from dotmap import DotMap
from torch import Tensor, flatten, long, nn, save, stack, tensor
from torch.nn import Module
from torch.utils.data import TensorDataset
from torchvision import transforms

from leakpro.import_helper import Self

from leakpro.tests.constants import STORAGE_PATH

parameters = DotMap()
parameters.epochs = 10
parameters.batch_size = 64
parameters.learning_rate = 0.001
parameters.optimizer = "sgd"
parameters.loss = "crossentropyloss"
parameters.data_points = 130
parameters.train_data_points = 20
parameters.test_data_points = 20
parameters.img_size = (3, 32, 32)
parameters.num_classes = 13
parameters.images_per_class = parameters.data_points // parameters.num_classes



class ConvNet(Module):
    """Convolutional Neural Network model."""

    def __init__(self:Self) -> None:
        """Initialize the ConvNet model."""
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, parameters.num_classes)

    def forward(self:Self, x:Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
        ----
            x (torch.Tensor): The input tensor.

        Returns:
        -------
            torch.Tensor: The output tensor.

        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = flatten(x, 1) # flatten all dimensions except batch
        return self.fc1(x)

class DatasetWithSubset(TensorDataset):
    """Dataset with a subset method."""

    def subset(self:Self, indices:np.ndarray) -> Self:
        """Extract a subset of the dataset based on the given indices."""

        subsets = [tensor[indices] for tensor in self.tensors]
        return DatasetWithSubset(*subsets)

def setup_image_test()->None:
    """Setup for the input handler test."""

    config = DotMap()

    # Set up the mock image dataset and add path to config
    dataset_path = create_mock_image_dataset()

    # ensure mock dataset is correct
    assert os.path.exists(dataset_path)

    with open(dataset_path, "rb") as data_file:
        dataset = pickle.load(data_file)  # noqa: S301

    # Ensure the dataset has the correct size and shape
    assert len(dataset) == parameters.data_points
    assert dataset.tensors[0][0].shape == parameters.img_size

    del dataset
    config.data_path = dataset_path

    # Set up the model and add path to config
    config.module_path = "./leakpro/tests/input_handler/image_utils.py"
    config.model_class = "ConvNet"

    model_path, metadata_path = create_mock_model_and_metadata()

    # ensure mock dataset is correct
    assert os.path.exists(model_path)
    assert os.path.exists(metadata_path)

    config.trained_model_path = model_path
    config.trained_model_metadata_path = metadata_path

    return config

def create_mock_image_dataset() -> str:
    """Creates a mock CIFAR-10 dataset with random images."""

    # Constants to create a mock image dataset same size as CIFAR10
    image_size = parameters.img_size  # CIFAR-10 image size
    num_classes = parameters.num_classes # CIFAR-10 has 10 classes
    images_per_class = parameters.images_per_class
    dataset_name = "image_handler_dataset.pkl"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    images = []
    labels = []

    # Generate random images and save them in the dataset
    for class_idx in range(num_classes):
        for _ in range(images_per_class):
            # Create a random image
            image_array = np.random.randint(0, 256, (image_size[1], image_size[2], image_size[0]), dtype=np.uint8)
            image_tensor = transform(image_array)
            images.append(image_tensor)
            labels.append(class_idx)

    # Stack the list of image tensors into a single tensor
    images_tensor = stack(images)
    # Convert labels list to a tensor
    labels_tensor = tensor(labels, dtype=long)
    dataset = DatasetWithSubset(images_tensor, labels_tensor)

    # Save the dataset to a .pkg file
    pkg_file_path = STORAGE_PATH + "/" + dataset_name
    with open(pkg_file_path, "wb") as pkg_file:
        pickle.dump(dataset, pkg_file)

    return pkg_file_path

def create_mock_model_and_metadata() -> str:
    """Creates a mock model and saves it to a file."""

    # Create a mock model
    model = ConvNet()
    model_path = STORAGE_PATH + "/mock_model.pkl"
    with open(model_path, "wb") as f:
        save(model.state_dict(), f)

    # Create metadata
    metadata = {
        "init_params": {},
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
    metadata_path = STORAGE_PATH + "/mock_metadata.pkl"

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    return model_path, metadata_path
