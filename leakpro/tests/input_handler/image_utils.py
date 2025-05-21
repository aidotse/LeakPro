"""Utility functions for the image handler tests."""
import os
import pickle

import numpy as np
import torch.nn.functional as F  # noqa: N812
from dotmap import DotMap
from torch import Tensor, flatten, long, nn, save, stack, tensor, optim
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision import transforms

from leakpro.tests.constants import STORAGE_PATH, get_image_handler_config
from leakpro.utils.import_helper import Self
from leakpro.schemas import MIAMetaDataSchema, EvalOutput, TrainingOutput
from leakpro.tests.input_handler.image_input_handler import ImageInputHandler
from leakpro import LeakPro

class ConvNet(Module):
    """Convolutional Neural Network model."""

    def __init__(self:Self) -> None:
        """Initialize the ConvNet model."""
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, get_image_handler_config().num_classes)

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



def setup_image_test()->None:
    """Setup for the input handler test."""

    config = DotMap()
    parameters = get_image_handler_config()
    # Set up the mock image dataset and add path to config
    dataset_path = create_mock_image_dataset()

    # ensure mock dataset is correct
    assert os.path.exists(dataset_path)

    with open(dataset_path, "rb") as data_file:
        dataset = pickle.load(data_file)  # noqa: S301

    # Ensure the dataset has the correct size and shape
    assert len(dataset) == parameters.data_points
    assert dataset[0][0].shape == parameters.img_size

    config.data_path = dataset_path

    # Set up the model and add path to config
    config.module_path = "./leakpro/tests/input_handler/image_utils.py"
    config.model_class = "ConvNet"
    config.target_folder = parameters.target_folder

    model_path, metadata_path = create_mock_model_and_metadata(dataset)

    # ensure mock dataset is correct
    assert os.path.exists(model_path)
    assert os.path.exists(metadata_path)

    return config

def create_mock_image_dataset() -> str:
    """Creates a mock CIFAR-10 dataset with random images."""
    parameters = get_image_handler_config()
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
    dataset = ImageInputHandler.UserDataset(images_tensor, labels_tensor)

    # Save the dataset to a .pkg file
    pkg_file_path = STORAGE_PATH + "/" + dataset_name
    with open(pkg_file_path, "wb") as pkg_file:
        pickle.dump(dataset, pkg_file)

    return pkg_file_path

def create_mock_model_and_metadata(dataset) -> str:
    """Creates a mock model and saves it to a file."""
    parameters = get_image_handler_config()

    if not os.path.exists(parameters.target_folder):
        os.makedirs(parameters.target_folder)

    # Create a mock model
    model = ConvNet()
    model_path = parameters.target_folder + "/target_model.pkl"
    with open(model_path, "wb") as f:
        save(model.state_dict(), f)

    # Create metadata
    train_result = TrainingOutput(model= model, metrics = EvalOutput(accuracy=0.9, loss=0.1))
    test_result = EvalOutput(accuracy=0.8, loss=0.2)
    optimizer = optim.SGD(model.parameters(), lr=parameters.learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(dataset, batch_size=parameters.batch_size, shuffle=False)
    train_indices = np.arange(parameters.train_data_points).tolist()
    test_indices = np.arange(parameters.train_data_points,parameters.train_data_points + parameters.test_data_points).tolist()
    dataset_name = "CIFAR-10"
    meta_data = LeakPro.make_mia_metadata(train_result=train_result,
                                      optimizer=optimizer,
                                      loss_fn=criterion,
                                      dataloader=train_loader,
                                      test_result=test_result,
                                      epochs=parameters.epochs,
                                      train_indices=train_indices,
                                      test_indices=test_indices,
                                      dataset_name=dataset_name)
    
    assert isinstance(meta_data, MIAMetaDataSchema)
    
    metadata_path = parameters.target_folder + "/model_metadata.pkl"

    with open(metadata_path, "wb") as f:
        pickle.dump(meta_data, f)

    return model_path, metadata_path
