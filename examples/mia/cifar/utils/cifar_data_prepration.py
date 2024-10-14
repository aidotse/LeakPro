import os
import numpy as np
import pandas as pd
import joblib
import pickle
from sklearn.model_selection import train_test_split
from torchvision import transforms, datasets
from torchvision.datasets import CIFAR10, CIFAR100
import urllib.request
from torch.utils.data import Dataset, Subset, DataLoader
from torch import tensor, float32, cat



class CifarDataset(Dataset):
    def __init__(self, x, y, transform=None,  indices=None):
        """
        Custom dataset for CIFAR data.

        Args:
            x (torch.Tensor): Tensor of input images.
            y (torch.Tensor): Tensor of labels.
            transform (callable, optional): Optional transform to be applied on the image tensors.
        """
        self.x = x
        self.y = y
        self.transform = transform  
        self.indices = indices

    def __len__(self):
        """Return the total number of samples."""
        return len(self.y)

    def __getitem__(self, idx):
        """Retrieve the image and its corresponding label at index 'idx'."""
        image = self.x[idx]
        label = self.y[idx]

        # Apply transformations to the image if any
        if self.transform:
            image = self.transform(image)

        return image, label
    
    @classmethod
    def from_cifar10(cls, root="./data", download=True, transform=None):
        # Load the CIFAR10 train and test datasets
        trainset = CIFAR10(root=root, train=True, download=download, transform=transforms.ToTensor())
        testset = CIFAR10(root=root, train=False, download=download, transform=transforms.ToTensor())

        # Concatenate both datasets' data and labels
        data = cat([tensor(trainset.data, dtype=float32), 
                          tensor(testset.data, dtype=float32)], 
                          dim=0)
        # Rescale data from [0, 255] to [0, 1]
        data /= 255.0
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        data = data.permute(0, 3, 1, 2)
        data = normalize(data)
        
        targets = cat([tensor(trainset.targets), tensor(testset.targets)], dim=0)

        return cls(data, targets)
    
    @classmethod
    def from_cifar100(cls, root="./data", download=True, transform=None):
        # Load the CIFAR10 train and test datasets
        trainset = CIFAR100(root=root, train=True, download=download, transform=transforms.ToTensor())
        testset = CIFAR100(root=root, train=False, download=download, transform=transforms.ToTensor())

        # Concatenate both datasets' data and labels
        data = cat([tensor(trainset.data, dtype=float32), 
                          tensor(testset.data, dtype=float32)], 
                          dim=0)
        # Rescale data from [0, 255] to [0, 1]
        data /= 255.0
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        data = data.permute(0, 3, 1, 2)
        data = normalize(data)
        
        targets = cat([tensor(trainset.targets), tensor(testset.targets)], dim=0)

        return cls(data, targets)

    def subset(self, indices):
        """Return a subset of the dataset based on the given indices."""
        return CifarDataset(self.x[indices], self.y[indices], transform=self.transform)


def get_cifar10_dataset(data_path):
    # Create the combined CIFAR-10 dataset

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    population = CifarDataset.from_cifar10(root=data_path, download=True, transform=transform)

    file_path = data_path + "cifar10.pkl"
    if not os.path.exists(file_path):
        with open(file_path, "wb") as file:
            pickle.dump(population, file)
            print(f"Save data to {file_path}.pkl")

    # Create a subset of the dataset (first 1000 samples)   
    pretrain_indices = list(range(50000)) # first 1000 indices is the training set
    test_indices = list(range(50001, 51000)) # next 1000 indices is the test set
    client_indices = list(range(51001, 51002)) # first 1000 indices is the pretrain set
    trainset = population.subset(client_indices)
    testset = population.subset(test_indices)
    pretrainset = population.subset(pretrain_indices)

    return trainset, testset, pretrainset


def get_cifar100_dataset(data_path):
    # Create the combined CIFAR-100 dataset

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    population = CifarDataset.from_cifar100(root=data_path, download=True, transform=transform)

    file_path = data_path + "cifar100.pkl"
    if not os.path.exists(file_path):
        with open(file_path, "wb") as file:
            pickle.dump(population, file)
            print(f"Save data to {file_path}.pkl")

    # Create a subset of the dataset (first 1000 samples)   
    pretrain_indices = list(range(50000)) # first 1000 indices is the training set
    test_indices = list(range(50001, 51000)) # next 1000 indices is the test set
    client_indices = list(range(51001, 51002)) # first 1000 indices is the pretrain set
    trainset = population.subset(client_indices)
    testset = population.subset(test_indices)
    pretrainset = population.subset(pretrain_indices)

    return trainset, testset, pretrainset



