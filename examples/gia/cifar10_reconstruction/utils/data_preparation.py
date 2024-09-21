"""Module with functions for preparing the dataset for training the target models."""
import os
import pickle

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms

class CombinedCIFAR10(Dataset):
    def __init__(self, data, targets, transform=None, indices=None):

        self.data = data
        self.targets = targets
        self.transform = transform
        self.indices = indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Retrieve the image and label at the given index
        image, label = self.data[index], self.targets[index]

        # Apply the transform, if specified
        if self.transform:
            image = self.transform(image)

        return image, label

    @classmethod
    def from_cifar10(cls, root="./data", download=True, transform=None):
        # Load the CIFAR10 train and test datasets
        trainset = CIFAR10(root=root, train=True, download=download, transform=transforms.ToTensor())
        testset = CIFAR10(root=root, train=False, download=download, transform=transforms.ToTensor())

        # Concatenate both datasets' data and labels
        data = torch.cat([torch.tensor(trainset.data, dtype=torch.float32), 
                          torch.tensor(testset.data, dtype=torch.float32)], 
                          dim=0)
        # Rescale data from [0, 255] to [0, 1]
        data /= 255.0
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # Permute to change shape from [batch_size, 32, 32, 3] to [batch_size, 3, 32, 32]
        data = data.permute(0, 3, 1, 2)
        data = normalize(data)
        
        targets = torch.cat([torch.tensor(trainset.targets), torch.tensor(testset.targets)], dim=0)

        
        
        # Return an instance of CombinedCIFAR10
        return cls(data, targets)

    def subset(self, indices):
        """
        Returns a new CombinedCIFAR10 object that is a subset of the original dataset.
        
        Args:
            indices (list): List of indices for the subset.

        Returns:
            CombinedCIFAR10: A new dataset that contains only the subset of data.
        """
        # Create a subset of the data and targets
        subset_data = self.data[indices]
        subset_targets = self.targets[indices]
        return CombinedCIFAR10(subset_data, subset_targets, self.transform, indices)



def get_cifar10_dataset(data_path):
    # Create the combined CIFAR-10 dataset
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    population = CombinedCIFAR10.from_cifar10(root=data_path, download=True, transform=transform)

    file_path = data_path + "cifar10.pkl"
    if not os.path.exists(file_path):
        with open(file_path, "wb") as file:
            pickle.dump(population, file)
            print(f"Save data to {file_path}.pkl")
            
    # Create a subset of the dataset (first 1000 samples)   
    train_indices = list(range(5,7)) # first 1000 indices is the training set
    test_indices = list(range(1000, 2000)) # next 1000 indices is the test set
    trainset = population.subset(train_indices)
    testset = population.subset(test_indices)

    return trainset, testset

