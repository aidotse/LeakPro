"""Module with functions for preparing the dataset for training the target models."""
from abc import ABC, abstractmethod

import numpy as np
import torchvision
from torch import Tensor, as_tensor, cat, cuda, device, mean, randn, std, tensor
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import transforms

from leakpro.import_helper import List, Self

DEVICE = device("cuda" if cuda.is_available() else "cpu")

class DataModule(ABC):
    """Abstract base class for data modules."""

    @abstractmethod
    def get_train_val_loaders(self: Self) -> tuple[DataLoader, DataLoader]:
        """Abstract method to get train and validation loaders."""
        pass

    @abstractmethod
    def get_meanstd(self: Self) -> tuple[Tensor, Tensor]:
        """Abstract method to get the mean and std of the dataset."""
        pass

    @abstractmethod
    def get_subset(self: Self, num_examples: int) -> DataLoader:
        """Abstract method to get a subset of the validation data."""
        pass

    @abstractmethod
    def get_subset_idx(self: Self, indexes: List[int]) -> DataLoader:
        """Abstract method to get a DataLoader with elements corresponding to the given indexes."""
        pass

    @abstractmethod
    def get_at_images(self: Self, client_loader: DataLoader) -> DataLoader:
        """Abstract method to get DataLoader with random noise images."""
        pass

class CifarModule(DataModule):
    """Module working with the Cifar10 dataset."""

    def __init__(self: Self, root: str = "./data", batch_size: int = 32, num_workers: int = 2) -> None:
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transforms.ToTensor())
        valset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transforms.ToTensor())

        data_mean, data_std = self._get_meanstd(trainset)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std)])

        trainset.transform = transform
        valset.transform = transform

        self.trainloader = DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, drop_last=True, num_workers=num_workers)
        self.valloader = DataLoader(valset, batch_size=batch_size,
                                                shuffle=False, drop_last=False, num_workers=num_workers)
        self.data_mean = as_tensor(data_mean)[:, None, None]
        self.data_std = as_tensor(data_std)[:, None, None]
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_train_val_loaders(self: Self) -> tuple[DataLoader, DataLoader]:
        """Getter for train and validation loader."""
        return self.trainloader, self.valloader

    def get_meanstd(self:Self) -> tuple[Tensor, Tensor]:
        """Get mean and std for the dataset."""
        return self.data_mean, self.data_std

    def _get_meanstd(self: Self, trainset: Dataset) -> tuple[Tensor, Tensor]:
        cc = cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
        data_mean = mean(cc, dim=1).tolist()
        data_std = std(cc, dim=1).tolist()
        return data_mean, data_std

    def get_subset(self: Self, num_examples: int) -> DataLoader:
        """Get a dataloader with num_examples elements from the validation loader."""
        target_ids = np.random.choice(len(self.valloader.dataset), size=num_examples, replace=False)
        subset = Subset(self.valloader.dataset, target_ids)

        return DataLoader(
            subset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def get_subset_idx(self: Self, indexes: List[int]) -> DataLoader:
        """Get a DataLoader with elements corresponding to the given indexes from the validation loader."""
        subset = Subset(self.valloader.dataset, indexes)
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def get_at_images(self: Self, client_loader: DataLoader) -> DataLoader:
        """DataLoader with random noise images of the same shape as the client_loader's dataset, using the same labels."""
        img_shape = client_loader.dataset[0][0].shape
        num_images = len(client_loader.dataset)
        reconstruction = randn((num_images, *img_shape))
        labels = []
        for _, label in client_loader:
            labels.extend(label.numpy())
        labels = tensor(labels)
        reconstruction_dataset = TensorDataset(reconstruction, labels)
        reconstruction_loader = DataLoader(reconstruction_dataset, batch_size=32, shuffle=True)
        return reconstruction, reconstruction_loader

