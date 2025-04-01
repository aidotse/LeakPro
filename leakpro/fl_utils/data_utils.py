"""Util functions relating to data."""
from abc import ABC, abstractmethod
from copy import deepcopy

import torch
from torch import Tensor, cat, mean, randn, std
from torch.utils.data import DataLoader, Dataset

from leakpro.utils.import_helper import Self, Any


class GiaDataModalityExtension(ABC):
    """Abstract class for data modality extensions for GIA."""

    @abstractmethod
    def get_at_data() -> Tensor:
        """Get a dataloader mimicing the shape of the original data used for recreating."""
        pass

class CustomTensorDataset(Dataset):
    """Custom generic tensor dataset."""

    def __init__(self:Self, reconstruction: torch.Tensor, labels: list) -> None:
        self.reconstruction = reconstruction
        self.labels = labels

    def __len__(self) -> int:
        """Dataset length."""
        return self.reconstruction.size(0)

    def __getitem__(self: Self, index: int) -> tuple[Tensor, Any]:
        """Get item from index."""
        return self.reconstruction[index], self.labels[index]

class GiaImageExtension(GiaDataModalityExtension):
    """Image extension for GIA."""

    def get_at_data(self: Self, client_loader: DataLoader) -> DataLoader:
        """DataLoader with random noise images of the same shape as the client_loader's dataset, using the same COCO labels."""
        img_shape = client_loader.dataset[0][0].shape
        num_images = len(client_loader.dataset)
        reconstruction = randn((num_images, *img_shape))
        labels = []
        for _, label in client_loader:
            if isinstance(label, Tensor):
                labels.extend(deepcopy(label))
            else:
                labels.append(deepcopy(label))
        reconstruction_dataset = CustomTensorDataset(reconstruction, labels)
        reconstruction_loader = DataLoader(reconstruction_dataset, batch_size=32, shuffle=True)

        return reconstruction, labels, reconstruction_loader


def get_meanstd(trainset: Dataset, axis_to_reduce: tuple=(-2,-1)) -> tuple[Tensor, Tensor]:
    """Get mean and std of a dataset."""
    cc = cat([trainset[i][0].unsqueeze(0) for i in range(len(trainset))], dim=0)
    axis_to_reduce += (0,)
    data_mean = mean(cc, dim=axis_to_reduce).tolist()
    data_std = std(cc, dim=axis_to_reduce).tolist()
    return data_mean, data_std
