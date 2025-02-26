"""Util functions relating to data."""
from abc import ABC, abstractmethod
from torch import Tensor, cat, mean, randn, std, tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch

class GiaDataModalityExtension(ABC):
    @abstractmethod
    def get_at_data():
        """Get a dataloader mimicing the shape of the original data used for recreating."""
        pass

class GiaImageClassifictaionExtension(GiaDataModalityExtension):

    def get_at_data(self, client_loader: DataLoader) -> DataLoader:
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

class ReconstructionDataset(Dataset):
    def __init__(self, reconstruction: torch.Tensor, labels: list):
        # Save the global tensor and labels.
        self.reconstruction = reconstruction
        self.labels = labels

    def __len__(self):
        return self.reconstruction.size(0)

    def __getitem__(self, index):
        # Return just the index (and corresponding label)
        return index, self.labels[index]

class GiaImageDetectionExtension(GiaDataModalityExtension):

    def get_at_data(self, client_loader: DataLoader) -> DataLoader:
        """DataLoader with random noise images of the same shape as the client_loader's dataset, using the same COCO labels."""
        img_shape = client_loader.dataset[0][0].shape
        num_images = len(client_loader.dataset)
        reconstruction = randn((num_images, *img_shape))  # Random noise images
        labels = []
        for _, label in client_loader:
            labels.append(label)
        reconstruction_dataset = ReconstructionDataset(reconstruction, labels) #list(zip(reconstruction, labels))
        reconstruction_loader = DataLoader(reconstruction_dataset, batch_size=32, shuffle=True)

        return reconstruction, labels, reconstruction_loader

class GiaTextMaskingExtension(GiaDataModalityExtension):
    def get_at_data():
        pass


def get_meanstd(trainset: Dataset, axis_to_reduce: tuple=(-2,-1)) -> tuple[Tensor, Tensor]:
    """Get mean and std of a dataset."""
    cc = cat([trainset[i][0].unsqueeze(0) for i in range(len(trainset))], dim=0)
    axis_to_reduce += (0,)
    data_mean = mean(cc, dim=axis_to_reduce).tolist()
    data_std = std(cc, dim=axis_to_reduce).tolist()
    return data_mean, data_std