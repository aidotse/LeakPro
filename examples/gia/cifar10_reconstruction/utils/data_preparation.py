"""Module with functions for preparing the dataset for training the target models."""

import torchvision
from torch import Tensor, as_tensor, cat, cuda, device, mean, std
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


DEVICE = device("cuda" if cuda.is_available() else "cpu")

class CifarDataset(Dataset):
    """Module working with the Cifar10 dataset."""

    def __init__(self, root: str = "./data", batch_size: int = 32, num_workers: int = 2) -> None:
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transforms.ToTensor())

        data_mean, data_std = self._get_meanstd(trainset)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std)])

        trainset.transform = transform
        testset.transform = transform

        self.trainloader = DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, drop_last=True, num_workers=num_workers)
        self.testloader = DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, drop_last=False, num_workers=num_workers)
        self.data_mean = as_tensor(data_mean)[:, None, None]
        self.data_std = as_tensor(data_std)[:, None, None]
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_train_val_loaders(self) -> tuple[DataLoader, DataLoader]:
        """Getter for train and validation loader."""
        return self.trainloader, self.testloader

    def get_meanstd(self) -> tuple[Tensor, Tensor]:
        """Get mean and std for the dataset."""
        return self.data_mean, self.data_std

    def _get_meanstd(self, trainset: Dataset) -> tuple[Tensor, Tensor]:
        cc = cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
        data_mean = mean(cc, dim=1).tolist()
        data_std = std(cc, dim=1).tolist()
        return data_mean, data_std



