"""Module with functions for preparing the dataset for training the target models."""
import torchvision
from torch import as_tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from leakpro.fl_utils.data_utils import get_meanstd


def get_cifar10_dataset(pre_train_batch_size: int = 64, num_workers:int = 2) -> Dataset:
    """Get the full dataset for CIFAR10."""
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
    client_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())
    data_mean, data_std = get_meanstd(trainset)
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(data_mean, data_std)])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transform])
    trainset.transform = transform_train
    client_dataset.transform = transform_train
    data_mean = as_tensor(data_mean)[:, None, None]
    data_std = as_tensor(data_std)[:, None, None]
    pre_train_loader = DataLoader(trainset, batch_size=pre_train_batch_size,
                                            shuffle=False, drop_last=True, num_workers=num_workers)
    return pre_train_loader, client_dataset, data_mean, data_std
