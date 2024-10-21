"""Module with functions for preparing the dataset for training the target models."""
import torchvision
from torch import Tensor, as_tensor, cuda, device, randperm
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from leakpro.fl_utils.data_utils import get_meanstd

DEVICE = device("cuda" if cuda.is_available() else "cpu")

def get_cifar100_loader(num_images:int = 16, client_batch_size:int = 32, pre_train_batch_size: int=64,
                        num_workers:int = 2 ) -> tuple[DataLoader, DataLoader, Tensor, Tensor]:
    """Get the full dataset for CIFAR10."""
    trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transforms.ToTensor())
    valset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transforms.ToTensor())
    data_mean, data_std = get_meanstd(trainset)
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(data_mean, data_std)])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transform])

    trainset.transform = transform_train
    valset.transform = transform_train

    total_examples = len(valset)
    random_indices = randperm(total_examples)[:num_images]
    subset_client_trainset = Subset(valset, random_indices)
    client_trainloader = DataLoader(subset_client_trainset, batch_size=client_batch_size,
                                            shuffle=False, drop_last=True, num_workers=num_workers)
    pre_train_loader = DataLoader(trainset, batch_size=pre_train_batch_size,
                                            shuffle=False, drop_last=True, num_workers=num_workers)
    data_mean = as_tensor(data_mean)[:, None, None]
    data_std = as_tensor(data_std)[:, None, None]
    return client_trainloader, pre_train_loader, data_mean, data_std
