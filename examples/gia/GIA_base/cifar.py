"""Module with functions for preparing the dataset for training the target models."""
import torchvision
from torch import Tensor, as_tensor, randperm
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from leakpro.fl_utils.data_utils import get_meanstd


def get_cifar10_loader(start_idx = None, num_images:int =1, batch_size:int = 1, num_workers:int = 2 ) -> tuple[DataLoader, Tensor, Tensor]:
    """Get the full dataset for CIFAR10."""
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
    data_mean, data_std = get_meanstd(trainset)
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std)])
    trainset.transform = transform

    total_examples = len(trainset)
    if start_idx is None:
        indices = randperm(total_examples)[:num_images]
    else:
        indices = list(range(start_idx, min(start_idx + num_images, total_examples)))
    subset_trainset = Subset(trainset, indices)
    trainloader = DataLoader(subset_trainset, batch_size=batch_size,
                                            shuffle=False, drop_last=True, num_workers=num_workers)
    data_mean = as_tensor(data_mean)[:, None, None]
    data_std = as_tensor(data_std)[:, None, None]
    return trainloader, data_mean, data_std
