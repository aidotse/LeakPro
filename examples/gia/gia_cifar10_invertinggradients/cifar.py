"""Module with functions for preparing the dataset for training the target models."""
import torchvision
from torch import as_tensor, cuda, device, randperm
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms

from leakpro.fl_utils.data_utils import get_meanstd

DEVICE = device("cuda" if cuda.is_available() else "cpu")

def get_cifar10_tensor(num_images:int =1, batch_size:int = 1, num_workers:int = 2 ) -> TensorDataset:
    """Get the full dataset for CIFAR10."""
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
    data_mean, data_std = get_meanstd(trainset)
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std)])
    trainset.transform = transform

    total_examples = len(trainset)
    random_indices = randperm(total_examples)[:num_images] 
    subset_trainset = Subset(trainset, random_indices)
    trainloader = DataLoader(subset_trainset, batch_size=batch_size,
                                            shuffle=False, drop_last=True, num_workers=num_workers)
    data_mean = as_tensor(data_mean)[:, None, None]
    data_std = as_tensor(data_std)[:, None, None]
    return trainloader, data_mean, data_std
