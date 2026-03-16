"""Module with functions for preparing the dataset for training the target models."""
import torchvision
from torch import Tensor, as_tensor, randperm
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from leakpro.fl_utils.data_utils import get_meanstd


def get_cifar10_loader(start_idx = None, num_images:int =1, batch_size:int = 1, num_workers:int = 2, excluded_classes: list[int] | None = None) -> tuple[DataLoader, Tensor, Tensor]:
    """Get the full dataset for CIFAR10.
    
    Args:
        start_idx: Starting index for selecting images. If None, random indices are used.
        num_images: Number of images to include in the loader.
        batch_size: Batch size for the DataLoader.
        num_workers: Number of worker threads for data loading.
        excluded_classes: List of class indices to exclude from the dataset (0-9 for CIFAR-10).
                         If None, all classes are included.
    
    Returns:
        Tuple of (DataLoader, data_mean, data_std)
    """
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
    data_mean, data_std = get_meanstd(trainset)
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std)])
    trainset.transform = transform

    # Filter out excluded classes if specified
    if excluded_classes is not None:
        valid_indices = [i for i in range(len(trainset)) if trainset.targets[i] not in excluded_classes]
    else:
        valid_indices = list(range(len(trainset)))
    
    total_examples = len(valid_indices)
    if start_idx is None:
        selected = randperm(total_examples)[:num_images]
        indices = [valid_indices[i] for i in selected]
    else:
        indices = valid_indices[start_idx:min(start_idx + num_images, total_examples)]
    
    subset_trainset = Subset(trainset, indices)
    trainloader = DataLoader(subset_trainset, batch_size=batch_size,
                                            shuffle=False, drop_last=True, num_workers=num_workers)
    data_mean = as_tensor(data_mean)[:, None, None]
    data_std = as_tensor(data_std)[:, None, None]
    return trainloader, data_mean, data_std
