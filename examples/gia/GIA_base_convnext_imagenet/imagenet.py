"""Module with functions for preparing the dataset for training the target models."""
import torchvision
from torch import Tensor, as_tensor, randperm
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def get_imagenette_loader(
    start_idx=None,
    num_images: int = 1,
    batch_size: int = 1,
    num_workers: int = 2,
    root: str = "./data",
    split: str = "train",     # "train" or "val"
    size: str = "160px",      # "160px" (fast) or "320px"
) -> tuple[DataLoader, Tensor, Tensor]:
    """
    Imagenette loader with ImageNet normalization.
    Input size is fixed to 224x224.
    """

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])

    dataset = torchvision.datasets.Imagenette(
        root=root,
        split=split,
        size=size,
        download=True,
        transform=transform,
    )

    total_examples = len(dataset)
    if start_idx is None:
        indices = randperm(total_examples)[:num_images].tolist()
    else:
        indices = list(range(start_idx, min(start_idx + num_images, total_examples)))

    subset = Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )

    data_mean = as_tensor(_IMAGENET_MEAN)[:, None, None]
    data_std = as_tensor(_IMAGENET_STD)[:, None, None]

    return loader, data_mean, data_std

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
