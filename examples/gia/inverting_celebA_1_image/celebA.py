"""Module with functions for preparing the dataset for training the target models."""
from random import sample

from torch import Tensor, as_tensor, randperm
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import CelebA

from leakpro.fl_utils.data_utils import get_meanstd

class CelebAFloatLabels(CelebA):
    def __init__(self, root, split, transform=None, target_transform=None, download=False):
        super().__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)
    
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        label = label.float()
        return image, label


def get_celeba_loader(num_images: int = 1, start_idx= None, batch_size: int = 1, num_workers: int = 2, val_split: float = 0.2) -> tuple[DataLoader, DataLoader, DataLoader, Tensor, Tensor]:
    """Get the CelebA dataset."""
    # Initial transform for computing mean and std
    initial_transform = transforms.Compose([transforms.ToTensor()])

    # Load dataset with initial transform
    dataset = CelebAFloatLabels(root="./data", split="train", download=True, transform=initial_transform)
    # Compute mean and std
    subset_indices = sample(range(len(dataset)), min(len(dataset), 20000))
    data_mean, data_std = get_meanstd(Subset(dataset, subset_indices))

    # Transform with normalization
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std)])
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(20),
        transforms.RandomCrop(227, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_mean, std=data_std)
    ])


    # Random subset of images for the client loader
    total_examples = len(dataset)
    if start_idx is not None:
        # Ensure start_idx + num_images doesn't exceed dataset length
        end_idx = min(start_idx + num_images, total_examples)
        indices = list(range(start_idx, end_idx))
    else:
        # Randomly sample indices
        indices = randperm(total_examples)[:num_images]
    client_subset = Subset(dataset, indices)

    # Remaining data
    remaining_indices = list(set(range(total_examples)) - set(indices))
    remaining_subset = Subset(dataset, remaining_indices)

    # Split remaining data into train and val
    val_size = int(len(remaining_subset) * val_split)
    train_size = len(remaining_subset) - val_size
    train_subset, val_subset = random_split(remaining_subset, [train_size, val_size])

    # Dataloaders
    client_subset.dataset.transform = transform
    train_subset.dataset.transform = transform_train
    client_loader = DataLoader(client_subset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

    return client_loader, train_loader, val_loader, as_tensor(data_mean)[:, None, None], as_tensor(data_std)[:, None, None]
