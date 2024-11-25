# ruff: noqa
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import pickle
from torch import cat, float32, tensor


class celebHqDataset(Dataset):
    def __init__(self, x, y, transform=None,  indices=None):
        """
        Custom dataset for CIFAR data.

        Args:
            x (torch.Tensor): Tensor of input images.
            y (torch.Tensor): Tensor of labels.
            transform (callable, optional): Optional transform to be applied on the image tensors.
        """
        self.x = x
        self.y = y
        self.transform = transform
        self.indices = indices

    def __len__(self):
        """Return the total number of samples."""
        return len(self.y)

    def __getitem__(self, idx):
        """Retrieve the image and its corresponding label at index 'idx'."""
        image = self.x[idx]
        label = self.y[idx]

        # Apply transformations to the image if any
        if self.transform:
            image = self.transform(image)

        return image, label


    @classmethod
    def from_celebHq(cls, config):
        data_dir = config["data"]["data_dir"]
        train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         ])

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transform)
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), test_transform)
        combined_dataset = ConcatDataset([train_dataset, test_dataset])

        # Prepare data loader to iterate over combined_dataset
        loader = DataLoader(combined_dataset, batch_size=1, shuffle=False)

        # Collect all data and targets
        data_list = []
        target_list = []
        for data, target in loader:
            data_list.append(data)  # Remove batch dimension
            target_list.append(target)

        # Concatenate data and targets into large tensors
        data = cat(data_list, dim=0)  # Shape: (N, C, H, W)
        targets = cat(target_list, dim=0)  # Shape: (N,)


        return cls(data, targets)
    

    def subset(self, indices):
        """Return a subset of the dataset based on the given indices."""
        return celebHqDataset(self.x[indices], self.y[indices], transform=self.transform)


def get_celebA_hq_dataloader(data_path, train_config):
    train_fraction = train_config["data"]["f_train"]
    test_fraction = train_config["data"]["f_test"]
    batch_size = train_config["train"]["batch_size"]
    data_dir =  train_config["data"]["data_dir"]

    population_dataset = celebHqDataset.from_celebHq(config=train_config)


    file_path =  "data2/celeba_hq.pkl"
    if not os.path.exists(file_path):
        with open(file_path, "wb") as file:
            pickle.dump(population_dataset, file)
            print(f"Save data to {file_path}")
    
    dataset_size = len(population_dataset)
    train_size = int(train_fraction * dataset_size)
    test_size = int(test_fraction * dataset_size)

    # Use sklearn's train_test_split to split into train and test indices
    selected_index = np.random.choice(np.arange(dataset_size), train_size + test_size, replace=False)
    train_indices, test_indices = train_test_split(selected_index, test_size=test_size)

    train_subset = Subset(population_dataset, train_indices)
    test_subset = Subset(population_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size =batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size= batch_size, shuffle=False)

    return train_loader, test_loader








# class ImageFolderDataset:
#     """Class to handle custom dataset loading and preprocessing."""

#     def __init__(self, data_dir, train_transform, test_transform):
#         """
#         Initialize the dataset class.

#         Args:
#             data_dir (str): Directory containing the data folders ('train' and 'test').
#             train_transform (callable): Transformations to apply to training data.
#             test_transform (callable): Transformations to apply to test data.
#         """
#         self.data_dir = data_dir
#         self.train_transform = train_transform
#         self.test_transform = test_transform
#         self.train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transform)
#         self.test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), test_transform)
#         self.combined_dataset = ConcatDataset([self.train_dataset, self.test_dataset])

#     def split_dataset(self, train_fraction=0.8):
#         """
#         Split the combined dataset into train and test subsets.

#         Args:
#             train_fraction (float): Fraction of data to be used for training.

#         Returns:
#             train_subset (Subset): Training subset.
#             test_subset (Subset): Testing subset.
#         """
#         dataset_size = len(self.combined_dataset)
#         indices = np.arange(dataset_size)
#         np.random.shuffle(indices)

#         train_size = int(train_fraction * dataset_size)
#         train_indices = indices[:train_size]
#         test_indices = indices[train_size:]

#         train_subset = Subset(self.combined_dataset, train_indices)
#         test_subset = Subset(self.combined_dataset, test_indices)

#         return train_subset, test_subset

#     @staticmethod
#     def get_dataloaders(train_subset, test_subset, batch_size=16, num_workers=2):
#         """
#         Create DataLoaders for train and test subsets.

#         Args:
#             train_subset (Subset): Training subset.
#             test_subset (Subset): Testing subset.
#             batch_size (int): Batch size for DataLoader.
#             num_workers (int): Number of workers for DataLoader.

#         Returns:
#             train_loader (DataLoader): DataLoader for training data.
#             test_loader (DataLoader): DataLoader for testing data.
#         """
#         train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#         test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#         return train_loader, test_loader


# def get_imagefolder_dataloaders(data_dir, train_config):
#     """
#     Generate DataLoaders for an ImageFolder dataset.

#     Args:
#         data_dir (str): Path to the dataset directory.
#         train_config (dict): Configuration dictionary with training parameters.

#     Returns:
#         train_loader (DataLoader): DataLoader for training data.
#         test_loader (DataLoader): DataLoader for testing data.
#     """
#     train_fraction = train_config["data"]["f_train"]
#     batch_size = train_config["train"]["batch_size"]

#     train_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     test_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     dataset = ImageFolderDataset(data_dir=data_dir, train_transform=train_transform, test_transform=test_transform)
#     train_subset, test_subset = dataset.split_dataset(train_fraction=train_fraction)
#     train_loader, test_loader = dataset.get_dataloaders(train_subset, test_subset, batch_size=batch_size)

#     return train_loader, test_loader
