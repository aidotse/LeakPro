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
    data_dir =  train_config["data"]["data_dir"] + "/celeba_hq_data.pkl"

    if not os.path.exists(data_dir):
        population_dataset = celebHqDataset.from_celebHq(config=train_config)
        with open(data_dir, "wb") as file:
            pickle.dump(population_dataset, file)
            print(f"Save data to {data_dir}")
    else:
        with open(data_dir, "rb") as file:
            population_dataset = pickle.load(file)
            print(f"Load data from {data_dir}")
    
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
