import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Dataset, Subset, DataLoader
from torch import tensor, float32, cat



class CifarDataset(Dataset):
    def __init__(self, x, y, transform=None, indices=None, metadata=None):
        """
        Custom dataset for CIFAR data.

        Args:
            x (torch.Tensor): Tensor of input images.
            y (torch.Tensor): Tensor of labels.
            transform (callable, optional): Optional transform to be applied on the image tensors.
        """
        self.x = x
        self.y = y
        self.metadata = metadata if metadata is not None else {
            "original_splits": [], # 'train' or 'test' for each sample
            "original_idx": [],    # Original index in CIFAR train/test sets
            "raw_data": None
        }
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
    def from_cifar(cls, config, download=True, transform=None):
        root = config["data"]["data_dir"]

        # Load the CIFAR train and test datasets
        if config["data"]["dataset"] == "cifar10":
            trainset = CIFAR10(root=root, train=True, download=download, transform=transforms.ToTensor()) # <class 'torchvision.datasets.cifar.CIFAR10'>
            testset = CIFAR10(root=root, train=False, download=download, transform=transforms.ToTensor())
        elif config["data"]["dataset"] == "cifar100":
            trainset = CIFAR100(root=root, train=True, download=download, transform=transforms.ToTensor())
            testset = CIFAR100(root=root, train=False, download=download, transform=transforms.ToTensor())
        else:
            raise ValueError("Unknown dataset type")
        
        # Create tracking metadata
        metadata = {
            "original_splits": [],
            "original_idx": [],
            "raw_data": np.concatenate([trainset.data, testset.data], axis=0) # <class 'numpy.ndarray'>
        }

        # Process training data (keeping as 0-255)
        train_data = tensor(trainset.data, dtype=torch.uint8) # <class 'torch.Tensor'>
        train_targets = tensor(trainset.targets)
        metadata["original_splits"].extend(["train"] * len(trainset))
        metadata["original_idx"].extend(range(len(trainset)))

        # Process test data
        test_data = tensor(testset.data, dtype=torch.uint8)
        test_targets = tensor(testset.targets)
        metadata["original_splits"].extend(["test"] * len(testset))
        metadata["original_idx"].extend(range(len(trainset), len(trainset) + len(testset)))

        # Concatenate both datasets' data and labels
        data = cat([train_data, test_data], dim=0).float() # <class 'torch.Tensor'>
        # Rescale data from [0, 255] to [0, 1]
        data /= 255.0 # <class 'torch.Tensor'>
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        data = data.permute(0, 3, 1, 2) # <class 'torch.Tensor'>
        data = normalize(data)
        
        targets = cat([train_targets, test_targets], dim=0)

        return cls(data, targets, metadata=metadata)
    
    def subset(self, indices):
        subset_metadata = {
            "original_splits": [self.metadata["original_splits"][i] for i in indices],
            "original_idx": [self.metadata["original_idx"][i] for i in indices],
            "raw_data": self.metadata["raw_data"]  # Preserve raw data reference
        }
        return CifarDataset(self.x[indices], self.y[indices], transform=self.transform, metadata=subset_metadata, indices=indices)


def get_cifar_dataloader(data_path, train_config):
    # Create the combined CIFAR-10 dataset
    train_fraction = train_config["data"]["f_train"]
    test_fraction = train_config["data"]["f_test"]
    cifar_type = train_config["data"]["dataset"]
    batch_size = train_config["train"]["batch_size"]

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    population_dataset = CifarDataset.from_cifar(config=train_config, download=True, transform=None) # <class 'examples.mia.cifar.utils.cifar_data_preparation.CifarDataset'>

    file_path =  "data/"+ cifar_type + ".pkl"
    if not os.path.exists(file_path):
        with open(file_path, "wb") as file:
            pickle.dump(population_dataset, file)
            print(f"Saved dataset with metadata to {file_path}")

    dataset_size = len(population_dataset)
    train_size = int(train_fraction * dataset_size)
    test_size = int(test_fraction * dataset_size)

    # Use sklearn's train_test_split to split into train and test indices
    selected_index = np.random.choice(np.arange(dataset_size), train_size + test_size, replace=False)
    train_indices, test_indices = train_test_split(selected_index, test_size=test_size)

    # train_subset = Subset(population_dataset, train_indices)
    # test_subset = Subset(population_dataset, test_indices)

    train_subset = population_dataset.subset(train_indices)
    test_subset = population_dataset.subset(test_indices)
    
    train_loader = DataLoader(train_subset, batch_size =batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size= batch_size, shuffle=False)

    return train_loader, test_loader
