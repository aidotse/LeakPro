import os
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from torch import cat, float32, tensor, stack
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CelebA
from torchvision.transforms import ToTensor, Normalize, Compose
from tqdm import tqdm

class celebADataset(Dataset):
    def __init__(self, x, y, transform=None,  indices=None) -> None:
        """Custom dataset for celebA data.

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
    def from_celebA(cls, config, download=True, transform=None):
        root = config["data"]["data_dir"]
        split = config["data"].get("split", "all")  # Use 'all' as the default split

        # Define transformation if not provided
        if transform is None:
            transform = Compose([
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        # Load CelebA dataset with the transformation
        celebA_dataset = CelebA(root=root, split=split, download=download, transform=transform)

        # Initialize lists to store data and targets
        images = []
        targets = []

        # Process the dataset
        for img, target in tqdm(celebA_dataset, desc="Processing CelebA data"):
            images.append(img)  # img is already transformed
            targets.append(target)

        # Convert lists to tensors
        data = stack(images, dim=0).type(float32)
        targets = stack(targets, dim=0).type(float32)

        return cls(data, targets)


    def subset(self, indices):
        """Return a subset of the dataset based on the given indices."""
        return celebADataset(self.x[indices], self.y[indices], transform=self.transform)


def get_celebA_dataloader(data_path, train_config):
    # Create the combined celebA dataset
    train_fraction = train_config["data"]["f_train"]
    test_fraction = train_config["data"]["f_test"]
    batch_size = train_config["train"]["batch_size"]

    transform =Compose([ToTensor(),Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    population_dataset = celebADataset.from_celebA(config=train_config, download=True, transform=transform)

    file_path =  "data/celebA.pkl"
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



