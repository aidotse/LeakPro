# ruff: noqa
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
import pickle
from torch import cat


class celebADataset(Dataset):
    def __init__(self, x, y, transform=None,  indices=None):
        """
        Dataset for celebA.

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
    
    def get_classes(self):
        return len(self.y.unique())
    
    

    @classmethod
    def from_celebA(cls, config, subfolder):
        re_size = 64
        crop_size = 108
        offset_height = (218 - crop_size) // 2
        offset_width = (178 - crop_size) // 2
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]    
        
        data_dir = config["data"]["data_dir"]
        train_transform = transforms.Compose([
           transforms.ToTensor(),
           transforms.Lambda(crop),
           transforms.ToPILImage(),
           transforms.Resize((re_size, re_size)),
           transforms.ToTensor(),
        ])

        train_dataset = datasets.ImageFolder(os.path.join(data_dir, subfolder), train_transform)
    
        train_dataset.class_to_idx = {cls_name: int(cls_name) for cls_name in train_dataset.class_to_idx.keys()}

        # Prepare data loader to iterate over combined_dataset
        loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

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
        return celebADataset(self.x[indices], self.y[indices], transform=self.transform)


def get_celebA_train_test_loader(train_config):
    """This function returns the train and test data loaders for the private CelebA dataset."""
    # TODO: Stratified sampling for train and test
    train_fraction = train_config["data"]["f_train"]
    test_fraction = train_config["data"]["f_test"]
    batch_size = train_config["train"]["batch_size"]
    data_dir =  train_config["data"]["data_dir"] + "/celebA_private_data.pkl"

    if not os.path.exists(data_dir):
        population_dataset = celebADataset.from_celebA(config=train_config, subfolder='private')
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


def get_celebA_publicloader(train_config):
    """This function returns the data loader for the public CelebA dataset."""
    batch_size = train_config["train"]["batch_size"]
    data_dir =  train_config["data"]["data_dir"] + "/celebA_public_data.pkl"

    if not os.path.exists(data_dir):
        population_dataset = celebADataset.from_celebA(config=train_config, subfolder='public')
        with open(data_dir, "wb") as file:
            pickle.dump(population_dataset, file)
            print(f"Save data to {data_dir}")
    else:
        with open(data_dir, "rb") as file:
            population_dataset = pickle.load(file)
            print(f"Load data from {data_dir}")

    return DataLoader(population_dataset, batch_size =batch_size, shuffle=False)


