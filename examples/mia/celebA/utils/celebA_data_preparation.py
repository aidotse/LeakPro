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
    def __init__(self,chunk_files, transform=None,  indices=None) -> None:
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
       
        print("Loading data from existing chunks...")
        for chunk_path in chunk_files:
            with open(chunk_path, "rb") as chunk_file:
                images, labels = pickle.load(chunk_file)
                self.image_tensors.append(images)
                self.labels.append(labels)

        # Combine tensors from all chunks
        self.image_tensors = cat(self.image_tensors, dim=0)
        self.labels = cat(self.labels, dim=0)



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


        # celebA_train = CelebA(root=root, split="train", target_type= "identity" , download=True,  transform=transform)
        # celebA_test = CelebA(root=root, split="test", target_type= "identity" , download=True,  transform=transform)
        # celebA_val = CelebA(root=root, split="valid", target_type= "identity" , download=True,  transform=transform)


        # Load CelebA dataset with the transformation
        celebA_dataset = CelebA(root=root, split=split, download=download, transform=transform)

        # Initialize lists to store data and targets
        images = []
        labels = []
        chunk_size = 20000
        for idx, (img, target) in enumerate(tqdm(celebA_dataset, desc="Processing CelebA")):
            # Ensure that the image is in tensor format
            # img = transform(img) if not isinstance(img, tensor) else img
            images.append(img)
            labels.append(target)

            # Save in chunks to avoid memory overload
            if (idx + 1) % chunk_size == 0 or idx == len(celebA_dataset) - 1:
                chunk_path = f"data/celebA_chunk_{idx // chunk_size}.pkl"
                with open(chunk_path, "wb") as chunk_file:
                    # Stack images and convert labels to tensor before saving
                    stacked_images = stack(images, dim=0).type(float32)
                    label_tensor = stack(labels,  dim=0).type(float32)
                    pickle.dump((stacked_images, label_tensor), chunk_file)
                print(f"Saved chunk to {chunk_path}")
                images, labels = [], []  # Reset lists to free memory

        return cls(None, None)  # Since we're saving the chunks, return an empty dataset


    def subset(self, indices):
        """Return a subset of the dataset based on the given indices."""
        return celebADataset(self.x[indices], self.y[indices], transform=self.transform)


def get_celebA_dataloader(data_path, train_config):
    # Create the combined celebA dataset
    train_fraction = train_config["data"]["f_train"]
    test_fraction = train_config["data"]["f_test"]
    batch_size = train_config["train"]["batch_size"]

    transform =Compose([ToTensor(),Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

   # Check if chunk files exist
    num_chunks = 10
    chunk_files = [f"{data_path}/celebA_chunk_{i}.pkl" for i in range(num_chunks)]
    all_images, all_labels = [], []

    if all(os.path.exists(chunk) for chunk in chunk_files):
        # Load data from chunks
        print("Loading data from existing chunks...")
        # Create the dataset
        dataset = celebADataset(chunk_files)
        # for chunk_path in chunk_files:
        #     with open(chunk_path, "rb") as chunk_file:
        #         images, labels = pickle.load(chunk_file)
        #         all_images.append(images)
        #         all_labels.append(labels)
        # # Concatenate all chunks
        # all_images = cat(all_images, dim=0)
        # all_labels = cat(all_labels, dim=0)
        # population_dataset = celebADataset(all_images, all_labels, transform=transform)
    else:
        # Process and save the dataset into chunks
        print("Processing and saving dataset into chunks...")
        population_dataset = celebADataset.from_celebA(config=train_config, download=True, transform=transform)
        for chunk_path in chunk_files:
            with open(chunk_path, "rb") as chunk_file:
                images, labels = pickle.load(chunk_file)
                all_images.append(images)
                all_labels.append(labels)


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


def load_chunk_files(data_dir):
    """
    Collects all chunk file paths from a given directory.
    """
    chunk_files = []
    for file_name in os.listdir(data_dir):
        if file_name.startswith("celebA_chunk") and file_name.endswith(".pkl"):
            chunk_files.append(os.path.join(data_dir, file_name))
    return chunk_files
