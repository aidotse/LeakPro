import torch
from torch import cuda, device, optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from leakpro import AbstractInputHandler 

class CelebAInputHandler(AbstractInputHandler):
    """Class to handle the user input for the CelebA_HQ dataset."""

    def __init__(self, configs: dict) -> None:
        super().__init__(configs=configs)
        print("Configurations:", configs)

    def get_criterion(self) -> torch.nn.Module:
        """Set the CrossEntropyLoss for the model."""
        return CrossEntropyLoss()

    def get_optimizer(self, model: torch.nn.Module) -> optim.Optimizer:
        """Set the optimizer for the model."""
        learning_rate =0.1  # Default learning rate
        momentum =  0.9  # Default momentum
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    def train(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: optim.Optimizer,
        epochs: int,
    ) -> dict:
        """Model training procedure."""

        if not epochs:
            raise ValueError("Epochs not found in configurations")

        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)

        for epoch in range(epochs):
            train_loss, train_acc = 0.0, 0
            model.train()
            for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
                inputs, labels = inputs.to(gpu_or_cpu), labels.to(gpu_or_cpu)
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Performance metrics
                preds = outputs.argmax(dim=1)
                train_acc += (preds == labels).sum().item()
                train_loss += loss.item()

        model.to("cpu")

        return {"model": model, "metrics": {"accuracy": train_acc / len(dataloader.dataset), "loss": train_loss}}

    def evaluate(self, dataloader: DataLoader, model: torch.nn.Module, criterion: torch.nn.Module) -> dict:
        """Evaluate the model."""
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)
        model.eval()

        test_loss, test_acc = 0.0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Evaluating"):
                inputs, labels = inputs.to(gpu_or_cpu), labels.to(gpu_or_cpu)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                preds = outputs.argmax(dim=1)
                test_acc += (preds == labels).sum().item()
                test_loss += loss.item()

        model.to("cpu")

        return {"accuracy": test_acc / len(dataloader.dataset), "loss": test_loss}
# import os
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, Subset
# import torch

# class CelebADatasetHandler:
#     """Class to handle CelebA-HQ dataset for face recognition."""

#     def __init__(self, data_dir, image_size=224, batch_size=16):
#         """
#         Initialize the CelebA dataset handler.

#         Args:
#             data_dir (str): Directory containing the dataset.
#             image_size (int): Size to resize the images to (square).
#             batch_size (int): Batch size for the dataloader.
#         """
#         self.data_dir = data_dir
#         self.image_size = image_size
#         self.batch_size = batch_size
#         self.transforms_train = transforms.Compose([
#             transforms.Resize((self.image_size, self.image_size)),
#             transforms.RandomHorizontalFlip(),  # Data augmentation
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalization
#         ])
#         self.transforms_test = transforms.Compose([
#             transforms.Resize((self.image_size, self.image_size)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
#         self.train_dataset = None
#         self.test_dataset = None

#     def load_datasets(self):
#         """Load the train and test datasets from the given directory."""
#         train_path = os.path.join(self.data_dir, 'train')
#         test_path = os.path.join(self.data_dir, 'test')

#         self.train_dataset = datasets.ImageFolder(train_path, transform=self.transforms_train)
#         self.test_dataset = datasets.ImageFolder(test_path, transform=self.transforms_test)

#     def combine_datasets(self):
#         """Combine train and test datasets for manual splitting."""
#         if self.train_dataset is None or self.test_dataset is None:
#             raise ValueError("Datasets not loaded. Call `load_datasets()` first.")

#         combined_dataset = self.train_dataset + self.test_dataset  # Concatenates both datasets
#         return combined_dataset

#     def create_dataloader(self, dataset):
#         """Create a dataloader for a given dataset."""
#         return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

#     def split_dataset(self, combined_dataset, train_fraction):
#         """Split the combined dataset into train and test subsets."""
#         dataset_size = len(combined_dataset)
#         train_size = int(train_fraction * dataset_size)
#         indices = torch.randperm(dataset_size).tolist()

#         train_indices = indices[:train_size]
#         test_indices = indices[train_size:]

#         train_subset = Subset(combined_dataset, train_indices)
#         test_subset = Subset(combined_dataset, test_indices)

#         return train_subset, test_subset
