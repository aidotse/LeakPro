"""CelebA-HQ data handler — UserDataset only."""

import os

from torch import cat
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms

from leakpro import AbstractInputHandler


class CelebADataHandler(AbstractInputHandler, role="data"):
    """Provides UserDataset for CelebA-HQ. No training logic."""

    class UserDataset(AbstractInputHandler.UserDataset):
        def __init__(self, data, targets, transform=None, indices=None):
            self.data = data
            self.targets = targets
            self.transform = transform
            self.indices = indices

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, idx):
            image = self.data[idx]
            label = self.targets[idx]
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
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), train_transform)
            test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), test_transform)
            combined_dataset = ConcatDataset([train_dataset, test_dataset])

            loader = DataLoader(combined_dataset, batch_size=64, shuffle=False)
            data_list, target_list = [], []
            for data, target in loader:
                data_list.append(data)
                target_list.append(target)

            data = cat(data_list, dim=0)
            targets = cat(target_list, dim=0)
            return cls(data, targets)
