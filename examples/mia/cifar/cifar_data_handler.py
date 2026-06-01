"""CIFAR data handler — UserDataset only (normalization + augmentation).

Shared by both standard and DP-SGD training pipelines.
Usage:
    from cifar_data_handler import CifarDataHandler
    population = CifarDataHandler.UserDataset(data, targets)
"""

from torchvision import transforms

from leakpro.input_handler.abstract_input_handler import AbstractInputHandler


class CifarDataHandler(AbstractInputHandler, role="data"):
    """Provides UserDataset for CIFAR. No training logic."""

    class UserDataset(AbstractInputHandler.UserDataset):
        def __init__(self, data, targets, **kwargs):
            assert data.shape[0] == targets.shape[0], "Data and targets must have the same length"
            assert data.max() <= 1.0 and data.min() >= 0.0, "Data should be in range [0,1]"

            self.data = data.float()
            self.targets = targets
            augment_strength = kwargs.pop("augment_strength", "none")

            easy = [
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
            ]
            medium = easy + [
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.RandomRotation(degrees=10, fill=0),
            ]
            strong = medium + [
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"),
            ]

            self.erase_post_norm = None
            if augment_strength == "none":
                self.augment = None
            elif augment_strength == "easy":
                self.augment = transforms.Compose(easy)
            elif augment_strength == "medium":
                self.augment = transforms.Compose(medium)
            elif augment_strength == "strong":
                self.augment = transforms.Compose(strong)
                self.erase_post_norm = transforms.RandomErasing(
                    p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0.0, inplace=False)
            else:
                raise ValueError(f"Unknown augment_strength: {augment_strength}")

            for key, value in kwargs.items():
                setattr(self, key, value)

            if not hasattr(self, "mean") or not hasattr(self, "std"):
                self.mean = self.data.mean(dim=(0, 2, 3)).view(-1, 1, 1)
                self.std = self.data.std(dim=(0, 2, 3)).view(-1, 1, 1)

        def _normalize(self, x):
            return (x - self.mean) / self.std

        def __getitem__(self, index):
            x = self.data[index]
            y = self.targets[index]
            if self.augment is not None:
                x = self.augment(x)
            x = self._normalize(x)
            if self.erase_post_norm is not None:
                x = self.erase_post_norm(x)
            return x, y

        def __len__(self):
            return len(self.targets)
