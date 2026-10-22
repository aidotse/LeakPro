#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""CIFAR data handler for the webapp: upload this as the dataset handler in Step 1.

The population file ``examples/mia/cifar/data/cifar10.pkl`` (built by ``cifar_main.ipynb``) pickles
instances of ``cifar_data_handler.CifarDataHandler.UserDataset``. The webapp registers this file under
the module name derived from the class name (``CifarDataHandler`` -> ``cifar_data_handler``), so the
class name and the attributes ``__getitem__`` reads must stay as they are here for that file to load.
"""

from torchvision import transforms

from leakpro.input_handler.abstract_input_handler import AbstractInputHandler


class CifarDataHandler(AbstractInputHandler, role="data"):
    """Provides UserDataset for CIFAR. No training logic."""

    class UserDataset(AbstractInputHandler.UserDataset):
        """CIFAR images in [0, 1], normalised per channel on access, with optional augmentation."""

        def __init__(self, data, targets, **kwargs) -> None:  # noqa: ANN001, ANN003
            assert data.shape[0] == targets.shape[0], "Data and targets must have the same length"
            assert data.max() <= 1.0, "Data should be in range [0,1]"
            assert data.min() >= 0.0, "Data should be in range [0,1]"

            self.data = data.float()
            self.targets = targets
            augment_strength = kwargs.pop("augment_strength", "none")
            if augment_strength == "none":
                self.augment = None
            elif augment_strength == "easy":
                self.augment = transforms.Compose([
                    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                    transforms.RandomHorizontalFlip(),
                ])
            else:
                raise ValueError(f"Unknown augment_strength: {augment_strength}")
            self.erase_post_norm = None

            # Subsets are rebuilt from return_params(), which carries mean, std and augment along.
            for key, value in kwargs.items():
                setattr(self, key, value)

            if not hasattr(self, "mean") or not hasattr(self, "std"):
                self.mean = self.data.mean(dim=(0, 2, 3)).view(-1, 1, 1)
                self.std = self.data.std(dim=(0, 2, 3)).view(-1, 1, 1)

        def __getitem__(self, index):  # noqa: ANN001, ANN204
            """Return the normalised (and possibly augmented) image and its label."""
            x = self.data[index]
            y = self.targets[index]
            if self.augment is not None:
                x = self.augment(x)
            x = (x - self.mean) / self.std
            if self.erase_post_norm is not None:
                x = self.erase_post_norm(x)
            return x, y

        def __len__(self) -> int:
            """Return the number of records."""
            return len(self.targets)
