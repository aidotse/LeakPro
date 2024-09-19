"""TabularExtension class for handling tabular data with one-hot encoding and decoding."""

from copy import deepcopy

import numpy as np
from torch import randn
from torch.utils.data import DataLoader

from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class ImageExtension:
    """Class for handling extra functionality for image data."""

    # Assumes that the data is in shape: batchsize, channels, height, width

    def __init__(self:Self) -> None:

        x, y = next(iter(self.get_dataloader(0)))

        # find out the name of the image attribute in the data object
        for attr in dir(self.population):
            if hasattr(getattr(self.population, attr), "shape"):
                if getattr(self.population, attr).shape[1:] == x.shape[1:]:
                    self.feature_name = attr
                if getattr(self.population, attr).shape[1:] == y.shape[1:]:
                    self.label_name = attr
        if self.feature_name is None:
            raise ValueError("Could not find a matching attribute for features in self.population.")
        if self.label_name is None:
            raise ValueError("Could not find a matching attribute for labels in self.population.")

        logger.info(f"Found feature attribute: {self.feature_name} and label attribute: {self.label_name}")

    def replace_data_with_noise(self: Self, indices:np.array, batch_size:int) -> DataLoader:
        """Get dataloader with random noise images of the same shape as the client_loader, using the same labels."""

        dataset = self.get_dataset(indices)

        num_images = len(dataset)
        random_images = randn((num_images, *self.img_shape))

        dataset_copy = deepcopy(dataset)
        setattr(dataset_copy, self.feature_name, random_images)

        return DataLoader(dataset_copy, batch_size=batch_size, shuffle=True)