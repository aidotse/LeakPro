"""TabularExtension class for handling tabular data with one-hot encoding and decoding."""

import numpy as np
from torch import Tensor, cat, mean, randn, std
from torch.utils.data import DataLoader, Dataset

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

    def replace_data_with_noise(self: Self, indices:np.array) -> DataLoader:
        """Get dataloader with random noise images of the same shape as the client_loader, using the same labels."""

        dataset = self.get_dataset(indices)
        dataset_shape = getattr(dataset, self.feature_name).shape
        random_images = randn(*dataset_shape)
        labels = getattr(dataset, self.label_name).clone()

        return random_images, labels




    def get_meanstd(self: Self, trainset: Dataset) -> tuple[Tensor, Tensor]:
        """Get the mean and standard deviation of the data in the trainset."""
        cc = cat([trainset[i].reshape(3, -1) for i in range(len(trainset))], dim=1)
        data_mean = mean(cc, dim=1, keepdim=True)
        data_std = std(cc, dim=1, keepdim=True)
        return data_mean[:, None, None], data_std[:, None, None]
