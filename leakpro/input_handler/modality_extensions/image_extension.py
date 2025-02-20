"""Module for handling image data with preprocessing and normalization."""
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms

from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class ImageExtension:
    """Class for handling image data with preprocessing and normalization."""

    def __init__(self: Self, transform: transforms.Compose) -> None:
        """Initialize the ImageExtension class and perform initial checks."""
        x, y = next(iter(self.get_dataloader(0)))
        if not isinstance(x, (Tensor, np.ndarray, Image.Image)) or not isinstance(y, (Tensor, np.ndarray)):
            raise ValueError("Data must be a tensor, numpy array, or PIL image.")

        if transform is not None:
            self.transform = transform
            logger.info("ImageExtension initialized with custom transformations.")
            return

        logger.info("ImageExtension initialized with NO transformations.")

    def preprocess_image(self: Self, image: Image.Image) -> Tensor:
        """Preprocess a single image."""
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL image.")
        return self.transform(image)

    def preprocess_batch(self: Self, images: list) -> Tensor:
        """Preprocess a batch of images."""
        return torch.stack([self.preprocess_image(img) for img in images])

