from PIL import Image
import numpy as np
import torch
from torch import Tensor
from torchvision import transforms
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger

class ImageExtension:
    """Class for handling image data with preprocessing and normalization."""

    def __init__(self: Self) -> None:
        """Initialize the ImageExtension class and perform initial checks."""
        x, y = next(iter(self.get_dataloader(0)))
        if not isinstance(x, (Tensor, np.ndarray, Image.Image)) or not isinstance(y, (Tensor, np.ndarray)):
            raise ValueError("Data must be a tensor, numpy array, or PIL image.")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Example resize
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization
        ])

        logger.info("ImageExtension initialized with default transformations.")

    def preprocess_image(self: Self, image: Image.Image) -> Tensor:
        """Preprocess a single image."""
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL image.")
        return self.transform(image)

    def preprocess_batch(self: Self, images: list) -> Tensor:
        """Preprocess a batch of images."""
        return torch.stack([self.preprocess_image(img) for img in images])

    def get_dataloader(self: Self, batch_size: int):
        """Placeholder method for getting a dataloader."""
        # Implement this method to return a dataloader for your dataset
        pass
