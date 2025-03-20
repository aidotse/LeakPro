"""ImageExtension class for handling image data transformation and augmentation."""

import numpy as np
import os
import torch
from torch import Tensor
import torchvision.transforms as transforms
from PIL import Image
import random
from torchvision.datasets import CIFAR10

from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.input_handler.modality_extensions.modality_extension import AbstractModalityExtension
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class ImageExtension(AbstractModalityExtension):
    """Class for handling extra functionality for image data."""

    # Assumes that the data is in shape: batchsize, channels, height, width

    def __init__(self:Self, handler:MIAHandler) -> None:

        super().__init__(handler)
        logger.info("Image extension initialized.")

    def augmentation(self:Self, data:Tensor, n_aug:int) -> Tensor:
        """Augment the data by generating additional samples.

        Args:
        ----
            data (Tensor): The input data tensor to augment.
            n_aug (int): The number of augmented samples to generate.

        Returns:
        -------
            Tensor: The augmented data tensor.

        """
        n_aug = n_aug // len(data) #dummy to pass ruff
        return data

    def get_data(self, dataset, index):
        """Retrieve raw sample using metadata from the PKL dataset."""
        try:
            # Get original index from metadata
            original_idx = dataset.metadata["original_idx"][index]
            # Return raw pixel values (0-255) from metadata
            return dataset.metadata["raw_data"][original_idx]
        except Exception as e:
            logger.error(f"Failed to retrieve raw data at index {index}: {e}")
            return None
    
    def to_pil_image(self, raw_sample):
        """Convert raw numpy array (HWC, uint8) to PIL Image."""
        try:
            # Direct conversion (no normalization needed)
            return Image.fromarray(raw_sample)
        except Exception as e:
            logger.error(f"Failed to convert to PIL image: {e}")
            return None
    
    
    def get_transform(self, transform_type="cifar", **kwargs):
        """Returns CIFAR-10 transformation function for RaMIA"""
        if transform_type == "cifar":
            def cifar_transform(image, transform_idx):
                # Seed random generators for reproducibility
                seed = transform_idx  # Unique seed per transformation index
                random.seed(seed)
                torch.manual_seed(seed)

                return transforms.Compose([
                    # Geometric transforms (mild)
                    transforms.RandomHorizontalFlip(p=0.3),
                    transforms.RandomRotation(degrees=(-8, 8)),
                    transforms.RandomResizedCrop(
                        size=32,  # Maintain original size
                        scale=(0.85, 1.0),  # Mild cropping
                        ratio=(0.9, 1.1)
                    ),
                    
                    # Photometric transforms (subtle)
                    transforms.ColorJitter(
                        brightness=0.1,  # ±10% brightness
                        contrast=0.1,     # ±10% contrast
                        saturation=0.1,   # ±10% saturation
                        hue=0.04          # ±4% hue
                    ),
                    
                    # Spatial transforms (conservative)
                    transforms.RandomAffine(
                        degrees=0,
                        translate=(0.08, 0.08)  # ±8% translation
                    ),
                    
                    # Mild noise/blur
                    transforms.GaussianBlur(
                        kernel_size=3,  # Small kernel for 32x32 images
                        sigma=(0.1, 0.75)
                    ),
                    
                    # Final tensor conversion
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])(image)

            return cifar_transform
        else:
            raise ValueError(f"Unsupported transform type: {transform_type}")
