"""ImageExtension class for handling image data transformation and augmentation."""

import numpy as np
import pickle
import torch
from torch import Tensor
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from typing import List, Union, Tuple, Optional
import random

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
    
    def to_pil_image(self, data: Union[Tensor, np.ndarray, Image.Image]) -> Image.Image:
        """Convert various data formats to PIL Image for transformation.
        
        This method handles conversion from tensors, numpy arrays, and validates
        PIL images, making it easier to apply transformations consistently.
        
        Args:
            data: Input data that needs to be converted to PIL Image
            _to
        Returns:
            PIL Image object ready for transformation
            
        Raises:
            ValueError: If the tensor shape is unexpected
            TypeError: If the data type is not supported for conversion
        """
        to_pil = transforms.ToPILImage()
        
        if isinstance(data, torch.Tensor):
            if data.dim() in {2, 3}:
                return to_pil(data)
            else:
                raise ValueError(f"Unexpected tensor shape: {data.shape}. Expected 2D or 3D tensor.")
        elif isinstance(data, np.ndarray):
            # Handle floating point arrays by scaling to uint8 range
            if data.dtype == np.float32 or data.dtype == np.float64:
                data = (data * 255).astype(np.uint8)
            # Convert from CHW to HWC format if needed
            if data.ndim == 3 and data.shape[0] in {1, 3}:  # CHW format
                data = data.transpose(1, 2, 0)
            return Image.fromarray(data)
        elif isinstance(data, Image.Image):
            return data
        else:
            raise TypeError(f"Unsupported data type: {type(data)}. Expected tensor, ndarray, or PIL Image.")
        
    def get_transform(self, transform_type, **kwargs):
        """Get transformation function based on specified type."""
        if transform_type == "cifar":
            # Create a list of transformations
            transform_list = [
                # 1. Identity transformation (original image)
                transforms.Compose([
                    transforms.ToTensor()
                ]),
                
                # 2. Horizontal flip
                transforms.Compose([
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.ToTensor()
                ]),
                
                # 3-4. Small rotations (±5 degrees)
                transforms.Compose([
                    transforms.RandomRotation(degrees=(5, 5)),
                    transforms.ToTensor()
                ]),
                transforms.Compose([
                    transforms.RandomRotation(degrees=(-5, -5)),
                    transforms.ToTensor()
                ]),
                
                # 5. Small crop with padding
                transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor()
                ]),
                
                # 6-7. Mild brightness adjustments
                transforms.Compose([
                    transforms.ColorJitter(brightness=(1.1, 1.1)),
                    transforms.ToTensor()
                ]),
                transforms.Compose([
                    transforms.ColorJitter(brightness=(0.9, 0.9)),
                    transforms.ToTensor()
                ]),
                
                # 8-9. Mild contrast adjustments
                transforms.Compose([
                    transforms.ColorJitter(contrast=(1.1, 1.1)),
                    transforms.ToTensor()
                ]),
                transforms.Compose([
                    transforms.ColorJitter(contrast=(0.9, 0.9)),
                    transforms.ToTensor()
                ]),
                
                # 10. Horizontal flip + small rotation
                transforms.Compose([
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.RandomRotation(degrees=(3, 3)),
                    transforms.ToTensor()
                ]),
                
                # 11. Small crop + mild brightness
                transforms.Compose([
                    transforms.RandomCrop(30, padding=2),
                    transforms.Resize(32),
                    transforms.ColorJitter(brightness=(1.05, 1.05)),
                    transforms.ToTensor()
                ]),
                
                # 12. Small translations (shift the image slightly)
                transforms.Compose([
                    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                    transforms.ToTensor()
                ]),
                
                # 13. Small scale change (zoom slightly)
                transforms.Compose([
                    transforms.RandomAffine(degrees=0, scale=(0.95, 0.95)),
                    transforms.ToTensor()
                ]),
                
                # 14. Slight sharpening effect
                transforms.Compose([
                    transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=1.0),
                    transforms.ToTensor()
                ]),
                
                # 15. Training-like combined transformation
                transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.ToTensor()
                ])
            ]
            
            # Define a function that applies one of these transformations based on index
            def transform_function(img, transform_idx=None):
                # Add logging to track which transformation is being applied
                if transform_idx is None:
                    transform_idx = random.randint(0, len(transform_list)-1)
                
                transform_idx = transform_idx % len(transform_list)
                           
                return transform_list[transform_idx](img)
            
            return transform_function
        
                
            
        elif transform_type == "custom":
            # Custom transforms based on provided parameters
            transform_list = []
            
            # Add transforms based on kwargs
            if kwargs.get("random_flip", False):
                transform_list.append(transforms.RandomHorizontalFlip())
                
            if "rotation_degrees" in kwargs:
                transform_list.append(transforms.RandomRotation(kwargs["rotation_degrees"]))
                
            if "resize" in kwargs:
                transform_list.append(transforms.Resize(kwargs["resize"]))
                
            if "crop_size" in kwargs:
                padding = kwargs.get("padding", 0)
                transform_list.append(transforms.RandomCrop(kwargs["crop_size"], padding=padding))
                
            if kwargs.get("color_jitter", False):
                brightness = kwargs.get("brightness", 0.2)
                contrast = kwargs.get("contrast", 0.2)
                saturation = kwargs.get("saturation", 0.2)
                hue = kwargs.get("hue", 0.1)
                transform_list.append(transforms.ColorJitter(
                    brightness=brightness, contrast=contrast, 
                    saturation=saturation, hue=hue
                ))
                
            # Always add ToTensor at the end
            transform_list.append(transforms.ToTensor())
            
            # Add normalization if means and stds are provided
            if "normalization_mean" in kwargs and "normalization_std" in kwargs:
                transform_list.append(transforms.Normalize(
                    mean=kwargs["normalization_mean"],
                    std=kwargs["normalization_std"]
                ))
                
            return transforms.Compose(transform_list)
            
        else:  # Default to standard
            # Standard transforms that work well for general cases
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomResizedCrop(size=32, scale=(0.7, 1.0)),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    #std=[0.229, 0.224, 0.225])
            ])
        

    def _load_dataset(self, data_path):
        """Load a CIFAR or similar image dataset from a file path.
        
        Args:
            data_path: Path to the dataset file, typically a pickle file
            
        Returns:
            The loaded dataset object
        """
        try:
            # Attempt to load from pickle
            with open(data_path, 'rb') as f:
                dataset = pickle.load(f)
                
            # Ensure dataset has necessary attributes
            if not hasattr(dataset, 'data') and not hasattr(dataset, '__getitem__'):
                logger.warning("Loaded dataset does not have required attributes")
                
            logger.info(f"Successfully loaded dataset from {data_path}")
            return dataset
        
        except (FileNotFoundError, ImportError, pickle.UnpicklingError) as e:
            logger.error(f"Failed to load dataset: {e}")
            # Optionally fallback to torchvision download
            logger.info("Attempting to download CIFAR10 as fallback...")
            return torchvision.datasets.CIFAR10(
                root='./data', 
                train=True, 
                download=True
            )

    def get_data(self, dataset, index):
        """Extract raw image data from a dataset at the specified index.
        
        Args:
            dataset: The dataset to extract data from
            index: Index of the image to extract
            
        Returns:
            Raw image data, typically as a numpy array
        """
        try:
            # Handle torchvision dataset case
            if isinstance(dataset, torchvision.datasets.CIFAR10) or isinstance(dataset, torchvision.datasets.CIFAR100):
                image, _ = dataset[index]
                # If tensor, convert to numpy
                if isinstance(image, torch.Tensor):
                    image = image.permute(1, 2, 0).numpy()  # Change to HWC format
                return image
            
            # Handle dataset with a data attribute
            if hasattr(dataset, 'data'):
                return dataset.data[index]  # Already numpy array
            
            # Fallback to generic __getitem__ approach
            item = dataset[index]
            if isinstance(item, tuple) and len(item) >= 1:
                image = item[0]  # Assume (data, label) format
                if isinstance(image, torch.Tensor):
                    image = image.permute(1, 2, 0).numpy()
                return image
            
            # If item itself is the image
            if isinstance(item, torch.Tensor):
                return item.permute(1, 2, 0).numpy()
            
            return item
            
        except Exception as e:
            logger.error(f"Error getting data at index {index}: {e}")
            return None
