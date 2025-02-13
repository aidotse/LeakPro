"""Parent class for user inputs."""

from abc import ABC, abstractmethod

from torch import Tensor

from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.utils.import_helper import Self


class AbstractModalityExtension(ABC):
    """Parent class for modality extension."""

    def __init__(self:Self, handler:MIAHandler) -> None:
        """Initialize the modality extension class."""
        self.handler = handler

    @abstractmethod
    def augmentation(self:Self, data:Tensor, n_aug:int) -> Tensor:
        """Abstract function to implement modality specific augmentations."""
        pass
