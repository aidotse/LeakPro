"""TabularExtension class for handling tabular data with one-hot encoding and decoding."""

from torch import Tensor

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
        return data
