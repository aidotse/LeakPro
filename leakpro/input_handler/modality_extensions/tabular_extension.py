"""TabularExtension class for handling tabular data with one-hot encoding and decoding."""

from numpy import ndarray
from torch import Tensor, argmax, cat, tensor
from torch.nn.functional import one_hot

from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.input_handler.modality_extensions.modality_extension import AbstractModalityExtension
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class TabularExtension(AbstractModalityExtension):
    """Class for handling tabular data with one-hot encoding and decoding.

    Assumes that the data is a tensor or numpy array.
    """

    def __init__(self:Self, handler:MIAHandler) -> None:
        """Check if the data is a pandas DataFrame."""

        super().__init__(handler)
        logger.info("Image extension initialized.")

        x,y = next(iter(self.get_dataloader(0)))
        if not isinstance(x, (Tensor, ndarray)) or not isinstance(y, (Tensor,ndarray)):
            raise ValueError("Data must be a tensor or nparray.")

        if hasattr(self.population, "dec_to_onehot"):
            self.dec_to_onehot = self.population.dec_to_onehot

            # Check number of continuous and categorical columns
            n_dec_cols = len(self.dec_to_onehot)
            n_cat_cols = len([v for v in self.dec_to_onehot.values() if len(v) > 1])
            n_cont_cols = n_dec_cols - n_cat_cols
            logger.info(f"Data contains {n_cat_cols} categorical columns and {n_cont_cols} continuous columns.")

            # Set flag to know if data is one-hot encoded
            self.one_hot_encoded = x.shape[1] != n_dec_cols
            logger.info(f"Data is one-hot encoded: {self.one_hot_encoded}")
        else:
            raise ValueError("Data object must contain dec_to_onehot dict.")

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

    def one_hot_encode(self:Self, data:tensor) -> tensor:
        """One-hot encode all categorical columns in the feature tensor.

        Args:
        ----
            data (torch.Tensor): The tensor to be one-hot encoded.

        Returns:
        -------
            tensor: The one-hot encoded tensor with all categorical columns encoded.

        """
        if self.one_hot_encoded:
            return data
        pointer = 0
        for i in range(len(self.dec_to_onehot)):
            num_classes = len(self.dec_to_onehot[i])
            categorical = num_classes > 1
            if categorical:
                one_hot_encoded = one_hot(data[:, pointer].long(), num_classes)
                data = cat([data[:, :pointer], one_hot_encoded, data[:, pointer + 1:]], dim=1)
                pointer += num_classes
            else:
                pointer += 1
        self.one_hot_encoded = True
        return data

    def one_hot_to_categorical(self:Self, data: tensor) -> tensor:
        """Convert one-hot encoded columns back to categorical labels.

        Args:
        ----
            data (torch.Tensor): The one-hot encoded tensor.

        Returns:
        -------
            torch.Tensor: The tensor with categorical columns restored.

        """
        # Iterate over all columns, decode one-hot where needed
        if not self.one_hot_encoded:
            return data
        for i in range(len(self.dec_to_onehot)):
            num_classes = len(self.dec_to_onehot[i])
            categorical = num_classes > 1
            if categorical:
                one_hot_columns = data[:, i:i + num_classes]
                # Get the categorical labels from one-hot encoded columns
                label = argmax(one_hot_columns, dim=1)
                # Concatenate the categorical labels back into the result tensor
                data = cat([data[:, :i], label.unsqueeze(1), data[:, i + num_classes:]], dim=1)
        self.one_hot_encoded = False
        return data
