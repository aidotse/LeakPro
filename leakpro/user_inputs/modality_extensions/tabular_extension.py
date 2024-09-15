"""TabularExtension class for handling tabular data with one-hot encoding and decoding."""

import numpy as np
from pandas import DataFrame, Series
from torch import argmax, cat, float32, tensor
from torch.nn.functional import one_hot

from leakpro.import_helper import Self
from leakpro.utils.logger import logger


class TabularExtension:
    """Class for handling tabular data with one-hot encoding and decoding.

    Assumes that the data is a pandas DataFrame where features are NOT one-hot encoded.
    """

    def __init__(self:Self) -> None:
        """Check if the data is a pandas DataFrame."""

        x,y = self.get_dataset(dataset_indices = np.arange(0, self.population_size))
        if isinstance(x, (DataFrame, Series)) and isinstance(y, (DataFrame, Series)):
            # find categorical columns, continuous columns and their indices
            self.cat_cols = x.select_dtypes(include=[np.int_]).columns.tolist()
            self.cat_index_to_n_classes = {x.columns.get_loc(col): x[col].nunique() for col in self.cat_cols}
            self.cont_cols = x.select_dtypes(include=[np.float_]).columns.tolist()
            self.n_cols = len(x.columns)
        else:
            raise ValueError("Data must be a pandas DataFrame.")

        logger.info(f"Continuous cols: {self.cont_cols}")
        logger.info(f"Categorical cols: {self.cat_cols}")

        # overwrite the pandas dataframes in population with tensor versions,
        # this will speed things up alot for training/attacking
        for attr_name in dir(self.population):
            attr_value = getattr(self.population, attr_name)

            # Check if the attribute is a pandas DataFrame
            if isinstance(attr_value, (DataFrame, Series)):
                # Convert DataFrame to a tensor and overwrite the attribute
                tensor_value = tensor(attr_value.values, dtype=float32)
                setattr(self.population, attr_name, tensor_value)
                logger.info(f"Converted {attr_name} to tensor.")

    def one_hot_encode(self:Self, data:tensor) -> tensor:
        """One-hot encode all categorical columns in the tensor.

        Args:
        ----
            data (torch.Tensor): The tensor to be one-hot encoded.

        Returns:
        -------
            tensor: The one-hot encoded tensor with all categorical columns encoded.

        """
        pointer = 0
        for i in range(self.n_cols):
            if i in self.cat_index_to_n_classes:
                num_classes = self.cat_index_to_n_classes[i]
                one_hot_encoded = one_hot(data[:, pointer].long(), num_classes)
                data = cat([data[:, :pointer], one_hot_encoded, data[:, pointer + 1:]], dim=1)
                pointer += num_classes
            else:
                pointer += 1
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
        for i in range(self.n_cols):
            if i in self.cat_index_to_n_classes:
                num_classes = self.cat_index_to_n_classes[i]
                one_hot_columns = data[:, i:i + num_classes]
                # Get the categorical labels from one-hot encoded columns
                categorical_col = argmax(one_hot_columns, dim=1)
                # Concatenate the categorical labels back into the result tensor
                data = cat([data[:, :i], categorical_col.unsqueeze(1), data[:, i + num_classes:]], dim=1)

        return data
