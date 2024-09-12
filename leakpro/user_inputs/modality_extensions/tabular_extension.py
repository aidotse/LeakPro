"""TabularExtension class for handling tabular data with one-hot encoding and decoding."""

import numpy as np
import pandas as pd

from leakpro.import_helper import Self
from leakpro.utils.logger import logger


class TabularExtension:
    """Class for handling tabular data with one-hot encoding and decoding."""

    def check_data(self:Self) -> None:
        """Check if the data is a pandas DataFrame."""

        data = self.get_dataset(dataset_indices = 0)
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame.")
        self.cat_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
        self.cont_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        logger.info(f"Continuous cols: {self.cont_cols}")
        logger.info(f"Categorical cols: {self.cat_cols}")
    

    def one_hot_encode(self:Self, data:pd.DataFrame) -> pd.DataFrame:
        """One-hot encode all categorical columns in the DataFrame.

        Returns
        -------
            pd.DataFrame: The one-hot encoded DataFrame with all categorical columns encoded.

        """
        return pd.get_dummies(data, columns=self.cat_cols)

    def one_hot_to_categorical(self:Self, one_hot_data: pd.DataFrame) -> pd.DataFrame:
        """Convert one-hot encoded columns back to their original categorical values.

        Args:
        ----
            one_hot_data (pd.DataFrame): The one-hot encoded DataFrame.

        Returns:
        -------
            pd.DataFrame: The DataFrame with one-hot encoded columns converted back to categorical columns.

        """
        result = one_hot_data.copy()

        for column in self.cat_cols:
            # Find all columns that are related to this original categorical column
            one_hot_columns = [col for col in one_hot_data.columns if col.startswith(column + "_")]

            # Convert one-hot encoded columns back to the original categorical values
            result[column] = one_hot_data[one_hot_columns].idxmax(axis=1).apply(lambda x: x.split("_")[-1])

            # Drop the one-hot encoded columns
            result = result.drop(columns=one_hot_columns)

        return result
