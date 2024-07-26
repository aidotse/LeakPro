"""Module with functions for preparing the dataset for training the target models."""

import numpy as np
from sklearn.model_selection import train_test_split


def prepare_fl_client_dataset(dataset_size: int, configs: dict) -> list:
    """Splits the data into client data and remaining data into train and test indicies."""
    all_index = np.arange(dataset_size)
    client_size = configs["gia_settings"]["client_batch_size"] * configs["gia_settings"]["num_train_batches"]
    rest_size = dataset_size - client_size  # rest of data
    selected_index = np.random.choice(all_index, client_size + rest_size, replace=False)
    client_index, rest_index = train_test_split(selected_index, test_size=rest_size)

    return client_index, rest_index
