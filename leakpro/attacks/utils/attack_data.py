"""Module providing a function to get attack data for the attack models."""
from logging import Logger

import numpy as np


def get_attack_data(
    population_size: int,
    train_indices: list,
    test_indices: list,
    sample_with_test_data: bool,
    logger:Logger
) -> np.ndarray:
    """Function to get attack data for the attack models.

    Args:
    ----
        population_size (int): The size of the population.
        train_indices (list): The indices of the training data.
        test_indices (list): The indices of the test data.
        sample_with_test_data (bool): Flag indicating whether to include test data.
        logger (Logger): The logger object for logging.

    Returns:
    -------
        np.ndarray: The selected attack data indices.

    """
    if population_size <= 0:
        raise ValueError("Population size must be greater than 0.")
    if train_indices is None:
        raise ValueError("Train indices must be provided.")
    if test_indices is None:
        raise ValueError("Test indices must be provided.")

    all_index = np.arange(population_size)
    not_allowed_indices = train_indices if sample_with_test_data else np.concatenate((train_indices, test_indices), axis=0)
    available_index = np.setdiff1d(all_index, not_allowed_indices, assume_unique=True)
    attack_data_size = len(available_index)

    logger.info(f"Selecting {attack_data_size} attack data points out of {len(available_index)} available data points.")

    if attack_data_size <= len(available_index):
        selected_index = np.random.choice(available_index, attack_data_size, replace=False)
    else:
        raise ValueError("Not enough remaining data points.")
    return selected_index
