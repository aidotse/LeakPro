"""Module that contains utility functions that are used in the attack classes."""
from typing import Any, Dict, List, Self

import numpy as np
from scipy.stats import norm

from .attack_objects import AttackObjects


class AttackUtils:
    """Utility class for the attacks."""

    def __init__(self:Self, attack_objects: AttackObjects)->None:
        """Initialize the AttackUtils class."""
        self.attack_objects = attack_objects

    def flatten_array(self:Self, arr: List[Any]) -> np.ndarray:
        """Recursively flatten a list of lists.

        Utility function that recursively flattens a list of lists.
        Each element in the list can be another list, tuple, set, or np.ndarray,
        and can have variable sizes.

        Args:
        ----
            arr: List of lists

        Returns:
        -------
            Flattened 1D np.ndarray version of arr.

        """
        flat_array = []
        for item in arr:
            if isinstance(item, (list, tuple, set, np.ndarray)):
                flat_array.extend(self.flatten_array(item))
            else:
                flat_array.append(item)
        return np.array(flat_array)

    def default_quantile() -> np.ndarray:
        """Return the default fprs.

        Returns
        -------
            arr: Numpy array, indicating the default fprs

        """
        return np.logspace(-5, 0, 100)

    def prepare_attack_dataset(self:Self, configs: dict) -> Dict[str, np.ndarray]:
        """Prepare the attack dataset based on the provided configurations.

        Args:
        ----
            configs: Dictionary containing the configurations for preparing the attack dataset.

        Returns:
        -------
            Dictionary containing the audit indices for the attack dataset.

        """
        audit_size = int(
            configs["data"]["f_audit"] * self.attack_objects.population_size
        )
        audit_index = self.sample_dataset_no_overlap(audit_size)
        return {"audit_indices": audit_index}

    def sample_dataset_uniformly(self:Self, size: float) -> np.ndarray:
        """Sample the dataset uniformly.

        Args:
        ----
            size: The size of the dataset to sample.

        Returns:
        -------
            np.ndarray: The selected indices of the dataset.

        """
        all_index = np.arange(self.attack_objects.population_size)
        if size <= len(all_index):
            selected_index = np.random.choice(all_index, size, replace=False)
        return selected_index

    def sample_dataset_no_overlap(self:Self, size: float) -> np.ndarray:
        """Sample the dataset without overlap.

        Args:
        ----
            size: The size of the dataset to sample.

        Returns:
        -------
            np.ndarray: The selected indices of the dataset.

        """
        all_index = np.arange(self.attack_objects.population_size)
        used_index = np.concatenate(
            (
                self.attack_objects.train_test_dataset["train_indices"],
                self.attack_objects.train_test_dataset["test_indices"],
            ),
            axis=0,
        )
        selected_index = np.setdiff1d(all_index, used_index, assume_unique=True)
        if size <= len(selected_index):
            selected_index = np.random.choice(selected_index, size, replace=False)
        else:
            raise ValueError("Not enough remaining data points.")
        return selected_index



    def threshold_func(self:Self, distribution: List[float], alpha: List[float], **kwargs: Dict[str, Any]) -> float:
        """Return the threshold as the alpha quantile of the provided distribution.

        Args:
        ----
            distribution: Sequence of values that form the distribution from which
            the threshold is computed.
            alpha: Quantile value that will be used to obtain the threshold from the
                distribution.
            **kwargs: Additional keyword arguments.

        Returns:
        -------
            threshold: alpha quantile of the provided distribution.

        """
        return np.quantile(distribution, q=alpha, interpolation="lower", **kwargs)

    ########################################################################################################################
    # HYPOTHESIS TEST: LINEAR INTERPOLATION THRESHOLDING
    ########################################################################################################################

    def linear_itp_threshold_func(
        self:Self,
        distribution: List[float],
        alpha: List[float],
        **kwargs: Dict[str, Any],
    ) -> float:
        """Return the threshold as the alpha quantile of a linear interpolation curve fit over the provided distribution.

        Args:
        ----
            distribution: Sequence of values that form the distribution from which
                the threshold is computed. (Here we only consider positive signal values.)
            alpha: Quantile value that will be used to obtain the threshold from the
                distribution.
            **kwargs: Additional keyword arguments.

        Returns:
        -------
            threshold: alpha quantile of the provided distribution.

        """
        if len(distribution.shape) > 1:
            # for reference attacks
            threshold = np.quantile(
                distribution, q=alpha[1:-1], method="linear", axis=1, **kwargs
            )
            threshold = np.concatenate(
                [
                    threshold,
                    np.repeat(distribution.max() + 1e-4, distribution.shape[0]).reshape(
                        1, -1
                    ),
                ],
                axis=0,
            )
            threshold = np.concatenate(
                [
                    np.repeat(distribution.min() - 1e-4, distribution.shape[0]).reshape(
                        1, -1
                    ),
                    threshold,
                ],
                axis=0,
            )

        else:
            threshold = np.quantile(
                distribution, q=alpha[1:-1], method="linear", **kwargs
            )
            threshold = np.concatenate(
                [
                    np.array(distribution.min() - 1e-4).reshape(-1),
                    threshold,
                    np.array(distribution.max() + 1e-4).reshape(-1),
                ],
                axis=0,
            )

        return threshold

    ########################################################################################################################
    # HYPOTHESIS TEST: LOGIT RESCALE THRESHOLDING
    ########################################################################################################################
    def logit_rescale_threshold_func(
        self:Self,
        distribution: List[float],
        alpha: List[float]
    ) -> float:
        """Return the threshold as the alpha quantile of a Gaussian fit over logit rescaling transform.

        Args:
        ----
            distribution: Sequence of values that form the distribution from which
                the threshold is computed. (Here we only consider positive signal values.)
            alpha: Quantile value that will be used to obtain the threshold from the
                distribution.

        Returns:
        -------
            threshold: alpha quantile of the provided distribution.

        """
        distribution = distribution + 0.000001  # avoid nan
        distribution = np.log(
            np.divide(np.exp(-distribution), (1 - np.exp(-distribution)))
        )


        if len(distribution.shape) > 1:
            parameters = np.array(
                [norm.fit(distribution[i]) for i in range(distribution.shape[0])]
            )
            num_threshold = alpha.shape[0]
            num_points = distribution.shape[0]
            loc = parameters[:, 0].reshape(-1, 1).repeat(num_threshold, 1)
            scale = parameters[:, 1].reshape(-1, 1).repeat(num_threshold, 1)
            alpha = np.array(alpha).reshape(-1, 1).repeat(num_points, 1)
            threshold = norm.ppf(1 - np.array(alpha), loc=loc.T, scale=scale.T)
        else:
            loc, scale = norm.fit(distribution)
            threshold = norm.ppf(1 - np.array(alpha), loc=loc, scale=scale)

        return np.log(np.exp(threshold) + 1) - threshold

    ########################################################################################################################
    # HYPOTHESIS TEST: GAUSSIAN THRESHOLDING
    ########################################################################################################################
    def gaussian_threshold_func(
        self:Self,
        distribution: List[float],
        alpha: List[float]
    ) -> float:
        """Return the threshold as the alpha quantile of a Gaussian curve fit over the provided distribution.

        Args:
        ----
            distribution: Sequence of values that form the distribution from which
                the threshold is computed.
            alpha: Quantile value that will be used to obtain the threshold from the
                distribution.

        Returns:
        -------
            threshold: alpha quantile of the provided distribution.

        """
        if len(distribution.shape) > 1:
            parameters = np.array(
                [norm.fit(distribution[i]) for i in range(distribution.shape[0])]
            )
            num_threshold = alpha.shape[0]
            num_points = distribution.shape[0]
            loc = parameters[:, 0].reshape(-1, 1).repeat(num_threshold, 1)
            scale = parameters[:, 1].reshape(-1, 1).repeat(num_threshold, 1)
            alpha = np.array(alpha).reshape(-1, 1).repeat(num_points, 1)
            threshold = norm.ppf(1 - np.array(alpha), loc=loc.T, scale=scale.T)
        else:
            loc, scale = norm.fit(distribution)
            threshold = norm.ppf(alpha, loc=loc, scale=scale)
        return threshold

