"""Module that contains the abstract class for constructing and performing a membership inference attack on a target."""

from abc import ABC, abstractmethod
from logging import Logger

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from leakpro.import_helper import List, Self, Union
from leakpro.metrics.attack_result import AttackResult
from leakpro.signals.signal import ModelLogits, ModelRescaledLogits

########################################################################################################################
# METRIC CLASS
########################################################################################################################


class AbstractMIA(ABC):
    """Interface to construct and perform a membership inference attack on a target model and dataset.

    This serves as a guideline for implementing a metric to be used for measuring the privacy leakage of a target model.
    """

    def __init__(
        self:Self,
        population: np.ndarray,
        audit_dataset: dict,
        target_model: nn.Module,
        logger:Logger
    )->None:
        """Initialize the AttackAbstract class.

        Args:
        ----
            population (np.ndarray): The population used for the attack.
            audit_dataset (dict): The audit dataset used for the attack.
            target_model (nn.Module): The target model used for the attack.
            logger (Logger): The logger used for logging.

        """
        self._population = population
        self._population_size = len(population)
        self._target_model = target_model
        self._audit_dataset = audit_dataset
        self.skip_indices = np.zeros(len(audit_dataset["data"]), dtype=bool)
        self.logger = logger
        self.signal_data = []

    @property
    def population(self:Self)-> List:
        """Get the population used for the attack.

        Returns
        -------
        List: The population used for the attack.

        """
        return self._population

    @property
    def population_size(self:Self)-> int:
        """Get the size of the population used for the attack.

        Returns
        -------
        int: The size of the population used for the attack.

        """
        return self._population_size

    @property
    def target_model(self:Self)-> Union[Self, List[Self] ]:
        """Get the target model used for the attack.

        Returns
        -------
        Union[Self, List[Self]]: The target model used for the attack.

        """
        return self._target_model

    @property
    def audit_dataset(self:Self)-> Self:
        """Get the audit dataset used for the attack.

        Returns
        -------
        Self: The audit dataset used for the attack.

        """
        return self._audit_dataset

    @property
    def train_indices(self:Self)-> np.ndarray:
        """Get the training indices of the audit dataset.

        Returns
        -------
        np.ndarray: The training indices of the audit dataset.

        """
        train_indices = self._audit_dataset["in_members"]
        return self._audit_dataset["data"][train_indices]


    @property
    def test_indices(self:Self)-> np.ndarray:
        """Get the test indices of the audit dataset.

        Returns
        -------
        np.ndarray: The test indices of the audit dataset.

        """
        test_indices = self._audit_dataset["out_members"]
        return self._audit_dataset["data"][test_indices]

    @abstractmethod
    def _configure_attack(self:Self, configs:dict)->None:
        """Configure the attack.

        Args:
        ----
            configs (dict): The configurations for the attack.

        """
        pass

    def _validate_config(self: Self, name: str, value: float, min_val: float, max_val: float) -> None:
        if not (min_val <= value <= (max_val if max_val is not None else value)):
            raise ValueError(f"{name} must be between {min_val} and {max_val}")

    @abstractmethod
    def description(self:Self) -> dict:
        """Return a description of the attack.

        Returns
        -------
        dict: A dictionary containing the reference, summary, and detailed description of the attack.

        """
        pass

    @abstractmethod
    def prepare_attack(self:Self) -> None:
        """Method that handles all computation related to the attack dataset."""
        pass

    @abstractmethod
    def run_attack(self:Self) -> Union[AttackResult, List[AttackResult]]:
        """Run the metric on the target model and dataset. This method handles all the computations related to the audit dataset.

        Args:
        ----
            fpr_tolerance_rate_list (optional): List of FPR tolerance values that may be used by the threshold function
                to compute the attack threshold for the metric.

        Returns:
        -------
            Result(s) of the metric.

        """
        pass

    def _memorization(self:Self):
        """Run memorization score enhancement.

        Memorization enhances the attack performance by only inlucing vulnerable data points

        """

        self.logger.info("Preparing memorization")
        if not self.online:
            self.logger.info("Using the offline version of the attack we make some assumptions in the absence of IN-models")

        if self.signal.__class__.__name__ == "ModelLogits":
            logits = self.shadow_models_logits
            target_logits = self.target_logits
        else:
            logits_function = ModelLogits()
            logits = np.swapaxes(logits_function(self.shadow_models, self.audit_data), 0, 1)
            target_logits = np.swapaxes(logits_function([self.target_model], self.audit_data), 0, 1)

        self.logger.info("Calculating memorization")

        # Initialize memorization score
        self.memorization_score = np.zeros(len(logits), dtype=float)
        logits = self.softmax_logits(logits)
        if self.online:
            for i, (logit, mask, label) in tqdm(enumerate(zip(logits, self.in_indices_mask, self.audit_data._labels))):
                self.memorization_score[i] = np.mean(logit[mask, label]) - np.mean(logit[~mask, label])
        else:
            for i, (logit, target_logit, label) in tqdm(enumerate(zip(logits, target_logits, self.audit_data._labels))):
                self.memorization_score[i] = np.mean(target_logit[label]) - np.mean(logit[~mask, label])

        self.privacy_score = self._privacy_score()
        
        mem_mask, privacy_mask = self.adjust_memorization_mask()

        self.skip_indices = (self.skip_indices | mem_mask) | privacy_mask

    def _privacy_score(self:Self) -> np.ndarray:
        """Run memorization score enhancement.

        Privacy score enhances the attack performance by only inlucing vulnerable data points

        """
        self.logger.info("Preparing privacy score")
        if self.signal.__class__.__name__ == "ModelRescaledLogits":
            logits = self.shadow_models_logits
            target_logits = self.target_logits
        else:
            logits_function = ModelRescaledLogits()
            logits = logits_function(self.shadow_models, self.audit_data)
            target_logits = logits_function([self.target_model], self.audit_data).squeeze()

        self.logger.info("Calculating privacy score")
        privacy_score = []

        if len(self.shadow_models) < 64:
            in_std, out_std = np.std(logits[self.in_indices_mask].flatten()), np.std(logits[~self.in_indices_mask].flatten())

        for i, (logit, target_logit, mask) in tqdm(enumerate(zip(logits, target_logits, self.in_indices_mask))):
            in_mean, out_mean = np.mean(logit[mask]), np.mean(logit[~mask])
            
            if len(self.shadow_models) >= 64:
                in_std, out_std = np.std(logit[mask]), np.std(logit[mask])

            if self.online:
                privacy_score.append(np.abs(in_mean-out_mean)/(in_std+out_std+1e-30))
            else:
                privacy_score.append(np.abs(target_logit-out_mean)/(2*out_std+1e-30))

        return np.asarray(privacy_score)

    def softmax_logits(self:Self, logits: np.ndarray) -> np.ndarray:
        """Rescale logits to (0, 1).

        Args:
        ----
            logits ( len(dataset) x ... x nb_classes ): Logits to be rescaled.

        """
        logits = torch.from_numpy(logits)
        logits = logits - torch.max(logits, dim=-1, keepdim=True).values
        logits = torch.exp(logits)
        logits = logits/torch.sum(logits, dim=-1, keepdim=True)
        return logits.numpy()

    def adjust_memorization_mask(self:Self):
        """Adjust thesholds to achieve the desired percentile most vulnerable dataponts

        """
        audit_dataset_len = len(self.target_logits)
        if audit_dataset_len*(1-self.memorization_threshold) < 30:
            self.logger.info("Trying to audit <30 datapoints, adjusting to 30 datapoints")
            self.memorization_threshold = (1-30/audit_dataset_len)

        mem_thrshld = 0.8
        priv_thrshld = 2.0
        # Adjust initial thresholds if they are set too high
        while np.count_nonzero((self.memorization_score < mem_thrshld) | (self.privacy_score < priv_thrshld) | self.skip_indices)/audit_dataset_len > self.memorization_threshold:
            mem_thrshld = mem_thrshld/2
            priv_thrshld = priv_thrshld/2

        # Find the thresholds corresponding to the percentile set 
        while np.count_nonzero((self.memorization_score < mem_thrshld) | (self.privacy_score < priv_thrshld) | self.skip_indices)/audit_dataset_len < self.memorization_threshold:
            mem_thrshld = 1 - (1 - mem_thrshld)/(1.001)
            priv_thrshld = priv_thrshld*1.001

        return self.memorization_score < mem_thrshld, self.privacy_score < priv_thrshld
# 