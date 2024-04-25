"""Module that contains the implementation of the attack P."""
from logging import Logger

import numpy as np
from torch import nn

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.attack_data import get_attack_data
from leakpro.attacks.utils.threshold_computation import linear_itp_threshold_func
from leakpro.import_helper import Self
from leakpro.metrics.attack_result import CombinedMetricResult
from leakpro.signals.signal import ModelLoss


class AttackP(AbstractMIA):
    """Implementation of the P-attack."""

    def __init__(
        self:Self,
        population: np.ndarray,
        audit_dataset: dict,
        target_model: nn.Module,
        logger:Logger,
        configs: dict
    ) -> None:
        """Initialize the AttackP class.

        Args:
        ----
            population (np.ndarray): The population data used for the attack.
            audit_dataset (dict): The audit dataset used for the attack.
            target_model (nn.Module): The target model to be attacked.
            logger (Logger): The logger object for logging.
            configs (dict): A dictionary containing the attack configurations.

        """
        # Initializes the parent metric
        super().__init__(population, audit_dataset, target_model, logger)

        self.f_attack_data_size = configs.get("data_fraction", 0.3)
        if self.f_attack_data_size < 0 or self.f_attack_data_size > 1:
            raise ValueError("data_fraction must be between 0 and 1")

        self.include_test_data = configs.get("include_test_data", True)

        self.signal = ModelLoss()
        self.hypothesis_test_func = linear_itp_threshold_func

    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "Population attack (P-attack)"

        reference_str = "Ye, Jiayuan, et al. Enhanced membership inference attacks against machine learning models. " \
                        "Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security. 2022."

        summary_str = "The Population attack (P-attack) is a membership inference attack based on the output loss of a black-box model."  # noqa: E501

        detailed_str = "The attack is executed according to: \
            1. A fraction of the population is sampled to create histograms of the output loss. \
            2. The histograms are used to compute thresholds for the output loss based on a given FPR \
            3. The thresholds are used to classify in-members and out-members. \
            4. The attack is evaluated on an audit dataset to determine the attack performance."

        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }


    def prepare_attack(self:Self) -> None:
        """Prepare data needed for running the metric on the target model and dataset."""
        # sample dataset to compute histogram
        self.logger.info("Preparing attack data for training the Population attack")
        self.attack_data_index = get_attack_data(
            self.population_size,
            self.train_indices,
            self.test_indices,
            self.include_test_data,
            self.logger
        )

        attack_data = self.population.subset(self.attack_data_index)
        # Load signals if they have been computed already; otherwise, compute and save them
        # signals based on training dataset
        self.attack_signal = np.array(self.signal([self.target_model], attack_data))

    def run_attack(self:Self) -> CombinedMetricResult:
        """Run the attack on the target model and dataset.

        Args:
        ----
            fpr_tolerance_rate_list (optional): List of FPR tolerance values that may be used by the threshold function
                to compute the attack threshold for the metric.

        Returns:
        -------
            Result(s) of the metric.

        """
        # map the threshold with the alpha
        self.quantiles = np.logspace(-5, 0, 100)
        # obtain the threshold values based on the reference dataset
        thresholds = self.hypothesis_test_func(self.attack_signal, self.quantiles).reshape(-1, 1)

        num_threshold = len(self.quantiles)

        self.logger.info("Running the Population attack on the target model")
        # get the loss for the audit dataset
        audit_data = self.population.subset(self.audit_dataset["data"])
        audit_signal = np.array(self.signal([self.target_model], audit_data)).squeeze()

        # pick out the in-members and out-members
        self.in_member_signals =  audit_signal[self.audit_dataset["in_members"]]
        self.out_member_signals = audit_signal[self.audit_dataset["out_members"]]

        # compute the signals for the in-members and out-members
        member_signals = (self.in_member_signals.reshape(-1, 1).repeat(num_threshold, 1).T)
        non_member_signals = (self.out_member_signals.reshape(-1, 1).repeat(num_threshold, 1).T)
        member_preds = np.less(member_signals, thresholds)
        non_member_preds = np.less(non_member_signals, thresholds)

        self.logger.info("Attack completed")

        # what does the attack predict on test and train dataset
        predictions = np.concatenate([member_preds, non_member_preds], axis=1)
        # set true labels for being in the training dataset
        true_labels = np.concatenate(
            [
                np.ones(len(self.in_member_signals)),
                np.zeros(len(self.out_member_signals)),
            ]
        )
        signal_values = np.concatenate(
            [self.in_member_signals, self.out_member_signals]
        )

        # compute ROC, TP, TN etc
        return CombinedMetricResult(
            predicted_labels=predictions,
            true_labels=true_labels,
            predictions_proba=None,
            signal_values=signal_values,
        )
