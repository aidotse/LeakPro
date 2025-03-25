"""Module that contains the implementation of the attack P."""

import numpy as np
from pydantic import BaseModel, Field

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.threshold_computation import linear_itp_threshold_func
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.signals.signal import ModelLoss
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class AttackP(AbstractMIA):
    """Implementation of the P-attack."""

    class AttackConfig(BaseModel):
        """Configuration for the RMIA attack."""

        attack_data_fraction: float = Field(default=0.5, ge=0.0, le=1.0, description="Fraction of population to use for the attack") # noqa: E501

    def __init__(
        self:Self,
        handler: MIAHandler,
        configs: dict
    ) -> None:
        """Initialize the AttackP class.

        Args:
        ----
            handler (MIAHandler): The input handler object.
            configs (dict): A dictionary containing the attack configurations.

        """
        logger.info("Configuring the Population attack")
        self.configs = self.AttackConfig() if configs is None else self.AttackConfig(**configs)

        # Initializes the parent
        super().__init__(handler)

        for key, value in self.configs.model_dump().items():
            setattr(self, key, value)

        if self.population_size == self.audit_size:
            raise ValueError("The audit dataset is the same size as the population dataset. \
                    There is no data left to find the thresholds.")

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
        logger.info("Preparing attack data for training the Population attack")
        self.attack_data_indices = self.sample_indices_from_population(include_train_indices = False,
                                                                include_test_indices = False)

        # subsample the attack data based on the fraction
        logger.info(f"Subsampling attack data from {len(self.attack_data_indices)} points")
        n_points = int(self.attack_data_fraction * len(self.attack_data_indices))
        attack_data_indices = np.random.choice(self.attack_data_indices, n_points, replace=False)
        logger.info(f"Number of attack data points after subsampling: {len(attack_data_indices)}")

        # signals based on training dataset
        logger.info("Computing signals for the Population attack")
        self.attack_signal = np.array(self.signal([self.target_model], self.handler, attack_data_indices)).squeeze()

    def run_attack(self:Self) -> MIAResult:
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
        thresholds = self.hypothesis_test_func(self.attack_signal, self.quantiles)[:, np.newaxis]

        logger.info("Running the Population attack on the target model")
        # get the loss for the audit dataset
        audit_signal = np.array(self.signal([self.target_model], self.handler, self.audit_dataset["data"])).squeeze()

        # set true labels for being in the training dataset
        true_labels = np.concatenate([np.ones(len(self.audit_dataset["in_members"])),
                                      np.zeros(len(self.audit_dataset["out_members"]))])

        # compute the true positive, false positive, true negative, and false negative
        tp = np.zeros(len(thresholds))
        fp = np.zeros(len(thresholds))
        tn = np.zeros(len(thresholds))
        fn = np.zeros(len(thresholds))

        for i, threshold in enumerate(thresholds):
            # compute the signal values
            signal_values = np.less(audit_signal, threshold)
            tp[i] = np.sum(np.logical_and(signal_values, true_labels))
            fp[i] = np.sum(np.logical_and(signal_values, np.logical_not(true_labels)))
            tn[i] = np.sum(np.logical_and(np.logical_not(signal_values), np.logical_not(true_labels)))
            fn[i] = np.sum(np.logical_and(np.logical_not(signal_values), true_labels))

        logger.info("Attack completed")

        return MIAResult(true_membership = true_labels,
                         signal_values = audit_signal,
                         result_name = "P-attack",
                         tp_fp_tn_fn = (tp, fp, tn, fn))
