"""Implementation of the LiRA attack."""

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator
from scipy.stats import norm
from tqdm import tqdm

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.boosting import Memorization
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.metrics.attack_result import MIAResult
from leakpro.signals.signal import ModelRescaledLogits
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class AttackLiRA(AbstractMIA):
    """Implementation of the LiRA attack."""

    class AttackConfig(BaseModel):
        """Configuration for the LiRA attack."""

        num_shadow_models: int = Field(default=1, ge=1, description="Number of shadow models")
        training_data_fraction: float = Field(default=0.5, ge=0.0, le=1.0, description="Part of available attack data to use for shadow models")  # noqa: E501
        online: bool = Field(default=False, description="Online vs offline attack")
        eval_batch_size: int = Field(default=32, ge=1, description="Batch size for evaluation")
        var_calculation: Literal["carlini", "individual_carlini", "fixed"] = Field(default="carlini", description="Variance estimation method to use [carlini, individual_carlini, fixed]")  # noqa: E501
        # memorization boosting
        memorization: bool = Field(default=False, description="Activate memorization boosting")
        use_privacy_score: bool = Field(default=False, description="Filter based on privacy score aswell as memorization score")
        memorization_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Set percentile for most vulnerable data points, use 0.0 for paper thresholds")  # noqa: E501
        min_num_memorization_audit_points: int = Field(default=10, ge=1, description="Set minimum allowed audit points after memorization")  # noqa: E501
        num_memorization_audit_points: int = Field(default=0, ge=0, description="Directly set number of most vulnerable audit data points (Overrides 'memorization_threshold')")  # noqa: E501

        @model_validator(mode="after")
        def check_num_shadow_models_if_online(self) -> Self:
            """Check if the number of shadow models is at least 2 when online is True.

            Returns
            -------
                Config: The attack configuration.

            Raises
            ------
                ValueError: If online is True and the number of shadow models is less than 2.

            """
            if self.online and self.num_shadow_models < 2:
                raise ValueError("When online is True, num_shadow_models must be >= 2")
            return self

    def __init__(self:Self,
                 handler: MIAHandler,
                 configs: dict
                 ) -> None:
        """Initialize the LiRA attack.

        Args:
        ----
            handler (MIAHandler): The input handler object.
            configs (dict): Configuration parameters for the attack.

        """
        self.configs = self.AttackConfig() if configs is None else self.AttackConfig(**configs)

        # Initializes the parent metric
        super().__init__(handler)

        # Assign the configuration parameters to the object
        for key, value in self.configs.model_dump().items():
            setattr(self, key, value)

        if self.online is False and self.population_size == self.audit_size:
            raise ValueError("The audit dataset is the same size as the population dataset. \
                    There is no data left for the shadow models.")

        self.shadow_models = []
        self.signal = ModelRescaledLogits()

    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "Likelihood Ratio Attack"

        reference_str = "Carlini N, et al. Membership Inference Attacks From First Principles"

        summary_str = "LiRA is a membership inference attack based on rescaled logits of a black-box model"

        detailed_str = "The attack is executed according to: \
            1. A fraction of the target model dataset is sampled to be included (in-) or excluded (out-) \
            from the shadow model training dataset. \
            2. The rescaled logits are used to estimate Gaussian distributions for in and out members \
            3. The thresholds are used to classify in-members and out-members. \
            4. The attack is evaluated on an audit dataset to determine the attack performance."

        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }

    def prepare_attack(self:Self)->None:
        """Prepares data to obtain metric on the target model and dataset, using signals computed on the auxiliary model/dataset.

        It selects a balanced subset of data samples from in-group and out-group members
        of the audit dataset, prepares the data for evaluation, and computes the logits
        for both shadow models and the target model.
        """

        # Fixed variance is used when the number of shadow models is below 32 (64, IN and OUT models)
        #       from (Membership Inference Attacks From First Principles)
        self.fix_var_threshold = 32

        self.attack_data_indices = self.sample_indices_from_population(include_train_indices = self.online,
                                                                       include_test_indices = self.online)

        self.shadow_model_indices = ShadowModelHandler().create_shadow_models(num_models = self.num_shadow_models,
                                                                              shadow_population =  self.attack_data_indices,
                                                                              training_fraction = self.training_data_fraction,
                                                                              online = self.online)

        self.shadow_models, _ = ShadowModelHandler().get_shadow_models(self.shadow_model_indices)

        logger.info("Create masks for all IN and OUT samples")
        self.in_indices_masks = ShadowModelHandler().get_in_indices_mask(self.shadow_model_indices, self.audit_dataset["data"])

        if self.online:
            # Exclude all audit points that have either no IN or OUT samples
            num_shadow_models_seen_points = np.sum(self.in_indices_masks, axis=1)
            mask = (num_shadow_models_seen_points > 0) & (num_shadow_models_seen_points < self.num_shadow_models)

            # Filter the audit data
            self.audit_data_indices = self.audit_dataset["data"][mask]
            self.in_indices_masks = self.in_indices_masks[mask, :]

            # Filter IN and OUT members
            self.in_members = np.arange(np.sum(mask[self.audit_dataset["in_members"]]))
            num_out_members = np.sum(mask[self.audit_dataset["out_members"]])
            self.out_members = np.arange(len(self.in_members), len(self.in_members) + num_out_members)

            assert len(self.audit_data_indices) == len(self.in_members) + len(self.out_members)

            if len(self.audit_data_indices) == 0:
                raise ValueError("No points in the audit dataset are used for the shadow models")

        else:
            self.audit_data_indices = self.audit_dataset["data"]
            self.in_members = self.audit_dataset["in_members"]
            self.out_members = self.audit_dataset["out_members"]

        # Check offline attack for possible IN- sample(s)
        if not self.online:
            count_in_samples = np.count_nonzero(self.in_indices_masks)
            if count_in_samples > 0:
                logger.info(f"Some shadow model(s) contains {count_in_samples} IN samples in total for the model(s)")
                logger.info("This is not an offline attack!")

        logger.info(f"Calculating the logits for all {self.num_shadow_models} shadow models")
        self.shadow_models_logits = np.swapaxes(self.signal(self.shadow_models,
                                                            self.handler,
                                                            self.audit_data_indices), 0, 1)

        # Calculate logits for the target model
        logger.info("Calculating the logits for the target model")
        self.target_logits = np.swapaxes(self.signal([self.target_model],
                                                     self.handler,
                                                     self.audit_data_indices), 0, 1).squeeze()

        # Using Memorizationg boosting
        if self.memorization:

            # Prepare for memorization
            org_audit_data_length = self.audit_data_indices.size
            audit_data_labels = self.handler.get_labels(self.audit_data_indices)

            logger.info("Running memorization")
            memorization = Memorization(
                self.use_privacy_score,
                self.memorization_threshold,
                self.min_num_memorization_audit_points,
                self.num_memorization_audit_points,
                self.in_indices_masks,
                self.shadow_models,
                self.target_model,
                self.audit_data_indices,
                audit_data_labels,
                org_audit_data_length,
                self.handler,
                self.online,
            )
            memorization_mask, _, _ = memorization.run()

            # Filter masks
            self.in_indices_masks = self.in_indices_masks[memorization_mask, :]

            # Filter IN and OUT members
            self.in_members = np.arange(np.sum(memorization_mask[self.in_members]))
            num_out_members = np.sum(memorization_mask[self.out_members])
            self.out_members = np.arange(len(self.in_members), len(self.in_members) + num_out_members)

            assert len(self.out_members) > 0
            assert len(self.in_members) > 0

            # Filter logits
            self.shadow_models_logits = self.shadow_models_logits[memorization_mask, :]
            self.target_logits = self.target_logits[memorization_mask]

    def get_std(self:Self, logits: list, mask: list, is_in: bool, var_calculation: str) -> np.ndarray:
        """A function to define what method to use for calculating variance for LiRA."""

        # Fixed/Global variance calculation.
        if var_calculation == "fixed":
            return self._fixed_variance(logits, mask, is_in)

        # Variance calculation as in the paper ( Membership Inference Attacks From First Principles )
        if var_calculation == "carlini":
            return self._carlini_variance(logits, mask, is_in)

        # Variance calculation as in the paper ( Membership Inference Attacks From First Principles )
        #   but check IN and OUT samples individualy
        if var_calculation == "individual_carlini":
            return self._individual_carlini(logits, mask, is_in)

        return np.array([None])

    def _fixed_variance(self:Self, logits: list, mask: list, is_in: bool) -> np.ndarray:
        if is_in and not self.online:
            return np.array([None])
        return np.std(logits[mask])

    def _carlini_variance(self:Self, logits: list, mask: list, is_in: bool) -> np.ndarray:
        if self.num_shadow_models >= self.fix_var_threshold*2:
                return np.std(logits[mask])
        if is_in:
            return self.fixed_in_std
        return self.fixed_out_std

    def _individual_carlini(self:Self, logits: list, mask: list, is_in: bool) -> np.ndarray:
        if np.count_nonzero(mask) >= self.fix_var_threshold:
            return np.std(logits[mask])
        if is_in:
            return self.fixed_in_std
        return self.fixed_out_std

    def run_attack(self:Self) -> MIAResult:
        """Runs the attack on the target model and dataset and assess privacy risks or data leakage.

        This method evaluates how the target model's output (logits) for a specific dataset
        compares to the output of shadow models to determine if the dataset was part of the
        model's training data or not.

        Returns
        -------
        Result(s) of the metric. An object containing the metric results, including predictions,
        true labels, and signal values.

        """
        n_audit_samples = self.shadow_models_logits.shape[0]
        score = np.zeros(n_audit_samples)  # List to hold the computed probability scores for each sample

        self.fixed_in_std = self.get_std(self.shadow_models_logits.flatten(), self.in_indices_masks.flatten(), True, "fixed")
        self.fixed_out_std = self.get_std(self.shadow_models_logits.flatten(), (~self.in_indices_masks).flatten(), False, "fixed")

        # Iterate over and extract logits for IN and OUT shadow models for each audit sample
        for i, (shadow_models_logits, mask) in tqdm(enumerate(zip(self.shadow_models_logits, self.in_indices_masks)),
                                                    total=len(self.shadow_models_logits),
                                                    desc="Processing audit samples"):

            # Calculate the mean for OUT shadow model logits
            out_mean = np.mean(shadow_models_logits[~mask])
            out_std = self.get_std(shadow_models_logits, ~mask, False, self.var_calculation)

            # Get the logit from the target model for the current sample
            target_logit = self.target_logits[i]

            # Calculate the log probability density function value
            pr_out = -norm.logpdf(target_logit, out_mean, out_std + 1e-30)

            if self.online:
                in_mean = np.mean(shadow_models_logits[mask])
                in_std = self.get_std(shadow_models_logits, mask, True, self.var_calculation)

                pr_in = -norm.logpdf(target_logit, in_mean, in_std + 1e-30)
            else:
                pr_in = 0

            score[i] = (pr_in - pr_out)  # Append the calculated probability density value to the score list
            if np.isnan(score[i]):
                raise ValueError("Score is NaN")

        # Generate thresholds based on the range of computed scores for decision boundaries
        self.thresholds = np.linspace(np.min(score), np.max(score), 1000)

        # Split the score array into two parts based on membership: in (training) and out (non-training)
        self.in_member_signals = score[self.in_members].reshape(-1,1)  # Scores for known training data members
        self.out_member_signals = score[self.out_members].reshape(-1,1)  # Scores for non-training data members

        # Create prediction matrices by comparing each score against all thresholds
        member_preds = np.less(self.in_member_signals, self.thresholds).T  # Predictions for training data members
        non_member_preds = np.less(self.out_member_signals, self.thresholds).T  # Predictions for non-members

        # Concatenate the prediction results for a full dataset prediction
        predictions = np.concatenate([member_preds, non_member_preds], axis=1)

        # Prepare true labels array, marking 1 for training data and 0 for non-training data
        true_labels = np.concatenate(
            [np.ones(len(self.in_member_signals)), np.zeros(len(self.out_member_signals))]
        )

        # Combine all signal values for further analysis
        signal_values = np.concatenate([self.in_member_signals, self.out_member_signals])

        # Return a result object containing predictions, true labels, and the signal values for further evaluation
        return MIAResult(
            predicted_labels=predictions,
            true_labels=true_labels,
            predictions_proba=None,
            signal_values=signal_values,
            audit_indices=self.audit_data_indices,
        )
