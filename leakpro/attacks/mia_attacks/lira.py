"""Implementation of the LiRA attack."""

import numpy as np
from scipy.stats import norm
from tqdm import tqdm

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.boosting import Memorization
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.metrics.attack_result import CombinedMetricResult
from leakpro.signals.signal import ModelRescaledLogits
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class AttackLiRA(AbstractMIA):
    """Implementation of the LiRA attack."""

    def __init__(self:Self,
                 handler: AbstractInputHandler,
                 configs: dict
                 ) -> None:
        """Initialize the LiRA attack.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.
            configs (dict): Configuration parameters for the attack.

        """
        # Initializes the parent metric
        super().__init__(handler)

        self.signal = ModelRescaledLogits()
        self._configure_attack(configs)

    def _configure_attack(self:Self, configs: dict) -> None:
        """Configure the RMIA attack.

        Args:
        ----
            configs (dict): Configuration parameters for the attack.

        """
        self.shadow_models = []
        self.num_shadow_models = configs.get("num_shadow_models", 64)
        self.online = configs.get("online", False)
        self.training_data_fraction = configs.get("training_data_fraction", 0.5)
        self.include_train_data = configs.get("include_train_data", self.online)
        self.include_test_data = configs.get("include_test_data", self.online)

        # Memorization config
        # Activate memorization
        self.memorization = configs.get("memorization", False)
        # Set True to filter based on privacy score aswell as memorization score
        self.use_privacy_score = configs.get("use_privacy_score", False)
        # Set percentile for most vulnerable data points, use 0.0 for paper thresholds
        self.memorization_threshold = configs.get("memorization_threshold", 0.8)
        # Set minimum allowed audit points after memorization
        self.min_num_memorization_audit_points = configs.get("min_num_memorization_audit_points", 10)
        # Directly set number of most vulnerable audit data points (Overrides "memorization_threshold" )
        self.num_memorization_audit_points = configs.get("num_memorization_audit_points", 0)

        # LiRA specific
        # Determine which variance estimation method to use [carlini, individual_carlini]
        self.var_calculation = configs.get("var_calculation", "carlini")

        # Define the validation dictionary as: {parameter_name: (parameter, min_value, max_value)}
        validation_dict = {
            "num_shadow_models": (self.num_shadow_models, 1, None),
            "training_data_fraction": (self.training_data_fraction, 0, 1),
            "memorization_threshold": (self.memorization_threshold, 0, 1),
            "min_num_memorization_audit_points": (self.min_num_memorization_audit_points, 1, 1_000_000),
            "num_memorization_audit_points": (self.num_memorization_audit_points, 0, 1_000_000),
        }

        # Validate parameters
        for param_name, (param_value, min_val, max_val) in validation_dict.items():
            self._validate_config(param_name, param_value, min_val, max_val)

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
            audit_data_indices = self.audit_dataset["data"][mask]
            self.in_indices_masks = self.in_indices_masks[mask, :]

            # Filter IN and OUT members
            self.in_members = np.arange(np.sum(mask[self.audit_dataset["in_members"]]))
            num_out_members = np.sum(mask[self.audit_dataset["out_members"]])
            self.out_members = np.arange(len(self.in_members), len(self.in_members) + num_out_members)

            assert len(audit_data_indices) == len(self.in_members) + len(self.out_members)

            if len(audit_data_indices) == 0:
                raise ValueError("No points in the audit dataset are used for the shadow models")

        else:
            audit_data_indices = self.audit_dataset["data"]
            self.in_members = self.audit_dataset["in_members"]
            self.out_members = self.audit_dataset["out_members"]

        # Check offline attack for possible IN- sample(s)
        if not self.online:
            count_in_samples = np.count_nonzero(self.in_indices_masks)
            if count_in_samples > 0:
                logger.info(f"Some shadow model(s) contains {count_in_samples} IN samples in total for the model(s)")
                logger.info("This is not an offline attack!")

        self.batch_size = len(audit_data_indices)
        logger.info(f"Calculating the logits for all {self.num_shadow_models} shadow models")
        self.shadow_models_logits = np.swapaxes(self.signal(self.shadow_models, self.handler, audit_data_indices,\
                                                            self.batch_size), 0, 1)

        # Calculate logits for the target model
        logger.info("Calculating the logits for the target model")
        self.target_logits = np.swapaxes(self.signal([self.target_model], self.handler, audit_data_indices, self.batch_size),\
                                        0, 1).squeeze()

        # Using Memorizationg boosting
        if self.memorization:

            # Prepare for memorization
            org_audit_data_length = self.audit_dataset["data"].size
            audit_data_indices = self.audit_dataset["data"][mask] if self.online else self.audit_dataset["data"]
            audit_data_labels = self.handler.get_labels(audit_data_indices)

            logger.info("Running memorization")
            memorization = Memorization(
                self.use_privacy_score,
                self.memorization_threshold,
                self.min_num_memorization_audit_points,
                self.num_memorization_audit_points,
                self.in_indices_masks,
                self.shadow_models,
                self.target_model,
                audit_data_indices,
                audit_data_labels,
                org_audit_data_length,
                self.handler,
                self.online,
                self.batch_size,
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

    def run_attack(self:Self) -> CombinedMetricResult:
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
        return CombinedMetricResult(
            predicted_labels=predictions,
            true_labels=true_labels,
            predictions_proba=None,  # Note: Direct probability predictions are not computed here
            signal_values=signal_values,
            # masks = masks
        )
