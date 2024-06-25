"""Implementation of the LiRA attack."""

import numpy as np
from scipy.stats import norm
from tqdm import tqdm

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.import_helper import Self
from leakpro.metrics.attack_result import CombinedMetricResult
from leakpro.signals.signal import ModelRescaledLogits
from leakpro.user_inputs.abstract_input_handler import AbstractInputHandler


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
        self.training_data_fraction = configs.get("training_data_fraction", 0.5)

        self.shadow_models = []
        self.num_shadow_models = configs.get("num_shadow_models", 64)
        self.exclude_logit_threshold = configs.get("exclude_logit_threshold", 1)

        self.online = configs.get("online", False)

        self.memorization = configs.get("memorization", False)
        self.memorization_threshold = configs.get("memorization_threshold", 0.5)
        self.privacy_score_threshold = configs.get("privacy_score_threshold", 1)

        self.include_train_data = configs.get("include_train_data", self.online)
        self.include_test_data = configs.get("include_test_data", self.online)

        # Define the validation dictionary as: {parameter_name: (parameter, min_value, max_value)}
        validation_dict = {
            "num_shadow_models": (self.num_shadow_models, 1, None),
            "exclude_logit_threshold": (self.exclude_logit_threshold, 0, int(self.num_shadow_models/2)),
            "training_data_fraction": (self.training_data_fraction, 0, 1),
            "memorization_threshold": (self.memorization_threshold, 0, 1),
            "privacy_score_threshold": (self.privacy_score_threshold, 0, 100),
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
        self.attack_data_indices = self.sample_indices_from_population(include_train_indices = self.online,
                                                                       include_test_indices = self.online)


        self.shadow_model_indices = ShadowModelHandler().create_shadow_models(num_models = self.num_shadow_models,
                                                                              shadow_population =  self.attack_data_indices,
                                                                              training_fraction = self.training_data_fraction,
                                                                              online = self.online)

        self.shadow_models, _ = ShadowModelHandler().get_shadow_models(self.shadow_model_indices)

        self.logger.info("Create masks for all IN samples")
        self.in_indices_mask = ShadowModelHandler().get_in_indices_mask(self.shadow_model_indices, self.audit_dataset["data"]).T

        # Check offline attack for possible IN- sample(s)
        if self.online:
            # filter out the points that no shadow model has seen and points that all shadow models have seen
            num_shadow_models_seen_points = np.sum(self.in_indices_mask, axis=0)
            # make sure that the audit points are included in the shadow model training (but not all)
            mask = (num_shadow_models_seen_points > 0) & (num_shadow_models_seen_points < self.num_shadow_models)

            # Select datapoints that are auditable
            audit_data_indices = self.audit_dataset["data"][mask]
            self.in_members = np.arange(np.sum(mask[self.audit_dataset["in_members"]]))
            # find out how many out-members survived the filtering
            num_out_members = np.sum(mask[self.audit_dataset["out_members"]])
            self.out_members = np.arange(len(self.in_members), len(self.in_members) + num_out_members)
            self.in_indices_mask = self.in_indices_mask[:,mask]

            assert len(audit_data_indices) == len(self.in_members) + len(self.out_members)

            if len(audit_data_indices) == 0:
                raise ValueError("No points in the audit dataset are used for the shadow models")
        else:
            audit_data_indices = self.audit_dataset["data"]
            self.in_members = self.audit_dataset["in_members"]
            self.out_members = self.audit_dataset["out_members"]

            count_in_samples = np.count_nonzero(self.in_indices_mask)
            if count_in_samples > 0:
                self.logger.info(f"Some shadow model(s) contains {count_in_samples} IN samples in total for the model(s)")
                self.logger.info("This is not an offline attack!")

        # Calculate logits for all shadow models
        self.logger.info(f"Calculating the logits for all {self.num_shadow_models} shadow models")
        self.shadow_models_logits = np.array(self.signal(self.shadow_models, self.handler, audit_data_indices))

        # Calculate logits for the target model
        self.logger.info("Calculating the logits for the target model")
        self.target_logits = np.array(self.signal([self.target_model],
                                                  self.handler,
                                                  audit_data_indices)).squeeze()

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
        n_audit_samples = self.shadow_models_logits.shape[1]
        score = np.zeros(n_audit_samples)  # List to hold the computed probability scores for each sample

        # If fixed_variance is to be used, calculate it from all logits of shadow models
        if len(self.shadow_models) < 64:
            out_std = np.std(self.shadow_models_logits[~self.in_indices_mask].flatten())
            if self.online:
                in_std = np.nanstd(self.shadow_models_logits[self.in_indices_mask].flatten())

        # Iterate and extract logits from shadow models for each sample in the audit dataset
        for i in tqdm(range(n_audit_samples), total=n_audit_samples, desc="Processing samples"):
            shadow_models_logits = self.shadow_models_logits[:, i]
            mask = self.in_indices_mask[:, i]

            # Calculate the mean for OUT shadow model logits
            out_mean = np.mean(shadow_models_logits[~mask])
            if len(self.shadow_models) >= 64:
                out_std = np.std(shadow_models_logits[~mask])

            # Get the logit from the target model for the current sample
            target_logit = self.target_logits[i]

            # Calculate the log probability density function value
            pr_out = -norm.logpdf(target_logit, out_mean, out_std + 1e-30)

            if self.online:
                in_mean = np.mean(shadow_models_logits[mask])
                if len(self.shadow_models) >= 64:
                    in_std = np.std(shadow_models_logits[mask])

                pr_in = -norm.logpdf(target_logit, in_mean, in_std + 1e-30)

            else:
                pr_in = 0

            score[i] = (pr_in - pr_out)  # Append the calculated probability density value to the score list

        # Generate thresholds based on the range of computed scores for decision boundaries
        self.thresholds = np.linspace(np.nanmin(score), np.nanmax(score), 2000)

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
        )
