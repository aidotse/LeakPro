"""Implementation of the LiRA attack."""

from logging import Logger

import numpy as np
from scipy.stats import norm
from torch import nn
from tqdm import tqdm

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.attack_data import get_attack_data
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.import_helper import Self
from leakpro.metrics.attack_result import CombinedMetricResult
from leakpro.signals.signal import ModelRescaledLogits


class AttackLiRA(AbstractMIA):
    """Implementation of the LiRA attack."""

    def __init__(self:Self,
                 population: np.ndarray,
                 audit_dataset: dict,
                 target_model: nn.Module,
                 logger:Logger,
                 configs: dict
                 ) -> None:
        """Initialize the LiRA attack.

        Args:
        ----
            attack_utils (AttackUtils): Utility class for the attack.
            configs (dict): Configuration parameters for the attack.

        """
        # Initializes the parent metric
        super().__init__(population, audit_dataset, target_model, logger)

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

        self.online = configs.get("online", False)
        self.fixed_variance = configs.get("fixed_variance", False)

        self.include_train_data = configs.get("include_train_data", self.online)
        self.include_test_data = configs.get("include_test_data", self.online)

        # Define the validation dictionary as: {parameter_name: (parameter, min_value, max_value)}
        validation_dict = {
            "num_shadow_models": (self.num_shadow_models, 1, None),
            "training_data_fraction": (self.training_data_fraction, 0, 1),
        }

        # Validate parameters
        for param_name, (param_value, min_val, max_val) in validation_dict.items():
            self._validate_config(param_name, param_value, min_val, max_val)

    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "Likelihood Ratio Attack"

        reference_str = "Carlini N, et al. Membership Inference Attacks From First Principles"

        summary_str = "The Likelihood Ratio Attack (LiRA) is a membership inference attack based on the rescaled logits of a black-box model"

        detailed_str = "The attack is executed according to: \
            1. A fraction of the target model dataset is sampled to be included (in-) or excluded (out-) from the shadow model training dataset. \
            2. The rescaled logits are used to estimate Gaussian distributions for in and out members \
            3. The thresholds are used to classify in-members and out-members. \
            4. The attack is evaluated on an audit dataset to determine the attack performance."

        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }

    def prepare_attack(self):
        """Prepares data needed for running the metric on the target model and dataset,
        using signals computed on the auxiliary model(s) and dataset.
    
        It selects a balanced subset of data samples from in-group and out-group members
        of the audit dataset, prepares the data for evaluation, and computes the logits
        for both shadow models and the target model.
        """
        self.attack_data_index = get_attack_data(
            self.population_size,
            self.train_indices,
            self.test_indices,
            train_data_included_in_auxiliary_data=self.include_train_data,
            test_data_included_in_auxiliary_data=self.include_test_data,
            logger = self.logger
        )

        ShadowModelHandler().create_shadow_models(
            self.num_shadow_models,
            self.population,
            self.attack_data_index,
            self.training_data_fraction,
        )

        self.shadow_models, _ = ShadowModelHandler().get_shadow_models(self.num_shadow_models)

        self.in_indices_mask = ShadowModelHandler().get_in_indices_mask(self.num_shadow_models, self.audit_dataset["data"])

        self.audit_data = self.population.subset(self.audit_dataset["data"])

        # Check offline attack for possible IN- sample(s)
        if not self.online:
            count_in_samples = np.count_nonzero(self.in_indices_mask)
            if count_in_samples > 0:
                raise ValueError("Some shadow model(s) contains IN samples, this is not an offline attack!")
                       
        if self.online:
            no_in = 0
            no_out = 0
            for mask in self.in_indices_mask():
                if len(mask) == np.count_nonzero(mask):
                    no_out += 1
                else:
                    no_in += 1
            if no_out > 0 or no_in > 0:
                self.logger.info(f"There are {no_out} audit examples with 0 OUT sample(s) and {no_in} 0 IN sample(s)\n \
                When using few shadow models in online attacks, some audit sample(s) might have no IN or OUT logits")

        # Calculate logits for all shadow models
        self.logger.info(f"Calculating the logits for all {self.num_shadow_models} shadow models")
        self.shadow_models_logits = np.swapaxes(np.array(self.signal(self.shadow_models, self.audit_data)), 0, 1)

        # Calculate logits for the target model
        self.logger.info("Calculating the logits for the target model")
        self.target_logits = np.array(self.signal([self.target_model], self.audit_data)).squeeze()

    def run_attack(self):
        """Runs the attack on the target model and dataset, computing and returning
        the result(s) of the metric, which assess privacy risks or data leakage.
    
        This method evaluates how the target model's output (logits) for a specific dataset 
        compares to the output of shadow models to determine if the dataset was part of the 
        model's training data or not.
    
        Returns
        -------
            Result(s) of the metric. An object containing the metric results, including predictions,
            true labels, and signal values.

        """
        score = []  # List to hold the computed probability scores for each sample

        # If fixed_variance is to be used, calculate it from all logits of shadow models
        if self.fixed_variance:
            out_std = np.std(self.shadow_models_logits[~self.in_indices_mask].flatten())
            if self.online:
                in_std = np.nanstd(self.shadow_models_logits[self.in_indices_mask].flatten())

        # Iterate and extract logits from shadow models for each sample in the audit dataset
        for i, (shadow_models_logits, mask) in tqdm(enumerate(zip(self.shadow_models_logits, self.in_indices_mask))):

            # Calculate the mean for OUT shadow model logits
            out_mean = np.mean(shadow_models_logits[~mask])  

            # Get the logit from the target model for the current sample
            target_logit = self.target_logits[i]

            # Calculate the log probability density function value
            pr_out = -norm.logpdf(target_logit, out_mean, out_std + 1e-30)
            
            if self.online:
                in_mean = np.mean(shadow_models_logits[mask])
                if not self.fixed_variance:
                    out_std = np.std(shadow_models_logits[~mask])
                    in_std = np.std(shadow_models_logits[mask])
                    
                pr_in = -norm.logpdf(target_logit, in_mean, in_std + 1e-30)
            else:
                pr_in = 0
                if not self.fixed_variance:
                    out_std = np.std(shadow_models_logits[~mask])

            score.append(pr_in - pr_out)  # Append the calculated probability density value to the score list
            
        score = np.asarray(score)  # Convert the list of scores to a numpy array

        # Generate thresholds based on the range of computed scores for decision boundaries
        self.thresholds = np.linspace(np.nanmin(score), np.nanmax(score), 1000)

        # Split the score array into two parts based on membership: in (training) and out (non-training)
        self.in_member_signals = score[self.audit_dataset["in_members"]].reshape(-1,1)  # Scores for known training data members
        self.out_member_signals = score[self.audit_dataset["out_members"]].reshape(-1,1)  # Scores for non-training data members

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
        
