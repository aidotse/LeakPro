"""Implementation of the LiRA attack."""

from logging import Logger
from tqdm import tqdm
from torch import nn
import numpy as np
from scipy.stats import norm
# from torch.utils.data import DataLoader

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.attack_data import get_attack_data
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.import_helper import Self
from leakpro.metrics.attack_result import CombinedMetricResult
from leakpro.signals.signal import ModelNegativeRescaledLogits, ModelRescaledLogits



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
        self.training_data_fraction = configs.get("data_fraction", 0.5)

        self.shadow_models = []
        self.num_shadow_models = configs.get("num_shadow_models", 64)

        self.online = configs.get("online", False)
        self.fixed_variance = configs.get("fixed_variance", False)
        self.include_test_data = configs.get("include_test_data", True)
        
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
        """
        Prepares data needed for running the metric on the target model and dataset,
        using signals computed on the auxiliary model(s) and dataset.
    
        It selects a balanced subset of data samples from in-group and out-group members
        of the audit dataset, prepares the data for evaluation, and computes the logits
        for both shadow models and the target model.
        """
        
        self.attack_data_index = get_attack_data(
            self.population_size,
            self.train_indices,
            self.test_indices,
            self.include_test_data,
            self.logger
        )
        attack_data = self.population.subset(self.attack_data_index)

        ShadowModelHandler().create_shadow_models(
            self.num_shadow_models,
            attack_data,
            self.training_data_fraction,
        )
        
        self.shadow_models, _ = ShadowModelHandler().get_shadow_models(self.num_shadow_models)
        
        self.in_indices_mask = ShadowModelHandler().get_in_indices_mask(self.num_shadow_models, self.audit_dataset)
        
        self.audit_data = self.population.subset(self.audit_dataset["data"])
        
        # # Extract a matching subset from the population dataset
        # audit_dataset = self.population.subset(self.audit_data)
        
        # Calculate logits for all shadow models
        print(f"Calculating the logits for all {self.num_shadow_models} shadow models")
        self.shadow_models_logits = np.swapaxes(np.array(self.signal(self.shadow_models, self.audit_data)), 0, 1)
        
        # Calculate logits for the target model
        print(f"Calculating the logits for the target model")
        self.target_logits = np.array(self.signal([self.target_model], self.audit_data)).squeeze()     

        # print(self.shadow_models_logits[:,self.audit_dataset["in_members"]].shape)
        # print(self.in_indices_mask[self.audit_dataset["in_members"]].shape)

        # in_members = self.shadow_models_logits[self.audit_dataset["in_members"]]
        # [:,~self.in_indices_mask[self.audit_dataset["in_members"]]])
        
        # out_members = self.shadow_models_logits[self.audit_dataset["out_members"]]
        # [:,~self.in_indices_mask[self.audit_dataset["out_members"]]])

    def run_attack(self):
        """
        Runs the attack on the target model and dataset, computing and returning
        the result(s) of the metric, which assess privacy risks or data leakage.
    
        This method evaluates how the target model's output (logits) for a specific dataset 
        compares to the output of shadow models to determine if the dataset was part of the 
        model's training data or not.
    
        Returns:
            Result(s) of the metric. An object containing the metric results, including predictions,
            true labels, and signal values.
        """
        
        score = []  # List to hold the computed probability scores for each sample
        
        # If global standard deviation is to be used, calculate it from all logits of shadow models
        if self.fixed_variance:
            in_std = np.nanstd(self.shadow_models_logits[self.in_indices_mask].flatten())
            out_std = np.nanstd(self.shadow_models_logits[~self.in_indices_mask].flatten())
            
        # Iterate over each sample in the audit dataset with a progress bar
        print(self.in_indices_mask.shape)
        for i, mask in tqdm(enumerate(self.in_indices_mask)):
            # Extract logits from shadow models for the current sample
            shadow_models_logits = self.shadow_models_logits[i, :]
            out_mean = np.nanmean(shadow_models_logits[~mask])  # Calculate the mean of shadow model logits
            
            # Get the logit from the target model for the current sample
            target_logit = self.target_logits[i]
    
            # Calculate the log probability density function value
            if not self.fixed_variance:
                out_std = np.nanstd(shadow_models_logits[~mask])
                in_std = np.nanstd(shadow_models_logits[mask])
            
            if self.online:
                in_mean = np.nanmean(shadow_models_logits[mask])
                pr_in = -norm.logpdf(target_logit, in_mean, in_std + 1e-30)
                
            else:
                pr_in = 0
            
            pr_out = -norm.logpdf(target_logit, out_mean, out_std + 1e-30)
            
            score.append(pr_in - pr_out)  # Append the calculated probability density value to the score list
        
        score = np.asarray(score)  # Convert the list of scores to a numpy array
        
        # Generate thresholds based on the range of computed scores for decision boundaries
        self.thresholds = np.linspace(np.nanmin(score)-0.5, np.nanmax(score), 3000)
    
        # Split the score array into two parts based on membership: in (training) and out (non-training)
        self.in_member_signals = score[self.audit_dataset["in_members"]].reshape(-1,1)  # Scores for known training data members
        self.out_member_signals = score[self.audit_dataset["out_members"]].reshape(-1,1)  # Scores for non-training data members
    
        # Create prediction matrices by comparing each score against all thresholds
        member_preds = np.less(self.in_member_signals, self.thresholds).T  # Predictions for training data members
        non_member_preds = np.less(self.out_member_signals, self.thresholds).T  # Predictions for non-members
        # member_preds = np.greater(self.in_member_signals, self.thresholds).T
        # non_member_preds = np.greater(self.out_member_signals, self.thresholds).T
    
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


    def shadow_models_without_traning_sample(self, shadow_models, training_sample_idx):
        """
        Counts how many shadow models do not contain the specified training sample and returns the count and indices.
        
        Args:
            ----
            shadow_models: List of shadow models to be searched
            training_sample_idx: Training sample to search for
        
        Returns:
            The count of shadow models that does not contain the training sample
            List of indices of the shadow model that was not trained of the training sample
        """
        # Initialize counter for sublists without the value and a list to store indices
        count = 0
        indices = []
        
        # Iterate through each sublist in the list of lists
        for index, shadow_model in enumerate(shadow_models):
            # If the value is not in the current sublist, increment the count and record the index
            if training_sample_idx not in shadow_model['train_idxs']:
                count += 1
                indices.append(index)
                
        # Return the count of sublists without the value and their indices
        return count, indices

    def get_out_members(self:Self, in_index):
        
        all_index = np.arange(self.population_size)
        available_index = np.setdiff1d(all_index, in_index, assume_unique=True)
        
        for model in self.shadow_models_metadata:
            available_index = np.setdiff1d(available_index, model['train_idxs'], assume_unique=True)

        return available_index
        