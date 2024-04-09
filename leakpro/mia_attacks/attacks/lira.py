"""Implementation of the LiRA attack."""

# typing package not available form < python-3.11, typing_extensions backports new and experimental type hinting features to older Python versions
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from tqdm import tqdm
import numpy as np
import scipy
from torch.utils.data import DataLoader

from leakpro.dataset import get_dataset_subset
from leakpro.metrics.attack_result import CombinedMetricResult
from leakpro.mia_attacks.attack_utils import AttackUtils
from leakpro.mia_attacks.attacks.attack import AttackAbstract
from leakpro.signals.signal import ModelNegativeRescaledLogits



class AttackLiRA(AttackAbstract):
    """Implementation of the LiRA attack."""
    
    def __init__(self, attack_utils: AttackUtils, configs: dict):
        """Initialize the LiRA attack.

        Args:
        ----
            attack_utils (AttackUtils): Utility class for the attack.
            configs (dict): Configuration parameters for the attack.

        """
        # Initializes the parent metric
        super().__init__(attack_utils)

        self.f_attack_data_size = configs["audit"].get("f_attack_data_size", 0.3)
            
        self.signal = ModelNegativeRescaledLogits()
        
        self.shadow_models = attack_utils.attack_objects.shadow_models
        self.num_shadow_models = len(self.shadow_models)
        self.shadow_models_metadata = attack_utils.attack_objects.shadow_models_metadata


    def prepare_attack(self):
        """
        Function to prepare data needed for running the metric on the target model and dataset, using signals computed
        on the auxiliary model(s) and dataset.
        """

        # in_index = self.audit_dataset["data"][self.audit_dataset["in_members"]]
        # out_index = self.get_out_members(in_index)
        # print(out_index)
        
        audit_dataset = get_dataset_subset(self.population, self.audit_dataset["data"])
        
        print(f"Calculating the logits for all {self.num_shadow_models} shadow models")
        self.shadow_models_logits = np.array(self.signal(self.shadow_models, audit_dataset))

        print(f"Calculating the logits for the target model")
        self.target_logits = np.array(self.signal([self.target_model], audit_dataset)).squeeze()

        # print(self.shadow_models_logits.shape, self.target_logits.shape)
        

    def run_attack(self):
        """
        Function to run the attack on the target model and dataset.

        Returns:
            Result(s) of the metric.
        """
        
        score = []
        out_of_sample_threshold = [1.] #, 0.9, 0.8, 0.75, 0.5]
        use_global_std = False
        
        if use_global_std:
            global_std = np.nanstd(self.shadow_models_logits.flatten())
            print(global_std)
            
        # for sample_threshold in out_of_sample_threshold:

            # Iterate over the dataset using the DataLoader (ensures we use transforms etc)
            for i, audit_idx in tqdm(enumerate(self.audit_dataset["data"])):
                # count, sm_idxs = self.shadow_models_without_traning_sample(self.shadow_models_metadata, audit_idx)
                # if count/self.num_shadow_models >= sample_threshold:
    
                # Collect the shadow logits
                shadow_models_logits = [self.shadow_models_logits[j, i] for j in sm_idxs]
                mean = np.nanmean(shadow_models_logits)
                std = np.nanstd(shadow_models_logits)
                
                # Collect the target logit
                target_logit = self.target_logits[i]
                
                # pr_out = scipy.stats.norm.logpdf(target_logit, mean, global_std+1e-30)
                pr_out = scipy.stats.norm.logpdf(target_logit, mean, std+1e-30)
                    
                score.append(pr_out)
                        
                # else:
                #     score.append(np.nan)
                
        score = np.asarray(score)

        print(np.nanmin(score),  np.nanmax(score))
        import time 
        time.sleep(5)
        self.thresholds = np.linspace(np.nanmin(score)-1, np.nanmax(score)+1, 5000) # np.min
        # mean_lsm = np.mean(self.shadow_models_logits, axis=0)
        # std_lsm = np.std(self.shadow_models_logits, axis=0)
        # score = []
        # print(target_logits.shape, mean_lsm.shape, std_lsm.shape)
        # for logit, mean, std in zip(target_logits, mean_lsm, std_lsm):
        #     pr_out = scipy.stats.norm.logpdf(logit, mean, std+1e-30)
        #     score.append(pr_out)
        # score = np.asarray(score)
        
        
        # score = []
        # print(target_logits.shape, self.mean_lsm.shape, self.std_lsm.shape)
        # for logit in target_logits:
        #     pr_out = scipy.stats.norm.logpdf(logit, self.mean_lsm, self.std_lsm+1e-30)
        #     score.append(pr_out)
        # score = np.asarray(score)
        
        # print(score.shape)
        # print(self.audit_dataset["in_members"])
        # print(score.shape, max(score), min(score))
        # import time 
        # time.sleep(10)
        
        # thresholds = self.thresholds

        # print(logit.shape, mean.shape, std.shape)
        # print(np.asarray(score).shape, np.asarray(sc).shape)
        # import time 
        # time.sleep(5)
        
        # # pick out the in-members and out-members signals
        # self.in_member_signals = score[self.in_members].reshape(-1,1)
        # self.out_member_signals = score[self.out_members].reshape(-1,1)
        # pick out the in-members and out-members signals
        self.in_member_signals = score[self.audit_dataset["in_members"]].reshape(-1,1)
        self.out_member_signals = score[self.audit_dataset["out_members"]].reshape(-1,1)
        
        # compute the signals for the in-members and out-members
        member_preds = np.greater(self.in_member_signals, self.thresholds).T
        non_member_preds = np.greater(self.out_member_signals, self.thresholds).T

        # compute the signals for the in-members and out-members
        # member_preds = np.less(self.in_member_signals, self.thresholds).T
        # non_member_preds = np.less(self.out_member_signals, self.thresholds).T
        
        # compute the difference between the signals and the thresholds
        # predictions_proba = np.hstack([member_signals, non_member_signals]) - thresholds

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
        
