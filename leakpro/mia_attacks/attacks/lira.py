import numpy as np

from leakpro.dataset import get_dataset_subset
from leakpro.metrics.attack_result import CombinedMetricResult
from leakpro.mia_attacks.attack_utils import AttackUtils
from leakpro.mia_attacks.attacks.attack import AttackAbstract
from leakpro.signals.signal import ModelNegativeRescaledLogits, ModelLogits



class AttackLiRA(AttackAbstract):

    def __init__(self, attack_utils: AttackUtils, configs: dict):

        # Initializes the parent metric
        super().__init__(attack_utils)

        self.f_attack_data_size = configs["audit"].get("f_attack_data_size", 0.3)
            
        self.signal = ModelNegativeRescaledLogits()
        # self.signal = ModelLogits()
        
        self.epsilon = 1e-6
        
        # self.hypothesis_test_func = attack_utils.logit_rescale_threshold_func
        self.hypothesis_test_func = attack_utils.gaussian_threshold_func

        self.shadow_models = attack_utils.attack_objects.shadow_models

    def prepare_attack(self):
        """
        Function to prepare data needed for running the metric on the target model and dataset, using signals computed
        on the auxiliary model(s) and dataset.
        """
        # sample dataset to compute histogram
        all_index = np.arange(self.population_size)
        attack_data_size = np.round(self.f_attack_data_size * self.population_size).astype(int)

        self.attack_data_index = np.random.choice(all_index, attack_data_size, replace=False)
        attack_data = get_dataset_subset(self.population, self.attack_data_index)
        
        # Load signals if they have been computed already; otherwise, compute and save them
        # signals based on training dataset
        self.attack_signal = self.signal([self.target_model], [attack_data])
        
        

    def run_attack(self):
        """
        Function to run the attack on the target model and dataset.

        Returns:
            Result(s) of the metric.
        """
        
        self.quantiles = AttackUtils.default_quantile()
        
        # obtain the threshold values based on the reference dataset
        thresholds = self.hypothesis_test_func(self.attack_signal, self.quantiles).reshape(-1, 1)

        num_threshold = len(self.quantiles)

        # get the loss for the audit dataset
        audit_data = get_dataset_subset(self.population, self.audit_dataset["data"])
        audit_signal = self.signal([self.target_model], [audit_data])[0]

        # pick out the in-members and out-members
        self.in_member_signals = audit_signal[self.audit_dataset["in_members"]]
        self.out_member_signals = audit_signal[self.audit_dataset["out_members"]]

        # compute the signals for the in-members and out-members
        member_signals = self.in_member_signals.reshape(-1, 1).repeat(num_threshold, 1).T
        non_member_signals = self.out_member_signals.reshape(-1, 1).repeat(num_threshold, 1).T
        member_preds = np.less(member_signals, thresholds)
        non_member_preds = np.less(non_member_signals, thresholds)

        # what does the attack predict on test and train dataset
        predictions = np.concatenate([member_preds, non_member_preds], axis=1)
        # set true labels for being in the training dataset
        true_labels = np.concatenate(
            [np.ones(len(self.in_member_signals)), np.zeros(len(self.out_member_signals))]
        )
        signal_values = np.concatenate([self.in_member_signals, self.out_member_signals])

        # compute the difference between the signals and the thresholds
        # predictions_proba = np.hstack([member_signals, non_member_signals]) - thresholds

        # compute ROC, TP, TN etc
        return CombinedMetricResult(
            predicted_labels=predictions,
            true_labels=true_labels,
            predictions_proba=None,
            signal_values=signal_values,
        )
