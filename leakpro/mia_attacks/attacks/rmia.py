"""Implementation of the RMIA attack."""
# typing package not available form < python-3.11, typing_extensions backports new and experimental type hinting features to older Python versions
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
    
import numpy as np

from leakpro.dataset import get_dataset_subset
from leakpro.metrics.attack_result import CombinedMetricResult
from leakpro.mia_attacks.attack_utils import AttackUtils
from leakpro.mia_attacks.attacks.attack import AttackAbstract
from leakpro.signals.signal import ModelLogits


class AttackRMIA(AttackAbstract):
    """Implementation of the RMIA attack."""

    def __init__(self:Self, attack_utils: AttackUtils, configs: dict) -> None:
        """Initialize the RMIA attack.

        Args:
        ----
            attack_utils (AttackUtils): Utility class for the attack.
            configs (dict): Configuration parameters for the attack.

        """
        # Initializes the parent metric
        super().__init__(attack_utils)

        self.shadow_models = attack_utils.attack_objects.shadow_models
        self.offline_a = 0.33 # parameter from which we compute p(x) from p_OUT(x) such that p_IN(x) = a p_OUT(x) + b.
        self.offline_b: 0.66
        self.gamma = 2.0 # threshold for the attack
        self.temperature = 2.0 # temperature for the softmax

        self.f_attack_data_size = configs["audit"].get("f_attack_data_size", 0.3)

        self.signal = ModelLogits()
        self.epsilon = 1e-6


    def softmax(self:Self, all_logits:np.ndarray,
                true_label_indices:np.ndarray,
                return_full_distribution:bool=False) -> np.ndarray:
        """Compute the softmax function.

        Args:
        ----
            all_logits (np.ndarray): Logits for each class.
            true_label_indices (np.ndarray): Indices of the true labels.
            return_full_distribution (bool, optional): return the full distribution or just the true class probabilities.

        Returns:
        -------
            np.ndarray: Softmax output.

        """
        logit_signals = all_logits / self.temperature
        max_logit_signals = np.max(logit_signals,axis=2)
        logit_signals = logit_signals - max_logit_signals.reshape(1,-1,1)
        exp_logit_signals = np.exp(logit_signals)
        exp_logit_sum = np.sum(exp_logit_signals, axis=2)

        if return_full_distribution is False:
            true_exp_logit =  exp_logit_signals[:, np.arange(exp_logit_signals.shape[1]), true_label_indices]
            output_signal = true_exp_logit / exp_logit_sum
        else:
            output_signal = exp_logit_signals / exp_logit_sum[:,:,np.newaxis]
        return output_signal

    def prepare_attack(self:Self) -> None:
        """Prepare data needed for running the attack on the target model and dataset.

        Signals are computed on the auxiliary model(s) and dataset.
        """
        # sample dataset to compute histogram
        all_index = np.arange(self.population_size)
        attack_data_size = np.round(
            self.f_attack_data_size * self.population_size
        ).astype(int)

        self.attack_data_index = np.random.choice(
            all_index, attack_data_size, replace=False
        )
        attack_data = get_dataset_subset(self.population, self.attack_data_index)

        # compute the ratio of p(z|theta) (target model) to p(z)=sum_{theta'} p(z|theta') (shadow models)
        # for all points in the attack dataset output from signal: # models x # data points x # classes

        # get the true label indices
        z_label_indices = np.array(attack_data.y)

        # run points through real model to collect the logits
        logits_theta = np.array(self.signal([self.target_model], attack_data))
        # collect the softmax output of the correct class
        p_z_given_theta = self.softmax(logits_theta, z_label_indices)

        # run points through shadow models and collect the logits
        logits_shadow_models = self.signal(self.shadow_models, attack_data)
        # collect the softmax output of the correct class for each shadow model
        p_z_given_shadow_models = [self.softmax(np.array(x).reshape(1,*x.shape), z_label_indices) for x in logits_shadow_models]
        # stack the softmax output of the correct class for each shadow model to dimension # models x # data points
        p_z_given_shadow_models = np.array(p_z_given_shadow_models).squeeze()

        # evaluate the marginal p(z)
        p_z = np.mean(p_z_given_shadow_models, axis=0) if len(self.shadow_models) > 1 else p_z_given_shadow_models.squeeze()
        p_z = 0.5*((self.offline_a + 1) * p_z + (1-self.offline_a))

        #TODO: pick the maximum value of the softmax output in p(z)
        self.ratio_z = p_z_given_theta / (p_z + self.epsilon)


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
        # get the logits for the audit dataset
        audit_data = get_dataset_subset(self.population, self.audit_dataset["data"])
        x_label_indices = np.array(audit_data.y)

        # run target points through real model to get logits
        logits_theta = np.array(self.signal([self.target_model], audit_data))
        # collect the softmax output of the correct class
        p_x_given_theta = self.softmax(logits_theta, x_label_indices)

        # run points through shadow models and collect the logits
        logits_shadow_models = self.signal(self.shadow_models, audit_data)
        # collect the softmax output of the correct class for each shadow model
        p_x_given_shadow_models = [self.softmax(np.array(x).reshape(1,*x.shape), x_label_indices) for x in logits_shadow_models]
        # stack the softmax output of the correct class for each shadow model
        # to dimension # models x # data points
        p_x_given_shadow_models = np.array(p_x_given_shadow_models).squeeze()
        # evaluate the marginal p_out(x) by averaging the output of the shadow models
        p_x_out = np.mean(p_x_given_shadow_models, axis=0) if len(self.shadow_models) > 1 else p_x_given_shadow_models.squeeze()

        # compute the marginal p(x) from P_out and p_in where p_in = a*p_out+b
        p_x = 0.5*((self.offline_a + 1) * p_x_out + (1-self.offline_a))

        # compute the ratio of p(x|theta) to p(x)
        ratio_x = p_x_given_theta / (p_x + self.epsilon)

        # for each x, compare it with the ratio of all z points
        likelihoods = ratio_x.T / self.ratio_z
        score = np.mean(likelihoods > self.gamma, axis=1)

        # pick out the in-members and out-members signals
        self.in_member_signals = score[self.audit_dataset["in_members"]].reshape(-1,1)
        self.out_member_signals = score[self.audit_dataset["out_members"]].reshape(-1,1)

        thresholds = np.linspace(1/likelihoods.shape[1], 1, 1000)


        member_preds = np.greater(self.in_member_signals, thresholds).T
        non_member_preds = np.greater(self.out_member_signals, thresholds).T

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


