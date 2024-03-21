from leakpro.mia_attacks.attacks.attack import AttackAbstract
from leakpro.mia_attacks.attack_objects import AttackObjects
from leakpro.mia_attacks.attack_utils import AttackUtils
from leakpro.signals.signal import ModelLoss


class AttackRMIA(AttackAbstract):
    def __init__(self, attack_objects: AttackObjects):
        self.population = attack_objects.population
        self.target_model = attack_objects.target_model
        self.audit_dataset = attack_objects.audit_dataset
        self.signal = ModelLoss()
        self.signal_data = []

    def prepare_attack(self):
        """
        Function to prepare data needed for running the metric on the target model and dataset, using signals computed
        on the auxiliary model(s) and dataset.
        """
        pass

    def run_attack(self, fpr_tolerance_rate_list=None):
        """
        Function to run the attack on the target model and dataset.

        Args:
            fpr_tolerance_rate_list (optional): List of FPR tolerance values that may be used by the threshold function
                to compute the attack threshold for the metric.

        Returns:
            Result(s) of the metric.
        """
        pass
