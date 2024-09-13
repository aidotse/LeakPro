import pickle  # noqa: D100

import numpy as np

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.import_helper import Self
from leakpro.metrics.attack_result import CombinedMetricResult
from leakpro.signals.signal import HopSkipJumpDistance
from leakpro.user_inputs.abstract_input_handler import AbstractInputHandler
from leakpro.utils.logger import logger


class AttackHopSkipJump(AbstractMIA):  # noqa: D101
    def __init__(self: Self,
                 handler: AbstractInputHandler,
                 configs: dict
                ) -> None:
        super().__init__(handler)
        """Initialize the HopSkipJump class.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.
            configs (dict): A dictionary containing the attack loss_traj configurations.

        """

        logger.info("Configuring label only attack")
        self._configure_attack(configs)
        self.signal = HopSkipJumpDistance()


    def _configure_attack(self:Self,
                          configs: dict) -> None:
        """Configure the attack using the configurations."""
        self.configs = configs
        self.attack_data_fraction = configs.get("attack_data_fraction", 0.1)
        self.target_metadata_path = configs.get("trained_model_metadata_path", "./target/model_metadata.pkl")
        with open(self.target_metadata_path, "rb") as f:
             self.target_model_metadata = pickle.load(f)  # noqa: S301

        self.norm = configs.get("norm", 2)
        self.y_target = configs.get("y_target", None)  # noqa: SIM910
        self.image_target = configs.get("image_target", None)  # noqa: SIM910
        self.initial_num_evals = configs.get("initial_num_evals", 100)
        self.max_num_evals = configs.get("max_num_evals", 10000)
        self.stepsize_search = configs.get("stepsize_search", "geometric_progression")
        self.num_iterations = configs.get("num_iterations", 100)
        self.gamma = configs.get("gamma", 1.0)
        self.constraint = configs.get("constraint", 2)
        self.batch_size = configs.get("batch_size", 128)
        self.verbose = configs.get("verbose", True)
        self.epsilon_threshold = configs.get("epsilon_threshold", 1e-6)
        self.user_input_validation()

    def user_input_validation(self:Self) -> None:
        """Validate the user input configurations."""
        self._validate_norm()
        self._validate_stepsize_search()
        self._validate_constraint()
        self._validate_initial_num_evals()
        self._validate_max_num_evals()
        self._validate_num_iterations()
        self._validate_gamma()
        self._validate_batch_size()
        self._validate_y_target()

    def _validate_norm(self:Self) -> None:
        """Validate the norm value."""
        valid_norm_values = [1, 2, np.inf]
        if self.norm not in valid_norm_values:
            raise ValueError(f"Invalid norm value: {self.norm}. Must be one of {valid_norm_values}")

    def _validate_stepsize_search(self:Self) -> None:
        """Validate the stepsize_search value."""
        valid_stepsize_search_values = ["geometric_progression"]
        if self.stepsize_search not in valid_stepsize_search_values:
            raise ValueError(f"Invalid stepsize_search. This version supports{valid_stepsize_search_values}")

    def _validate_constraint(self:Self) -> None:
        """Validate the constraint value."""
        valid_constraint_values = [1, 2]
        if self.constraint not in valid_constraint_values:
            raise ValueError(f"Invalid constraint value: {self.constraint}. Must be one of {valid_constraint_values}")

    def _validate_initial_num_evals(self:Self) -> None:
        """Validate the initial_num_evals value."""
        if not (1 <= self.initial_num_evals <= 1000):
            raise ValueError(f"Invalid initial_num_evals value: {self.initial_num_evals}. "
                            "Must be between 1 and 1000 (inclusive).")

    def _validate_max_num_evals(self:Self) -> None:
        """Validate the max_num_evals value."""
        if self.max_num_evals <= self.initial_num_evals:
            raise ValueError("max_num_evals must be greater than initial_num_evals")

    def _validate_num_iterations(self:Self) -> None:
        """Validate the num_iterations value."""
        if self.num_iterations <= 0:
            raise ValueError("num_iterations must be greater than 0")

    def _validate_gamma(self:Self) -> None:
        """Validate the gamma value."""
        if self.gamma <= 0:
            raise ValueError("gamma must be greater than 0")

    def _validate_batch_size(self:Self) -> None:
        """Validate the batch_size value."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

    def _validate_y_target(self:Self) -> None:
        """Validate the y_target value."""
        num_classes = self.target_model_metadata["init_params"]["num_classes"]
        if self.y_target is not None and self.y_target not in range(num_classes):
            raise ValueError("y_target must be an integer and in the range of the number of classes in the target model.")

    def _validate_epsilon_threshold(self:Self) -> None:
        """Validate the epsilon_threshold value."""
        if self.epsilon_threshold <= 0:
            raise ValueError("epsilon_threshold must be greater than 0")
        if self.epsilon_threshold >= 0.001:
            raise ValueError("epsilon_threshold must be a very small value")


    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "Label-Only Membership Inference Attacks"
        reference_str = "Christopher A. Choquette-Choo, Florian Tramer, Nicholas Carlini and Nicolas Papernot\
            Label-Only Membership Inference Attacks. (2020)."
        summary_str = "This attack is one of the introduce black-box membership inference attacks in the paper."
        detailed_str = "The distance attack executed based on the estimation of the distance between the input \
                        data and the decision boundary of a shadow model of the target model. \
                        The attack aims to use this distance as a signal to infer membership."
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }

    def prepare_attack(self:Self) -> None:
        """Prepare the attack by loading the shadow model and target model.

        Args:
        ----
            self (Self): The instance of the class.

        Returns:
        -------
            None

        """
        logger.info("Preparing the data for Hop Skip Jump attack")

        in_member_indices = self.audit_dataset["in_members"]
        out_member_indices = self.audit_dataset["out_members"]

        audit_in_member_indicies = np.random.choice(in_member_indices,
                                              int(len(in_member_indices) * self.attack_data_fraction),
                                                replace=False)

        audit_out_member_indicies = np.random.choice(out_member_indices,
                                                int(len(out_member_indices) * self.attack_data_fraction),
                                                replace=False)
        audit_indices = np.concatenate((audit_in_member_indicies, audit_out_member_indicies))

        self.attack_dataloader = self.handler.get_dataloader(audit_indices, batch_size=self.batch_size)



    def run_attack(self:Self) -> CombinedMetricResult:
        """Run the attack and return the combined metric result.

        Returns
        -------
            CombinedMetricResult: The combined metric result containing predicted labels, true labels,
            predictions probabilities, and signal values.

        """

        logger.info("Running Hop Skip Jump distance attack")

        # compute the perturbation distances of the attack data from the target model decision boundary
        _ , perturbation_distances = self.signal(self.target_model,
                                                    self.attack_dataloader,
                                                    self.norm,
                                                    self.y_target,
                                                    self.image_target,
                                                    self.initial_num_evals,
                                                    self.max_num_evals,
                                                    self.stepsize_search,
                                                    self.num_iterations,
                                                    self.gamma,
                                                    self.constraint,
                                                    self.batch_size,
                                                    self.verbose
                                                    )

        # create thresholds
        min_signal_val = np.min(perturbation_distances)
        max_signal_val = np.max(perturbation_distances)
        thresholds = np.linspace(min_signal_val, max_signal_val, 1000)
        num_threshold = len(thresholds)

        # compute the signals for the in-members and out-members
        member_signals = (np.array(perturbation_distances).reshape(-1, 1).repeat(num_threshold, 1).T)

        member_preds = np.greater(member_signals, thresholds[:, np.newaxis])

        # set true labels for being in the training dataset
        true_labels = np.concatenate(
            [
                np.ones(int(len(self.attack_dataloader.dataset)/2)),
                np.zeros(int(len(self.attack_dataloader.dataset)/2)),
            ]
        )

        # compute ROC, TP, TN etc
        return CombinedMetricResult(
            predicted_labels=member_preds,
            true_labels=true_labels,
            predictions_proba=None,
            signal_values= perturbation_distances,
        )
