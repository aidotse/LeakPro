import pickle  # noqa: D100
from logging import Logger

import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.attack_data import get_attack_data
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.import_helper import Self
from leakpro.metrics.attack_result import CombinedMetricResult
from leakpro.signals.signal import HopSkipJumpDistance


class AttackHopSkipJump(AbstractMIA):  # noqa: D101
    def __init__(self: Self,
                 population: np.ndarray,
                 audit_dataset: dict,
                 target_model: nn.Module,
                 logger: Logger,
                 configs: dict
                ) -> None:
        super().__init__(population, audit_dataset, target_model, logger)

        self.logger.info("Configuring label only attack")
        self._configure_attack(configs)
        self.signal = HopSkipJumpDistance()


    def _configure_attack(self:Self,
                          configs: dict) -> None:
        """Configure the attack using the configurations."""
        self.configs = configs
        self.target_metadata_path = configs.get("trained_model_metadata_path", "./target/model_metadata.pkl")
        with open(self.target_metadata_path, "rb") as f:
             self.target_model_metadata = pickle.load(f)  # noqa: S301

        target_train_indices = self.target_model_metadata["model_metadata"]["train_indices"]
        target_test_indices = self.target_model_metadata["model_metadata"]["test_indices"]
        self.target_train_dataset =  self.population.subset(target_train_indices)
        self.target_test_dataset = self.population.subset(target_test_indices)


        self.attack_data_fraction = configs.get("attack_data_fraction", 0.5)
        self.num_shadow_models = configs.get("num_shadow_models", 1)
        self.norm = configs.get("norm", 2)
        self.y_target = configs.get("y_target")
        self.image_target = configs.get("image_target")
        self.initial_num_evals = configs.get("initial_num_evals", 100)
        self.max_num_evals = configs.get("max_num_evals", 10000)
        self.stepsize_search = configs.get("stepsize_search", "geometric_progression")
        self.num_iterations = configs.get("num_iterations", 100)
        self.gamma = configs.get("gamma", 1.0)
        self.constraint = configs.get("constraint", 2)
        self.batch_size = configs.get("batch_size", 128)
        self.verbose = configs.get("verbose", True)
        self.clip_min = configs.get("clip_min", -1)
        self.clip_max = configs.get("clip_max", 1)



    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "Label-Only Membership Inference Attacks"
        reference_str = "Christopher A. Choquette-Choo, Florian Tramer, Nicholas Carlini and Nicolas Papernot\
            Label-Only Membership Inference Attacks. (2020)."
        summary_str = ""
        detailed_str = ""
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
        self.logger.info("Preparing the data for Hop Skip Jump attack")
        include_target_training_data = False
        include_target_testing_data = False

        # Get all available indices for the only shadow model dataset
        shadow_data_index = get_attack_data(
            self.population_size,
            self.train_indices,
            self.test_indices,
            include_target_training_data,
            include_target_testing_data,
            self.logger
        )

        # create auxiliary dataset
        shadow_data_size = len(shadow_data_index)
        shadow_train_data_size = int(shadow_data_size * self.attack_data_fraction)
        shadow_train_data_indices = np.random.choice(shadow_data_index, shadow_train_data_size, replace=False)
        shadow_test_data_indices = np.setdiff1d(shadow_data_index, shadow_train_data_indices)

        self.shadow_train_dataset = self.population.subset(shadow_train_data_indices)
        self.shadow_test_dataset = self.population.subset(shadow_test_data_indices)


        # train shadow models
        self.logger.info(f"Training shadow models on {len(self.shadow_train_dataset)} points")
        ShadowModelHandler().create_shadow_models(
            self.num_shadow_models,
            self.shadow_train_dataset,
            shadow_train_data_indices,
            training_fraction = 5.0,
            retrain= False,
        )
        # load shadow models
        self.shadow_models, self.shadow_model_indices = \
            ShadowModelHandler().get_shadow_models(self.num_shadow_models)
        self.shadow_metadata = ShadowModelHandler().get_shadow_model_metadata(1)



    def run_attack(self:Self) -> CombinedMetricResult:
        """Run the attack and return the combined metric result.

        Returns
        -------
            CombinedMetricResult: The combined metric result containing predicted labels, true labels,
            predictions probabilities, and signal values.

        """
        shadow_model = self.shadow_models[0]

        shadow_train_loader = DataLoader(self.shadow_train_dataset, batch_size=self.batch_size, shuffle=True)
        shadow_test_loader = DataLoader(self.shadow_test_dataset, batch_size=self.batch_size, shuffle=True)

        self.logger.info("Running Hop Skip Jump distance attack")
        _ , perturbation_distances_in = self.signal( model = shadow_model,
                                                    data_loader = shadow_train_loader,
                                                    norm = self.norm,
                                                    y_target = self.y_target,
                                                    image_target = self.image_target,
                                                    initial_num_evals = self.initial_num_evals,
                                                    max_num_evals = self.max_num_evals,
                                                    stepsize_search = self.stepsize_search,
                                                    num_iterations = self.num_iterations,
                                                    gamma = self.gamma,
                                                    constraint = self.constraint,
                                                    batch_size = self.batch_size,
                                                    verbose = self.verbose,
                                                    clip_min = self.clip_min,
                                                    clip_max = self.clip_max,
                                                    logger = self.logger
                                                    )

        _ , perturbation_distances_out = self.signal(model = shadow_model,
                                                data_loader = shadow_test_loader,
                                                norm = self.norm,
                                                y_target = self.y_target,
                                                image_target = self.image_target,
                                                initial_num_evals = self.initial_num_evals,
                                                max_num_evals = self.max_num_evals,
                                                stepsize_search = self.stepsize_search,
                                                num_iterations = self.num_iterations,
                                                gamma = self.gamma,
                                                constraint = self.constraint,
                                                batch_size = self.batch_size,
                                                verbose = self.verbose,
                                                clip_min = self.clip_min,
                                                clip_max = self.clip_max,
                                                logger = self.logger
                                                )

        np.save("perturbation_in.npy", perturbation_distances_in)
        np.save("perturbation_out.npy", perturbation_distances_out)



        perturbed_dist = np.concatenate([perturbation_distances_in , perturbation_distances_out], axis=0)


        # create thresholds
        min_signal_val = np.min(perturbed_dist)
        max_signal_val = np.max(perturbed_dist)
        thresholds = np.linspace(min_signal_val, max_signal_val,1000)
        num_threshold = len(thresholds)


        perturbation_distances_in_n= self.normalize(perturbation_distances_in, min_signal_val, max_signal_val)
        perturbation_distances_out_n= self.normalize(perturbation_distances_out, min_signal_val, max_signal_val)

        perturbation_distances_in_tr = self.transformation(perturbation_distances_in_n)
        perturbation_distances_out_tr = self.transformation(perturbation_distances_out_n)


        # compute the signals for the in-members and out-members
        member_signals = (np.array(perturbation_distances_in_tr).reshape(-1, 1).repeat(num_threshold, 1).T)
        non_member_signals = (np.array(perturbation_distances_out_tr).reshape(-1, 1).repeat(num_threshold, 1).T)

        member_preds = np.greater(member_signals, thresholds[:, np.newaxis])
        non_member_preds = np.greater(non_member_signals, thresholds[:, np.newaxis])


        # what does the attack predict on test and train dataset
        predictions = np.concatenate([member_preds, non_member_preds], axis=1)
        # set true labels for being in the training dataset
        true_labels = np.concatenate(
            [
                np.ones(len(perturbation_distances_in)),
                np.zeros(len(perturbation_distances_out)),
            ]
        )
        signal_values = np.concatenate(
            [perturbation_distances_in_tr, perturbation_distances_out_tr]
        )

        # compute ROC, TP, TN etc
        return CombinedMetricResult(
            predicted_labels=predictions,
            true_labels=true_labels,
            predictions_proba=None,
            signal_values=signal_values,
        )


    def transformation( self: Self, a: np.ndarray) -> np.ndarray:
        """Apply log transformation to the input array.

        Parameters
        ----------
        a : np.ndarray
            The input array.

        Returns
        -------
        np.ndarray
            The transformed array.

        """
        epsilon = 1e-10  # Small constant to avoid division by zero

        # Applying log transformation with epsilon regularization
        return np.log((a + epsilon) / (1 - a + epsilon))

    def normalize(self: Self, x:np.ndarray, min_value:float, max_value: float) -> np.ndarray:
        """Normalize the input array.

        Parameters
        ----------
        x : np.ndarray
            The input array.
        min_value : float
            The minimum value for normalization.
        max_value : float
            The maximum value for normalization.

        Returns
        -------
        np.ndarray
            The normalized array.

        """
        near_one = 0.999999
        return ((x - min_value) / (max_value-min_value) * near_one)
