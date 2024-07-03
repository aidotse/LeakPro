import pickle  # noqa: D100

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import DataLoader

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.import_helper import Self
from leakpro.metrics.attack_result import CombinedMetricResult
from leakpro.signals.signal import HopSkipJumpDistance
from leakpro.user_inputs.abstract_input_handler import AbstractInputHandler


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
        self.verbose = False # configs.get("verbose", True)
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

        # Get all available indices for auxiliary dataset
        shadow_data_index = self.sample_indices_from_population(include_train_indices = False,
                                                                include_test_indices = False)


        # create auxiliary dataset
        shadow_data_size = len(shadow_data_index)
        shadow_train_data_size = int(shadow_data_size * self.attack_data_fraction)
        shadow_train_data_indices = np.random.choice(shadow_data_index, shadow_train_data_size, replace=False)
        shadow_test_data_indices = np.setdiff1d(shadow_data_index, shadow_train_data_indices)

        self.shadow_train_dataset = self.population.subset(shadow_train_data_indices)
        self.shadow_test_dataset = self.population.subset(shadow_test_data_indices)

        #--------------------------------------------------------
        # Train and load shadow model
        #--------------------------------------------------------
        self.logger.info(f"Training shadow models on {len(shadow_train_data_indices)} points")
        self.shadow_model_indices = ShadowModelHandler().create_shadow_models(self.num_shadow_models,
                                                                         shadow_train_data_indices,
                                                                         training_fraction = 1.0)
        # load shadow models
        self.shadow_model, self.shadow_model_indices = ShadowModelHandler().get_shadow_models(self.shadow_model_indices)



    def run_attack(self:Self) -> CombinedMetricResult:
        """Run the attack and return the combined metric result.

        Returns
        -------
            CombinedMetricResult: The combined metric result containing predicted labels, true labels,
            predictions probabilities, and signal values.

        """
        shadow_model = self.shadow_model[0]

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
        sanity_check = True
        if sanity_check:
            self.sanity_check(perturbation_distances_in, perturbation_distances_out)
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

    def sanity_check(self: Self, shadow_distances_in: np.ndarray, shadow_distances_out: np.ndarray) -> None:
        """Perform a sanity check on the perturbation distances.

        Parameters
        ----------
        shadow_distances_in : np.ndarray
            The perturbation distances for the in-members.
        shadow_distances_out : np.ndarray
            The perturbation distances for the out-members.

        Returns
        -------
        None

        """
        self.logger.info("Performing sanity check on the perturbation distances")
        assert shadow_distances_in.shape[0] == len(self.shadow_train_dataset), "Perturbation distances in not equal to the shadow train dataset size"  # noqa: E501
        assert shadow_distances_out.shape[0] == len(self.shadow_test_dataset), "Perturbation distances out not equal to the shadow test dataset size"  # noqa: E501
        assert shadow_distances_in.shape[1] == 1, "Perturbation distances in not equal to 1"
        assert shadow_distances_out.shape[1] == 1, "Perturbation distances out not equal to 1"
        self.logger.info("Sanity check passed")

        target_train_indices = self.target_model_metadata["train_indices"]
        target_test_indices = self.target_model_metadata["test_indices"]

        target_train_dataset = self.population.subset(target_train_indices)
        target_test_dataset = self.population.subset(target_test_indices)

        target_train_loader = DataLoader(target_train_dataset, batch_size=self.batch_size, shuffle=True)
        target_test_loader = DataLoader(target_test_dataset, batch_size=self.batch_size, shuffle=True)

        _, in_data_distances = self.signal( model = self.target_model,
                                                    data_loader = target_train_loader,
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
        _ , out_data_distances = self.signal(model = self.target_model,
                                            data_loader = target_test_loader,
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



        # Find the optimal threshold
        all_distances = np.concatenate([in_data_distances, out_data_distances])
        thresholds = np.sort(all_distances)

        best_threshold = None
        best_statistic = -np.inf

        for threshold in thresholds:

            # Use the Kolmogorov-Smirnov statistic as the metric
            statistic, _ = ks_2samp(in_data_distances, out_data_distances)

            if statistic > best_statistic:
                best_statistic = statistic
                best_threshold = threshold

        self.logger(f"Best threshold: {best_threshold}")



        # Apply the threshold to classify shadow model's data
        shadow_in_predictions = [self.is_member(dist, best_threshold) for dist in shadow_distances_in]
        shadow_out_predictions = [self.is_member(dist, best_threshold) for dist in shadow_distances_out]


        # True labels for shadow model data
        # 1 indicates in-data, 0 indicates out-data
        shadow_in_labels = np.ones(len(shadow_distances_in))
        shadow_out_labels = np.zeros(len(shadow_distances_out))

        # Combine predictions and labels
        all_predictions = np.concatenate([shadow_in_predictions, shadow_out_predictions])
        all_labels = np.concatenate([shadow_in_labels, shadow_out_labels])

        # Calculate precision and recall
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)

        self.logger(f"Precision: {precision:.4f}")
        self.logger(f"Recall: {recall:.4f}")



    def is_member(self: Self, distance: np.ndarray, threshold: float) -> bool:
        """Check if the distance is above the threshold.

        Parameters
        ----------
        distance : np.ndarray
            The distance to check.
        threshold : float
            The threshold value.

        Returns
        -------
        bool
            True if the distance is above the threshold, False otherwise.

        """
        return distance >= threshold

