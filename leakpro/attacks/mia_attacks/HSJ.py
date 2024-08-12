import pickle  # noqa: D100
from itertools import chain

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import precision_score, recall_score, roc_curve
from torch.utils.data import DataLoader

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.mia_attacks.delete2 import hop_skip_jump_attack
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

        self.shadow_data_fraction = configs.get("shadow_data_fraction", 1)
        self.shadow_train_fraction = configs.get("shadow_train_fraction", 0.5)
        self.num_shadow_models = configs.get("num_shadow_models", 1)
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

        self.reproducing_paper_results = configs.get("reproducing_paper_results", True)
        self.paper_data_fr = configs.get("paper_data_fr", 0.83)
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
        self._validate_num_shadow_models()
        self._validate_shadow_data_fraction()
        self._validate_shadow_train_fraction()
        self._validate_y_target()

    def _validate_norm(self:Self) -> None:
        """Validate the norm value."""
        valid_norm_values = [1, 2, np.inf]
        if self.norm not in valid_norm_values:
            raise ValueError(f"Invalid norm value: {self.norm}. Must be one of {valid_norm_values}")

    def _validate_stepsize_search(self:Self) -> None:
        """Validate the stepsize_search value."""
        valid_stepsize_search_values = ["geometric_progression", "grid_search"]
        if self.stepsize_search not in valid_stepsize_search_values:
            raise ValueError(f"Invalid stepsize_search value: {self.stepsize_search}. Must be one of {valid_stepsize_search_values}")

    def _validate_constraint(self:Self) -> None:
        """Validate the constraint value."""
        valid_constraint_values = [1, 2]
        if self.constraint not in valid_constraint_values:
            raise ValueError(f"Invalid constraint value: {self.constraint}. Must be one of {valid_constraint_values}")

    def _validate_initial_num_evals(self) -> None:
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

    def _validate_num_shadow_models(self:Self) -> None:
        """Validate the num_shadow_models value."""
        if self.num_shadow_models <= 0:
            raise ValueError("num_shadow_models must be greater than 0")

    def _validate_shadow_data_fraction(self:Self) -> None:
        """Validate the shadow_data_fraction value."""
        if self.shadow_data_fraction <= 0 or self.shadow_data_fraction > 1:
            raise ValueError("shadow_data_fraction must be in the range (0, 1]")

    def _validate_shadow_train_fraction(self:Self) -> None:
        """Validate the shadow_train_fraction value."""
        if self.shadow_train_fraction <= 0 or self.shadow_train_fraction > 1:
            raise ValueError("shadow_train_fraction must be in the range (0, 1]")

    def _validate_y_target(self:Self) -> None:
        """Validate the y_target value."""
        num_classes = self.target_model_metadata["init_params"]["num_classes"]
        if self.y_target is not None and self.y_target not in range(num_classes):
            raise ValueError("y_target must be an integer and in the range of the number of classes in the target model.")


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
        self.logger.info("Preparing the data for Hop Skip Jump attack")

        # Get all available indices for auxiliary dataset
        aux_data_index = self.sample_indices_from_population(include_train_indices = False,
                                                                include_test_indices = False)


        # create shadow dataset from aux data
        aux_data_size = len(aux_data_index)
        shadow_data_size = int(aux_data_size * self.shadow_data_fraction)
        shadow_data_indices = np.random.choice(aux_data_index, shadow_data_size, replace=False)

        # Split the shadow data into training and test datasets
        split_point = int(len(shadow_data_indices) * self.shadow_train_fraction)
        self.shadow_train_data_indices = shadow_data_indices[:split_point]
        self.shadow_test_data_indices = shadow_data_indices[split_point:]

        self.shadow_train_dataset = self.population.subset(self.shadow_train_data_indices)
        self.shadow_test_dataset = self.population.subset(self.shadow_test_data_indices)

        #--------------------------------------------------------
        # Train and load shadow model
        #--------------------------------------------------------
        self.logger.info(f"Training shadow models on {len(self.shadow_train_data_indices)} points")
        self.shadow_model_indices = ShadowModelHandler().create_shadow_models(self.num_shadow_models,
                                                                         self.shadow_train_data_indices,
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
        # shadow_model = self.target_model


        # if self.reproducing_paper_results:
        aux_data_size = int(len(self.shadow_train_data_indices)*self.paper_data_fr)
        attack_in_indices = np.random.choice(self.shadow_train_data_indices, aux_data_size, replace=False)
        attack_out_indices = np.random.choice(self.shadow_test_data_indices, aux_data_size, replace=False)

        attack_in_dataset = self.population.subset(attack_in_indices)
        attack_out_dataset = self.population.subset(attack_out_indices)

        attack_in_dataloader = DataLoader(attack_in_dataset, batch_size=self.batch_size, shuffle=False)
        attack_out_dataloader = DataLoader(attack_out_dataset, batch_size=self.batch_size, shuffle=False)
        # else:
        #     attack_in_dataloader = DataLoader(self.shadow_train_dataset, batch_size=self.batch_size, shuffle=False)
        #     attack_out_dataloader = DataLoader(self.shadow_test_dataset, batch_size=self.batch_size, shuffle=False)


        self.logger.info("Running Hop Skip Jump distance attack, in data ")

        # compute the perturbation distances for the in-members of the shadow model
        _ , perturbation_distances_in = self.signal(shadow_model,
                                                    attack_in_dataloader,
                                                    self.logger,
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

        # compute the perturbation distances for the out-members of the shadow model
        self.logger.info("Running Hop Skip Jump distance attack, out data")
        _ , perturbation_distances_out = self.signal( shadow_model,
                                                attack_out_dataloader,
                                                self.logger,
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

        np.save("perturbation_in.npy", perturbation_distances_in)
        np.save("perturbation_out.npy", perturbation_distances_out)
        # perturbation_distances_in = np.load("perturbation_in.npy")
        # perturbation_distances_out = np.load("perturbation_out.npy")

        if self.reproducing_paper_results:

            # getting HSJ for in and out data in target model from cleverhans and save them for sanity check
            self.logger.info("first sanity check")
            hans_distances_in = self.sanity_check_hansclevernce(attack_in_dataset, self.target_model)
            # hans_distances_out = self.sanity_check_hansclevernce(attack_out_dataset, self.target_model.model_obj)
            np.save("hans_distances_in_target.npy", hans_distances_in)

            self.logger.info("Second sanity check")
            # getting HSJ for in and out data in target model and save them for sanity check
            self.sanity_check(perturbation_distances_in, perturbation_distances_out)

            # np.save("hans_distances_out.npy", hans_distances_out)



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
        target_train_indices = self.target_model_metadata["train_indices"]
        target_test_indices = self.target_model_metadata["test_indices"]

        aux_data_size = int(len(target_train_indices)*self.paper_data_fr)
        attack_in_indices = np.random.choice(target_train_indices, aux_data_size, replace=False)
        attack_out_indices = np.random.choice(target_test_indices, aux_data_size, replace=False)

        attack_in_dataset = self.population.subset(attack_in_indices)
        attack_out_dataset = self.population.subset(attack_out_indices)

        attack_in_dataloader = DataLoader(attack_in_dataset, batch_size=self.batch_size, shuffle=False)
        attack_out_dataloader = DataLoader(attack_out_dataset, batch_size=self.batch_size, shuffle=False)

        _, in_data_distances = self.signal( self.target_model,
                                            attack_in_dataloader,
                                            self.logger,
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
        _ , out_data_distances = self.signal(self.target_model,
                                            attack_out_dataloader,
                                            self.logger,
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
        np.save("in_data_distances_target_load_init.npy", in_data_distances)
        np.save("out_data_distances_target.npy", out_data_distances)
        # in_data_distances = np.load("perturbation_in.npy")
        out_data_distances = np.load("perturbation_out.npy")

        # Find the optimal threshold
        self.best_threshold_precision(self.find_threshold1(in_data_distances, out_data_distances),
                                       shadow_distances_in, shadow_distances_out)
        self.best_threshold_precision(self.find_threshold2(in_data_distances, out_data_distances),
                                      shadow_distances_in, shadow_distances_out)
        self.best_threshold_precision(self.find_threshold3(in_data_distances, out_data_distances),
                                      shadow_distances_in, shadow_distances_out)





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

    def find_threshold1(self: Self, in_data: np.ndarray, out_data: np.ndarray) -> float:
        """Find the optimal threshold based on the Kolmogorov-Smirnov statistic.

        Parameters
        ----------
        in_data : np.ndarray
            The in-data.
        out_data : np.ndarray
            The out-data.

        Returns
        -------
        float
            The optimal threshold.

        """
        # Find the optimal threshold
        all_distances = np.concatenate([in_data, out_data])
        thresholds = np.sort(all_distances)

        best_threshold = None
        best_statistic = -np.inf

        for threshold in thresholds:

            in_above_threshold = in_data >= threshold
            out_above_threshold = out_data >= threshold

            tp = np.sum(in_above_threshold)
            fn = np.sum(~in_above_threshold)
            fp = np.sum(out_above_threshold)
            tn = np.sum(~out_above_threshold)

            # Use the Kolmogorov-Smirnov statistic as the metric
            statistic, _ = ks_2samp(in_data, out_data)

            if statistic > best_statistic:
                best_statistic = statistic
                best_threshold = threshold
                best_tp = tp
                best_fn = fn
                best_fp = fp
                best_tn = tn


        self.logger.info(f"Best threshold1: {best_threshold}, with best tp: {best_tp}, best fn: {best_fn}, best fp: {best_fp}, best tn: {best_tn}")  # noqa: E501
        return best_threshold

    def find_threshold2(self: Self, in_data: np.ndarray, out_data: np.ndarray) -> float:
        """Find the optimal threshold based on the ROC curve.

        Parameters
        ----------
        in_data : np.ndarray
            The in-data.
        out_data : np.ndarray
            The out-data.

        Returns
        -------
        float
            The optimal threshold.

        """
        # Create label array
        labels = np.array([1] * len(in_data) + [0] * len(out_data))

        # Find optimal threshold
        fpr, tpr, thresholds = roc_curve(labels, np.concatenate((in_data, out_data)))
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        self.logger.info(f"Optimal Threshold2: {optimal_threshold}")

        return optimal_threshold


    def find_threshold3(self: Self, in_data: np.ndarray, out_data: np.ndarray) -> float:
        """Find the optimal threshold based on the KS test.

        Parameters
        ----------
        in_data : np.ndarray
            The in-data.
        out_data : np.ndarray
            The out-data.

        Returns
        -------
        float
            The optimal threshold.

        """
        # Merge
        all_data = np.concatenate((in_data, out_data))

        # Perform the KS test
        ks_stat, p_value = ks_2samp(in_data, out_data)

        # Find the optimal threshold
        thresholds = np.linspace(min(all_data), max(all_data), num=100)
        distances = []

        for threshold in thresholds:
            a_above = np.sum(in_data > threshold) / len(in_data)
            b_above = np.sum(out_data > threshold) / len(out_data)
            distances.append(abs(a_above - b_above))

        optimal_idx = np.argmax(distances)
        optimal_threshold = thresholds[optimal_idx]

        self.logger.info(f"Optimal Threshold3: {optimal_threshold}")

        return optimal_threshold

    def best_threshold_precision(self: Self,
                                 best_threshold: float,
                                 shadow_distances_in: np.ndarray
                                 , shadow_distances_out: np.ndarray) -> None:
        """Calculate precision, recall, and accuracy based on the best threshold.

        Parameters
        ----------
        best_threshold : float
            The best threshold value.
        shadow_distances_in : np.ndarray
            The distances of shadow model's in-data.
        shadow_distances_out : np.ndarray
            The distances of shadow model's out-data.

        Returns
        -------
        None

        """
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

        self.logger.info(f"Precision: {precision:.4f}")
        self.logger.info(f"Recall: {recall:.4f}")

        #stats
        in_above_threshold = shadow_distances_in >= best_threshold
        out_above_threshold = shadow_distances_out >= best_threshold

        tp = np.sum(in_above_threshold)
        fn = np.sum(~in_above_threshold)
        fp = np.sum(out_above_threshold)
        tn = np.sum(~out_above_threshold)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        self.logger.info(f"Accuracy: {accuracy:.4f}")

    def sanity_check_hansclevernce(self, dataset, model):
        """Sanity check the hop skip jump attack using the cleverhans library.

        Parameters
        ----------
        dataloder : DataLoader
            The dataloader for the dataset.
        model : torch.nn.Module
            The target model.

        Returns
        -------
        None

        """
        distances = []
        # dataset = dataloder.dataset
        new_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        attacked_model = model.model_obj
        attacked_model.eval()
        attacked_model.to("cpu")
        init = []
        distances = []

        for idx , (data, _) in enumerate(new_dataloader):
            if idx <15 :
                adv_x, distance, init_i = hop_skip_jump_attack(attacked_model, data,self.logger, 2, None, None, 100,10000,"geometric_progression", 64,
                                             1.0, 2, 128, True, -1, 1)

                distances.append(distance)
                init.append(init_i)
        numpy_list = [tensor.cpu().numpy() for tensor in init]
        final_numpy_array = np.concatenate(numpy_list)

        # Step 4: Save the final NumPy array
        np.save("init_hans_in.npy", final_numpy_array)

        flat_distances = list(chain.from_iterable(distances))
        # Convert tensors to numpy arrays, ensuring each tensor is at least one-dimensional
        nump_dist_list = [tensor.cpu().numpy() if tensor.dim() > 0 else np.array([tensor.cpu().item()]) for tensor in flat_distances]

        # Concatenate the numpy arrays if necessary
        final_numpy_array_dist = np.concatenate(nump_dist_list)

        # Save the final numpy array
        np.save("cleverhans_target_in_distance.npy", final_numpy_array_dist)
        return final_numpy_array_dist
