"""Implementation of the Ensemble Attack from "Improving Membership Inference Attacks against Classification Models"."""

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator
from warnings import filterwarnings
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning


from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.signals.signal import get_signal_from_name
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class AttackEnsemble(AbstractMIA):
    """Implementation of the Ensemble attack."""

    class AttackConfig(BaseModel):
        """Configuration for the Ensemble attack."""

        signal_names: list[str] = Field(default=["ModelRescaledLogits"], description="What signals to use.")
        individual_mia: bool = Field(default=False, description="Run individual-level MIA.")
        num_instances: int = Field(default=50, ge=1, description="Number of instances to run.")
        subset_size: int = Field(default=50, ge=1, description="Amount of datapoints within each data subset.")
        num_pairs: int = Field(default=20, ge=1, description="Number of pairs of subsets to create membership classifiers for.")
        num_runs: int = Field(default=5, ge=1, description="Number of runs for each subset pair.")
        audit: bool = Field(default=False, description="Audit mode implies that membership classifiers are trained on target model membership labels, otherwise shadow model labels.")
        training_data_fraction: float = Field(default=0.5, ge=0.0, le=1.0, description="Part of available attack data to use for shadow models")  # noqa: E501

    def __init__(self:Self,
                 handler: MIAHandler,
                 configs: dict
                 ) -> None:
        """Initialize the Ensemble attack.

        Args:
        ----
            handler (MIAHandler): The input handler object.
            configs (dict): Configuration parameters for the attack.

        """
        self.configs = self.AttackConfig() if configs is None else self.AttackConfig(**configs)

        # Initializes the parent metric
        super().__init__(handler)

        # Assign the configuration parameters to the object
        for key, value in self.configs.model_dump().items():
            setattr(self, key, value)

        self.shadow_models = []
        self.signals = [get_signal_from_name(signal_name) for signal_name in self.signal_names]
        self.online = True

    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "Ensemble attack"
        reference_str = "Shlomit Shachor, Natalia Razinkov, Abigail Goldsteen and Ariel Farkash. \
            Improving Membership Inference Attacks against Classification Models. (2024)."
        summary_str = "The Ensemble attack is a membership inference attack based on an ensemble of classifications models."
        detailed_str = "The attack is executed according to: \
            1. The shadow model training dataset is split into multiple non-overlapping subsets. \
            2. A set amount of pairs created using these subsets. \
            3. For multiple runs we randomly assign membership label to all datapoints in a pair.\
            4. For each run run multiple combinations of classification models."
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }

    def prepare_attack(self:Self) -> None:
        """Prepare data needed for running the attack on the target model and dataset.

        Signals are computed on the auxiliary model(s) and dataset.
        """

        filterwarnings(action='ignore', category=ConvergenceWarning)
        logger.info("Preparing shadow models for Ensemble attack")
        # Check number of shadow models that are available

        # sample dataset to compute histogram
        logger.info("Preparing attack data for training the Ensemble attack")

        # Get all available indices for attack dataset including training and test data
        self.attack_data_indices = self.sample_indices_from_population(include_aux_indices = not self.online,
                                                                       include_train_indices = self.online,
                                                                       include_test_indices = self.online)

        if not self.audit:
            # train shadow models
            logger.info(f"Check for {self.num_instances} shadow models (dataset: {len(self.attack_data_indices)} points)")
            self.shadow_model_indices = ShadowModelHandler().create_shadow_models(
                num_models = self.num_instances,
                shadow_population = self.attack_data_indices,
                training_fraction = self.training_data_fraction,
                online = self.online)
            # load shadow models
            self.shadow_models, _ = ShadowModelHandler().get_shadow_models(self.shadow_model_indices)

            self.in_indices_masks = ShadowModelHandler().get_in_indices_mask(self.shadow_model_indices, self.audit_dataset["data"])
            self.out_indices_masks = np.logical_not(self.in_indices_masks)
            


    def run_attack(self:Self) -> MIAResult:
        """Run the attack on the target model and dataset.

        Returns
        -------
            Result(s) of the metric.

        """
        if self.audit:
            logger.info("Running Ensemble shadow attack (audit mode)")
        else:
            logger.info("Running Ensemble shadow attack (attack mode)")
            

        ensemble_models = []
        for instance in range(self.num_instances):
            logger.info(f"Running instance number {instance+1}/{self.num_instances}")

            if self.audit:
                current_model = self.target_model
                in_indices = self.audit_dataset["in_members"]
                out_indices = self.audit_dataset["out_members"]
            else:
                # Get current shadow model
                current_model = self.shadow_models[instance]

                # Get indices which the current shadow model is trained or not trained on
                in_indices = self.attack_data_indices[self.in_indices_masks[:, instance]]
                out_indices = self.attack_data_indices[self.out_indices_masks[:, instance]]

            # Choose a subset of these to train the membership classifiers on for this instance
            in_indices = np.random.choice(in_indices, self.subset_size * self.num_pairs, replace=False)
            out_indices = np.random.choice(out_indices, self.subset_size * self.num_pairs, replace=False)

            # Create set of features and in/out label for each indices in subsets
            in_features = []
            out_features = []
            for signal, signal_name in zip(self.signals, self.signal_names):
                ts2vec_params = ([self.attack_data_indices] if signal_name == 'TS2VecLoss' else [])

                in_features.append(np.squeeze(signal([current_model],
                                                     self.handler,
                                                     in_indices,
                                                     *ts2vec_params)))
                out_features.append(np.squeeze(signal([current_model],
                                                      self.handler,
                                                      out_indices,
                                                      *ts2vec_params)))
            in_features = np.swapaxes(np.array(in_features), 0, 1)
            out_features = np.swapaxes(np.array(out_features), 0, 1)


            pair_models = []
            for pair_i in tqdm(range(self.num_pairs),
                               total=self.num_pairs,
                               desc="Training the best membership classifier for each pair"):
                pair_subset = list(range(pair_i * self.subset_size, (pair_i + 1) * self.subset_size))

                pair_features = np.vstack((in_features[pair_subset], out_features[pair_subset]))
                pair_label = np.hstack((np.full(self.subset_size, 0), np.full(self.subset_size, 1)))

                run_models = []
                run_auc = []
                for run_i in range(self.num_runs):
                    # Randomly split the pair 50-50 into train and test data for the membership classifier
                    features_train, features_test, label_train, label_test = train_test_split(
                            pair_features, pair_label, test_size=0.5)
                    
                    # Try each combination of scaler and model, record auc score on test set
                    for scaler in [StandardScaler, MinMaxScaler, RobustScaler]:
                        models = [RandomForestClassifier(),
                                  GradientBoostingClassifier(),
                                  LogisticRegression(),
                                  DecisionTreeClassifier(),
                                  KNeighborsClassifier(),
                                  MLPClassifier(hidden_layer_sizes=(512,100,64), max_iter=100),
                                  SVC(kernel="poly"),
                                  SVC(kernel="rbf"),
                                  SVC(kernel="sigmoid")]
                        
                        for model in models:
                            pipe = make_pipeline(scaler(), model)
                            pipe = pipe.fit(features_train, label_train)
                            
                            probs = pipe.predict(features_test)

                            run_models.append(pipe)
                            run_auc.append(roc_auc_score(label_test, probs))

                # Choose model with best ROC-AUC
                best_model = run_models[0]
                best_auc = 0.0
                for i in range(len(run_models)):
                    if run_auc[i] > best_auc:
                        best_auc = run_auc[i]
                        best_model = run_models[i]
                pair_models.append(best_model)
            ensemble_models.append(pair_models)
        
        self.audit_data_indices = self.audit_dataset["data"]
        self.in_members = self.audit_dataset["in_members"]
        self.out_members = self.audit_dataset["out_members"]

        features = []
        for signal, signal_name in zip(self.signals, self.signal_names):
            ts2vec_params = ([self.attack_data_indices] if signal_name == 'TS2VecLoss' else [])
            features.append(np.squeeze(signal([self.target_model],
                                              self.handler,
                                              self.audit_data_indices,
                                              *ts2vec_params)))
        features = np.swapaxes(np.array(features), 0, 1)
        
        # Average membership score over all instances and models
        self.score = np.zeros(features.shape[0])
        for best_models in ensemble_models:
            instance_score = np.zeros(features.shape[0])
            for model in best_models:
                instance_score += model.predict(features)
            self.score += instance_score / len(best_models)
        self.score = self.score / self.num_instances

        # Split the score array into two parts based on membership: in (training) and out (non-training)
        self.in_member_signals = self.score[self.in_members].reshape(-1,1)  # Scores for known training data members
        self.out_member_signals = self.score[self.out_members].reshape(-1,1)  # Scores for non-training data members

        if self.individual_mia:
            samples_per_individual = self.handler.population.samples_per_individual
            in_num_individuals = len(self.in_member_signals) // samples_per_individual 
            out_num_individuals = len(self.out_member_signals) // samples_per_individual
            num_individuals = in_num_individuals + out_num_individuals
            logger.info(f"Running individual-level MI on {num_individuals} individuals with {samples_per_individual} samples per individual.")

            self.in_member_signals = self.in_member_signals.reshape((in_num_individuals, samples_per_individual)).mean(axis=1, keepdims=True)
            self.out_member_signals = self.out_member_signals.reshape((out_num_individuals, samples_per_individual)).mean(axis=1, keepdims=True)
            self.audit_data_indices = np.arange(num_individuals)

        # Prepare true labels array, marking 1 for training data and 0 for non-training data
        true_labels = np.concatenate(
            [np.ones(len(self.in_member_signals)), np.zeros(len(self.out_member_signals))]
        )

        # Combine all signal values for further analysis
        signal_values = np.concatenate([self.in_member_signals, self.out_member_signals])

    
        # Return a result object containing predictions, true labels, and the signal values for further evaluation
        return MIAResult.from_full_scores(true_membership=true_labels,
                                    signal_values=signal_values,
                                    result_name="Ensemble",
                                    metadata=self.configs.model_dump())


