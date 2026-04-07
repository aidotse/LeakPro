"""Implementation of the Deep Time Series attack."""

import numpy as np
import torch

from pydantic import BaseModel, Field, model_validator
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader, TensorDataset
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm
from typing import Literal, Dict, Any

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.attacks.utils.dts_mia_classifier.model_api import MIClassifier
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger

class AttackDTS(AbstractMIA):
    """Implementation of the Deep Time Series attack."""

    class AttackConfig(BaseModel):
        """Configuration for the Deep Time Series attack."""

        num_shadow_models: int = Field(default=16, ge=1, description="Number of shadow models")
        training_data_fraction: float = Field(default=0.5, ge=0.0, le=1.0, description="Part of available attack data to use for shadow models")  # noqa: E501
        online: bool = Field(default=True, description="Online vs offline attack: whether the shadow models' training data includes the audit set (online) or excludes it (offline)")
        clf_model: Literal["LSTM", "InceptionTimeS"] = Field(default="LSTM", description="MI classifier model to use [LSTM, InceptionTime]")    # TODO: Rename IT after new runs
        clf_model_kwargs: Dict[str, Any] = Field(default=None, description="Dictionary of additional keyword arguments passed to the classifier model constructor. See LeakPro/leakpro/attacks/utils/clf_mia_classifier/models for possible/default arguments")
        clf_data_fraction: float = Field(default=0.1, ge=0.0, le=1.0, description="Fraction of shadow population to predict for each shadow model and append to the MI classifier data set")
        clf_batch_size: int = Field(default=128, ge=0, description="The batch size to use when training MI classifier")
        clf_max_epochs: int = Field(default=32, ge=1, description="The maximum amount of epochs when training MI classifier")
        clf_val_fraction: float = Field(default=0.2, ge=0.05, le=0.5, description="Fraction of the MI classifier data set to use as validation for early stopping") 
        clf_early_stopping_patience: int = Field(default=2, ge=0, description="The maximum allowed number of epochs without validation loss improvement when training MI classifier")
        clf_fit_verbose: Literal[0, 1] = Field(default=0, description="The amount of information to print when training the MI classifier")
        individual_mia: bool = Field(default=False, description="Run individual-level MIA.")

        @model_validator(mode="after")
        def check_num_shadow_models_if_online(self) -> Self:
            """Check if the number of shadow models is at least 2 when online is True.

            Returns
            -------
                Config: The attack configuration.

            Raises
            ------
                ValueError: If online is True and the number of shadow models is less than 2.

            """
            if self.online and self.num_shadow_models < 2:
                raise ValueError("When online is True, num_shadow_models must be >= 2")
            return self

    def __init__(self:Self,
                 handler: MIAHandler,
                 configs: dict
                 ) -> None:
        """Initialize the Deep Time Series attack.

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

        if self.online is False and self.population_size == self.audit_size:
            raise ValueError("The audit dataset is the same size as the population dataset. \
                    There is no data left for the shadow models.")

        self.shadow_models = []

    def description(self:Self) -> dict: 
        """Return a description of the attack."""
        title_str = "Deep Time Series Attack"

        reference_str = "Johansson N., & Olsson T. (2025). Privacy risks in time series models: Membership inference in deep learning-based time series forecasting models"

        summary_str = "DTS-MIA is a membership inference attack based on predictions of a black-box model"

        detailed_str = "The attack is executed according to: \
            1. A fraction of the target model dataset is sampled to be included (in-) or excluded (out-) \
            from the shadow model training dataset. \
            2. Shadow models' (raw) predictions and corresponding target, together with the label (in/out) are used to construct MI classification samples. \
            3. A time series binary classification model is trained on the extracted MI classification data. \
            4. The MI classifier is applied over the target model outputs, and the resulting confidences are used to classify in-members and out-members. \
            5. The attack is evaluated on an audit dataset to determine the attack performance."

        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }
    
    def create_MI_classifier_dataset(self:Self)->TensorDataset:
        mi_features = []
        mi_labels = []

        metadata = ShadowModelHandler().get_shadow_model_metadata(self.shadow_model_indices)
        shadow_models_in_indices = [data.train_indices for data in metadata]

        for _, (shadow_model, in_indices) in tqdm(enumerate(zip(self.shadow_models, shadow_models_in_indices)),
                                                    total=len(self.shadow_models),
                                                    desc=f"Constructing MI classifier dataset from {self.num_shadow_models} shadow models' forecasts"):

            # Select specified fraction of random indices from audit population
            data_size = int(len(self.attack_data_indices)*self.clf_data_fraction)
            data_indices = np.random.choice(self.attack_data_indices, data_size, replace=False)
            data_loader = self.handler.get_dataloader(data_indices, batch_size=self.clf_batch_size, shuffle=False)
            assert isinstance(data_loader.sampler, SequentialSampler), "DataLoader must not shuffle data to maintain order of indices"

            # Inference and MI data collection
            with torch.no_grad():
                for (x, y) in data_loader:
                    preds = shadow_model.get_logits(x)
                    mi_features.extend(torch.cat((y, torch.tensor(preds)), dim=-1))

            # Create membership mask for shadow model and use as labels
            in_set = set(in_indices)
            mask = torch.tensor([1 if idx in in_set else 0 for idx in data_indices])
            mi_labels.append(mask)

        # Concatenate and return dataset
        X = torch.stack(mi_features, dim=0)
        y = torch.cat(mi_labels, dim=0).float().view(-1, 1)
        return TensorDataset(X, y)
    
    def get_target_model_MI_features(self: Self) -> torch.Tensor:
        mi_features = []
        data_loader = self.handler.get_dataloader(self.audit_data_indices, batch_size=self.clf_batch_size, shuffle=False)
        assert isinstance(data_loader.sampler, SequentialSampler), "DataLoader must not shuffle data to maintain order of indices"

        # Forecast entire audit set and get the pred respectively target time series as MI features
        with torch.no_grad():
            for x, y in tqdm(data_loader, desc="Generating MI features from target model"):
                preds = self.target_model.get_logits(x)
                mi_features.extend(torch.cat((y, torch.tensor(preds)), dim=-1))

        X = torch.stack(mi_features, dim=0)
        return X

    def prepare_attack(self:Self)->None:
        """Prepares data to obtain metric on the target model and dataset, using MI classifier trained on the auxiliary model/dataset.

        It selects a balanced subset of data samples from in-group and out-group members
        of the audit dataset, prepares the data for evaluation, and extracts MI classifier data
        for both shadow models (full data) and the target model (features only, no labels).
        """

        self.attack_data_indices = self.sample_indices_from_population(include_aux_indices = not self.online,
                                                                       include_train_indices = self.online,
                                                                       include_test_indices = self.online)

        self.shadow_model_indices = ShadowModelHandler().create_shadow_models(num_models = self.num_shadow_models,
                                                                              shadow_population = self.attack_data_indices,
                                                                              training_fraction = self.training_data_fraction,
                                                                              online = self.online)

        self.shadow_models, _ = ShadowModelHandler().get_shadow_models(self.shadow_model_indices)

        logger.info("Create masks for all IN and OUT samples")
        self.in_indices_masks = ShadowModelHandler().get_in_indices_mask(self.shadow_model_indices, self.audit_dataset["data"])

        if self.online:
            # Exclude all audit points that have either no IN or OUT samples
            num_shadow_models_seen_points = np.sum(self.in_indices_masks, axis=1)
            mask = (num_shadow_models_seen_points > 0) & (num_shadow_models_seen_points < self.num_shadow_models)

            # Filter the audit data
            self.audit_data_indices = self.audit_dataset["data"][mask]
            self.in_indices_masks = self.in_indices_masks[mask, :]

            # Filter IN and OUT members
            self.in_members = np.arange(np.sum(mask[self.audit_dataset["in_members"]]))
            num_out_members = np.sum(mask[self.audit_dataset["out_members"]])
            self.out_members = np.arange(len(self.in_members), len(self.in_members) + num_out_members)

            assert len(self.audit_data_indices) == len(self.in_members) + len(self.out_members)

            if len(self.audit_data_indices) == 0:
                raise ValueError("No points in the audit dataset are used for the shadow models")

        else:
            self.audit_data_indices = self.audit_dataset["data"]
            self.in_members = self.audit_dataset["in_members"]
            self.out_members = self.audit_dataset["out_members"]

        # Check offline attack for possible IN- sample(s)
        if not self.online:
            count_in_samples = np.count_nonzero(self.in_indices_masks)
            if count_in_samples > 0:
                logger.info(f"Some shadow model(s) contains {count_in_samples} IN samples in total for the model(s)")
                logger.info("This is not an offline attack!")

        # Get MI classifier data and target errors
        self.mi_classifier_data = self.create_MI_classifier_dataset()
        self.target_features = self.get_target_model_MI_features()

    def run_attack(self:Self) -> MIAResult:
        """Runs the attack on the target model and dataset and assess privacy risks or data leakage.

        This method evaluates how the target model's output (predicted time series) for a specific dataset
        compares to the output of shadow models to determine if the dataset was part of the model's training data or not. 
        This is approximated by training and assessing a deep MI classifier model.

        Returns
        -------
        Result(s) of the metric. An object containing the metric results, including predictions,
        true labels, and signal values (MI classifier confidences).

        """

        # Use sklearn's train_test_split to split MI classification data into train and val indices
        data_indices = np.arange(len(self.mi_classifier_data))
        train_indices, val_indices = train_test_split(data_indices, test_size=self.clf_val_fraction)

        train_subset = Subset(self.mi_classifier_data, train_indices)
        val_subset = Subset(self.mi_classifier_data, val_indices)

        train_loader = DataLoader(train_subset, batch_size=self.clf_batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=self.clf_batch_size, shuffle=False)

        # Init and train MI Classifier
        _, seq_len, num_variables = self.mi_classifier_data.tensors[0].shape
        mi_classifier = MIClassifier(seq_len, num_variables, self.clf_model, self.clf_model_kwargs)
        mi_classifier.fit(train_loader, val_loader, self.clf_max_epochs, self.clf_early_stopping_patience, verbose=self.clf_fit_verbose)

        # Iterate over all target samples and get MI Classifier predictions as MI scores
        score = mi_classifier.predict(self.target_features, self.clf_batch_size)    # Values between 0 and 1 since classifier utilizes sigmoid output
        score = score.flatten()

        # Split the score array into two parts based on membership: in (training) and out (non-training)
        self.in_member_signals = score[self.in_members].reshape(-1,1)  # Scores for known training data members
        self.out_member_signals = score[self.out_members].reshape(-1,1)  # Scores for non-training data members

        if self.individual_mia:
            samples_per_individual = self.handler.population.samples_per_individual
            in_num_individuals = len(self.in_member_signals) // samples_per_individual
            out_num_individuals = len(self.out_member_signals) // samples_per_individual
            num_individuals = in_num_individuals + out_num_individuals
            logger.info(f"Running individual-level MI on {num_individuals} individuals "
                        f"with {samples_per_individual} samples per individual.")
            self.in_member_signals = (self.in_member_signals
                                      .reshape((in_num_individuals, samples_per_individual))
                                      .mean(axis=1, keepdims=True))
            self.out_member_signals = (self.out_member_signals
                                       .reshape((out_num_individuals, samples_per_individual))
                                       .mean(axis=1, keepdims=True))
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
                                          result_name="DTS-MIA",
                                          metadata=self.configs.model_dump())
