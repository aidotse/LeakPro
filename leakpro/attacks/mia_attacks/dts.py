"""Implementation of the Deep Time Series (DTS) attack.

Unlike other time series attacks, DTS does not rely on signal extraction.
Instead, it utilizes deep learing to automatically map raw time series to membership labels.
This process involves constructing a Membership Inference Classification (MIC) dataset
and training a binary time series classifier (MIC model) to infer membership.
"""

from typing import Any, Dict, Literal

import numpy as np
import torch
from pydantic import BaseModel, Field, model_validator
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, TensorDataset
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.dts_mia_classifier.mi_classifier import MIClassifier
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
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
        online: bool = Field(default=True, description="Online vs offline attack: whether the shadow models' training data includes the audit set (online) or excludes it (offline)")  # noqa: E501
        clf_model: Literal["LSTM", "InceptionTime"] = Field(default="LSTM", description="MIC model architecture to use [LSTM, InceptionTime]")  # noqa: E501
        clf_model_kwargs: Dict[str, Any] = Field(default=None, description="Dictionary of additional keyword arguments passed to the MIC model constructor. See LeakPro/leakpro/attacks/utils/dts_mia_classifier/models for possible/default arguments")  # noqa: E501
        clf_data_fraction: float = Field(default=0.1, ge=0.0, le=1.0, description="Fraction of shadow population to predict for each shadow model and append to the MIC dataset")  # noqa: E501
        clf_batch_size: int = Field(default=128, ge=0, description="The batch size to use when training MIC model")
        clf_max_epochs: int = Field(default=32, ge=1, description="The maximum amount of epochs when training MIC model")
        clf_val_fraction: float = Field(default=0.2, ge=0.05, le=0.5, description="Fraction of the MIC dataset to use as validation for early stopping")  # noqa: E501
        clf_early_stopping_patience: int = Field(default=2, ge=0, description="The maximum allowed number of epochs without validation loss improvement when training MIC model")  # noqa: E501
        clf_fit_verbose: Literal[0, 1] = Field(default=0, description="The amount of information (0 = nothing, 1 = loss and accuracy per epoch) to print when training the MIC model")  # noqa: E501

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

    def __init__(self: Self,
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

    def description(self: Self) -> dict:
        """Return a description of the attack."""
        title_str = "Deep Time Series Attack"

        reference_str = "Johansson N., & Olsson T. Privacy Risks in Time Series Models:  \
        Membership Inference in Deep Learning-Based Time Series Forecasting Models. 2025."

        summary_str = "The DTS attack is a time series membership inference attack based on predictions of a black-box model"

        detailed_str = "The attack is executed according to: \
            1. A fraction of the target model dataset is sampled to be included (in-) or excluded (out-) \
            from the shadow model training dataset. \
            2. Shadow models' (raw) predictions and corresponding target (true horizon), together with the label (in/out) \
            are used to construct MIC samples. \
            3. A time series binary classifier (MIC model) is trained on the extracted MIC data. \
            4. The MIC model is evaluated on the target model's outputs on the audit set, where the resulting confidences \
            are used to classify in-members and out-members."

        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }

    def create_MIC_dataset(self: Self) -> TensorDataset:  # noqa: N802
        """Construct and return a dataset for training the MIC model."""
        mic_features = []
        mic_labels = []

        metadata = ShadowModelHandler().get_shadow_model_metadata(self.shadow_model_indices)
        shadow_models_in_indices = [data.train_indices for data in metadata]

        for _, (shadow_model, in_indices) in tqdm(enumerate(zip(self.shadow_models, shadow_models_in_indices)),
                                                    total=len(self.shadow_models),
                                                    desc=f"Constructing MIC dataset from {self.num_shadow_models} shadow models' forecasts"):  # noqa: E501

            # Randomly select specified fraction of indices from shadow population
            data_size = int(len(self.attack_data_indices)*self.clf_data_fraction)
            data_indices = np.random.choice(self.attack_data_indices, data_size, replace=False)
            data_loader = self.handler.get_dataloader(data_indices, batch_size=self.clf_batch_size, shuffle=False)
            assert isinstance(data_loader.sampler, SequentialSampler), "DataLoader must not shuffle data to maintain order of indices"  # noqa: E501

            # Inference and MIC data collection
            with torch.no_grad():
                for (x, y) in data_loader:
                    preds = shadow_model.get_logits(x)
                    mic_features.extend(torch.cat((y, torch.tensor(preds)), dim=-1))

            # Create membership mask for shadow model and use as labels
            in_set = set(in_indices)
            mask = torch.tensor([1 if idx in in_set else 0 for idx in data_indices])
            mic_labels.append(mask)

        # Concatenate and return MIC dataset
        X = torch.stack(mic_features, dim=0)  # noqa: N806
        y = torch.cat(mic_labels, dim=0).float().view(-1, 1)
        return TensorDataset(X, y)

    def get_target_MIC_features(self: Self) -> torch.Tensor:  # noqa: N802
        """Get the MIC features from the target model on the audit set."""
        mic_features = []

        data_loader = self.handler.get_dataloader(self.audit_data_indices, batch_size=self.clf_batch_size, shuffle=False)
        assert isinstance(data_loader.sampler, SequentialSampler), "DataLoader must not shuffle data to maintain order of indices"

        # Forecast entire audit set and get the pred respectively target time series as MIC features
        with torch.no_grad():
            for x, y in tqdm(data_loader, desc="Extracting MIC features from target model"):
                preds = self.target_model.get_logits(x)
                mic_features.extend(torch.cat((y, torch.tensor(preds)), dim=-1))

        return torch.stack(mic_features, dim=0)

    def prepare_attack(self: Self) -> None:
        """Prepares data to obtain metric on the target model and dataset, using MIC model trained on the auxiliary model/dataset.

        It selects a balanced subset of data samples from in-group and out-group members
        of the audit dataset, prepares the data for evaluation, and extracts MIC data
        for both shadow models (full data) and the target model (features only, no labels).
        """

        self.attack_data_indices = self.sample_indices_from_population(include_train_indices = self.online,
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

        # Get MIC dataset and target MIC features
        self.mic_data = self.create_MIC_dataset()
        self.target_features = self.get_target_MIC_features()

    def run_attack(self: Self) -> MIAResult:
        """Runs the attack on the target model and dataset and assess privacy risks or data leakage.

        This method evaluates how the target model's output (predicted time series) for a specific dataset
        compares to the output of shadow models to determine if the dataset was part of the model's training data or not.
        This is approximated by training and assessing a deep membership classifier (MIC model).

        Returns
        -------
        Result(s) of the metric. An object containing the metric results, including predictions,
        true labels, and signal values (MIC model confidences).

        """

        # Use sklearn's train_test_split to split MIC dataset into train and val sets
        data_indices = np.arange(len(self.mic_data))
        train_indices, val_indices = train_test_split(data_indices, test_size=self.clf_val_fraction)

        train_subset = Subset(self.mic_data, train_indices)
        val_subset = Subset(self.mic_data, val_indices)

        train_loader = DataLoader(train_subset, batch_size=self.clf_batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=self.clf_batch_size, shuffle=False)

        # Init and train MIC model
        _, seq_len, num_variables = self.mic_data.tensors[0].shape
        mi_classifier = MIClassifier(seq_len, num_variables, self.clf_model, self.clf_model_kwargs)
        mi_classifier.fit(
            train_loader,
            val_loader,
            self.clf_max_epochs,
            self.clf_early_stopping_patience,
            verbose=self.clf_fit_verbose
        )

        # Iterate over all target samples and get MIC model predictions as membership scores
        score = mi_classifier.predict(self.target_features, self.clf_batch_size)    # Values between 0 and 1 since classifier utilizes sigmoid output activation  # noqa: E501
        score = score.flatten()

        # Split the score array into two parts based on membership: in (training) and out (non-training)
        self.in_member_signals = score[self.in_members].reshape(-1,1)  # Scores for known training data members
        self.out_member_signals = score[self.out_members].reshape(-1,1)  # Scores for non-training data members

        # Prepare true labels array, marking 1 for training data and 0 for non-training data
        true_labels = np.concatenate(
            [np.ones(len(self.in_member_signals)), np.zeros(len(self.out_member_signals))]
        )

        # Combine all signal values for further analysis
        signal_values = np.concatenate([self.in_member_signals, self.out_member_signals])

        # Return a result object containing predictions, true labels, and the signal values for further evaluation
        return MIAResult.from_full_scores(true_membership=true_labels,
                                          signal_values=signal_values,
                                          result_name="DTS",
                                          metadata=self.configs.model_dump())
