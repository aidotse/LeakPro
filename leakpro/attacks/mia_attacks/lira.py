"""Implementation of the LiRA attack."""

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator
from scipy.stats import norm
from tqdm import tqdm

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.utils.import_helper import Self
from leakpro.attacks.mia_attacks.lira_versions import lira_vectorized, lira_iterative

class AttackLiRA(AbstractMIA):
    """Implementation of the LiRA attack."""

    class AttackConfig(BaseModel):
        """Configuration for the LiRA attack."""

        num_shadow_models: int = Field(default=1, ge=1, description="Number of shadow models")
        training_data_fraction: float = Field(default=0.5, ge=0.0, le=1.0, description="Part of available attack data to use for shadow models")  # noqa: E501
        online: bool = Field(default=False, description="Online vs offline attack")
        var_calculation: Literal["carlini", "individual_carlini", "fixed"] = Field(default="carlini", description="Variance estimation method to use [carlini, individual_carlini, fixed]")  # noqa: E501
        vectorized: bool = Field(default=False, description="Compute shadow-model scores in a single vectorized pass (faster) instead of per-sample loop (safer).")

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
        """Initialize the LiRA attack.

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

    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "Likelihood Ratio Attack"

        reference_str = "Carlini N, et al. Membership Inference Attacks From First Principles"

        summary_str = "LiRA is a membership inference attack based on rescaled logits of a black-box model"

        detailed_str = "The attack is executed according to: \
            1. A fraction of the target model dataset is sampled to be included (in-) or excluded (out-) \
            from the shadow model training dataset. \
            2. The rescaled logits are used to estimate Gaussian distributions for in and out members \
            3. The thresholds are used to classify in-members and out-members. \
            4. The attack is evaluated on an audit dataset to determine the attack performance."

        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }

    def rescale_logits(self:Self, logits: np.ndarray, true_label:np.ndarray) -> np.ndarray:
        """Rescale the logits to a range of [0, 1].

        Args:
            logits (np.ndarray): The logits to be rescaled.
            true_label (np.ndarray): The true labels for the logits.

        Returns:
            np.ndarray: The rescaled logits.

        """
        if logits.shape[1] == 1:
            def sigmoid(z:np.ndarray) -> np.ndarray:
                return 1/(1 + np.exp(-z))
            positive_class_prob = sigmoid(logits).reshape(-1, 1)
            predictions = np.concatenate([1 - positive_class_prob, positive_class_prob], axis=1)
        else:
            predictions = logits - np.max(logits, axis=1, keepdims=True)
            predictions = np.exp(predictions)
            predictions = predictions/np.sum(predictions,axis=1, keepdims=True)

        count = predictions.shape[0]
        y_true = predictions[np.arange(count), true_label]
        predictions[np.arange(count), true_label] = 0

        y_wrong = np.sum(predictions, axis=1)
        output_signals = np.log(y_true+1e-45) - np.log(y_wrong+1e-45)
        return output_signals  # noqa: RET504

    def prepare_attack(self:Self)->None:
        """Prepares data to obtain metric on the target model and dataset, using signals computed on the auxiliary model/dataset.

        It selects a balanced subset of data samples from in-group and out-group members
        of the audit dataset, prepares the data for evaluation, and computes the logits
        for both shadow models and the target model.
        """

        # Fixed variance is used when the number of shadow models is below 32 (64, IN and OUT models)
        #       from (Membership Inference Attacks From First Principles)
        self.fix_var_threshold = 32

        self.attack_data_indices = self.sample_indices_from_population(include_train_indices = True,
                                                                       include_test_indices = True)

        self.shadow_model_indices = ShadowModelHandler().create_shadow_models(num_models = self.num_shadow_models,
                                                                              shadow_population =  self.attack_data_indices,
                                                                              training_fraction = self.training_data_fraction,
                                                                              online = self.online)

        self.shadow_models, _ = ShadowModelHandler().get_shadow_models(self.shadow_model_indices)

        self.out_indices = ~ShadowModelHandler().get_in_indices_mask(self.shadow_model_indices, self.audit_dataset["data"]).T

        true_labels = self.handler.get_labels(self.audit_dataset["data"])
        self.target_logits = ShadowModelHandler().load_logits(name="target")
        self.shadow_models_logits = []
        for indx in self.shadow_model_indices:
            self.shadow_models_logits.append(ShadowModelHandler().load_logits(indx=indx))

        self.shadow_models_logits = np.array([self.rescale_logits(x, true_labels) for x in self.shadow_models_logits])
        self.target_logits = self.rescale_logits(self.target_logits, true_labels)

    def run_attack(self:Self) -> MIAResult:
        """Runs the attack on the target model and dataset and assess privacy risks or data leakage.

        This method evaluates how the target model's output (logits) for a specific dataset
        compares to the output of shadow models to determine if the dataset was part of the
        model's training data or not.

        Returns
        -------
        Result(s) of the metric. An object containing the metric results, including predictions,
        true labels, and signal values.

        """
        n_audit_samples = self.shadow_models_logits.shape[1]
        score = np.zeros(n_audit_samples)  # List to hold the computed probability scores for each sample

        #  Decides which score calculation method should be used
        if(self.vectorized):
            score = lira_vectorized.lira_vectorized(self.shadow_models_logits,
                                                self.target_logits,
                                                self.in_indices_masks,
                                                self.var_calculation,
                                                self.online)
        else:
            score = lira_iterative.lira_iterative(self.shadow_models_logits,
                                                self.target_logits,
                                                self.in_indices_masks,
                                                self.var_calculation,
                                                self.online)

        # Split the score array into two parts based on membership: in (training) and out (non-training)
        in_members = self.audit_dataset["in_members"]
        out_members = self.audit_dataset["out_members"]
        self.in_member_signals = score[in_members].reshape(-1,1)  # Scores for known training data members
        self.out_member_signals = score[out_members].reshape(-1,1)  # Scores for non-training data members

        # Prepare true labels array, marking 1 for training data and 0 for non-training data
        true_labels = np.concatenate(
            [np.ones(len(self.in_member_signals)), np.zeros(len(self.out_member_signals))]
        )

        # Combine all signal values for further analysis
        signal_values = np.concatenate([self.in_member_signals, self.out_member_signals])

        # Return a result object containing predictions, true labels, and the signal values for further evaluation
        return MIAResult.from_full_scores(true_membership=true_labels,
                                    signal_values=signal_values,
                                    result_name="LiRA",
                                    metadata=self.configs.model_dump())
