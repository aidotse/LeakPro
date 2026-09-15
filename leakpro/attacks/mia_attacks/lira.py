#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Implementation of the LiRA attack."""

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import norm
from tqdm import tqdm

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.signals import functional
from leakpro.signals.utils.get_TS2Vec import bind_ts2vec_encoder
from leakpro.utils.import_helper import Self


class AttackLiRA(AbstractMIA):
    """Implementation of the LiRA attack."""

    class AttackConfig(BaseModel):
        """Configuration for the LiRA attack."""

        model_config = ConfigDict(extra="forbid")
        num_shadow_models: int = Field(default=1, ge=1, description="Number of shadow models")
        training_data_fraction: float = Field(default=0.5, ge=0.0, le=1.0, description="Part of available attack data to use for shadow models")  # noqa: E501
        online: bool = Field(default=False, description="Online vs offline attack")
        var_calculation: Literal["carlini", "individual_carlini", "fixed"] = Field(default="carlini", description="Variance estimation method to use [carlini, individual_carlini, fixed]")  # noqa: E501
        signal: str = Field(default="rescaled_logits",
                            description="What signal to use. Must be a function in leakpro.signals.functional.")

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

        # Accepts both functional names and the class-style names used by older configs.
        self.signal = functional.get(self.configs.signal)

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

    def prepare_attack(self:Self)->None:
        """Prepares data to obtain metric on the target model and dataset, using signals computed on the auxiliary model/dataset.

        It selects a balanced subset of data samples from in-group and out-group members
        of the audit dataset, prepares the data for evaluation, and computes the configured
        signal for both shadow models and the target model.
        """

        # Fixed variance is used when the number of shadow models is below 32 (64, IN and OUT models)
        #       from (Membership Inference Attacks From First Principles)
        self.fix_var_threshold = 32

        self.attack_data_indices = self.sample_indices_from_population(include_train_indices = True,
                                                                       include_test_indices = True)

        self.shadow_model_indices = ShadowModelHandler().create_shadow_models(num_models = self.num_shadow_models,
                                                                              shadow_population =  self.attack_data_indices,
                                                                              training_fraction = self.training_data_fraction,
                                                                              )

        # The shadow models themselves are never loaded: the attack scores their cached logits.
        self.out_indices = ~ShadowModelHandler().get_in_indices_mask(self.shadow_model_indices, self.audit_dataset["data"]).T

        # ts2vec carries a fitted artefact, and the target and shadow values below are only
        # comparable if they share it. Fit it once on the shadow population; other signals are
        # returned unchanged.
        self.signal = bind_ts2vec_encoder(self.signal, self.handler, self.attack_data_indices)

        # The signal is applied to the cached logits, so these hold signal values (rescaled logits,
        # an error, a distance, ...) rather than logits; named accordingly, as in MS-LiRA.
        true_labels = self.handler.get_labels(self.audit_dataset["data"])
        target_logits = ShadowModelHandler().load_logits(name="target")
        # Classification labels arrive with a spurious singleton axis (e.g. (N, 1)) that must be
        # dropped before rescaled_logits/loss index them as (N,). Forecasting targets, though,
        # already match the cached logits' shape (N, horizon, num_variables) — squeezing them
        # would strip the real num_variables axis and break the mse/dtw/msm/... shape assert.
        if true_labels.shape != target_logits.shape:
            true_labels = true_labels.squeeze()
        self.target_signals = self._check_signal_shape(self.signal(target_logits, true_labels),
                                                       n_audit_points=target_logits.shape[0])
        self.shadow_models_signals = []
        for indx in self.shadow_model_indices:
            shadow_logits = ShadowModelHandler().load_logits(indx=indx)
            self.shadow_models_signals.append(
                self._check_signal_shape(self.signal(shadow_logits, true_labels),
                                         n_audit_points=shadow_logits.shape[0])
            )
        self.shadow_models_signals = np.array(self.shadow_models_signals)

    def _check_signal_shape(self:Self, signal_values: np.ndarray, n_audit_points: int) -> np.ndarray:
        """Verify the signal produced exactly one value per audit point.

        LiRA models each point's signal as a scalar Gaussian, so a signal returning a vector per
        point (``logits``, which passes the raw per-class logits through) cannot be scored. Left
        unchecked, the extra axis silently becomes the audit-sample axis in run_attack and the
        attack reports scores for the wrong number of points.

        Args:
            signal_values (np.ndarray): Values returned by the configured signal function.
            n_audit_points (int): Number of audit points the logits were computed on.

        Returns:
            np.ndarray: signal_values unchanged.

        Raises:
            ValueError: If the signal did not return one scalar per audit point.

        """
        if signal_values.shape != (n_audit_points,):
            raise ValueError(
                f"Signal '{self.configs.signal}' returned shape {signal_values.shape}, but LiRA "
                f"requires one scalar per audit point, i.e. {(n_audit_points,)}. Use a scalar "
                "signal such as 'rescaled_logits' or 'loss'."
            )
        return signal_values

    def get_std(self:Self, signals: list, mask: list, is_in: bool, var_calculation: str) -> np.ndarray:
        """A function to define what method to use for calculating variance for LiRA."""

        # Fixed/Global variance calculation.
        if var_calculation == "fixed":
            return self._fixed_variance(signals, mask, is_in)

        # Variance calculation as in the paper ( Membership Inference Attacks From First Principles )
        if var_calculation == "carlini":
            return self._carlini_variance(signals, mask, is_in)

        # Variance calculation as in the paper ( Membership Inference Attacks From First Principles )
        #   but check IN and OUT samples individualy
        if var_calculation == "individual_carlini":
            return self._individual_carlini(signals, mask, is_in)

        return np.array([None])

    def _fixed_variance(self:Self, signals: list, mask: list, is_in: bool) -> np.ndarray:
        if is_in and not self.online:
            return np.array([None])
        return np.std(signals[mask])

    def _carlini_variance(self:Self, signals: list, mask: list, is_in: bool) -> np.ndarray:
        if self.num_shadow_models >= self.fix_var_threshold*2:
                return np.std(signals[mask])
        if is_in:
            return self.fixed_in_std
        return self.fixed_out_std

    def _individual_carlini(self:Self, signals: list, mask: list, is_in: bool) -> np.ndarray:
        if np.count_nonzero(mask) >= self.fix_var_threshold:
            return np.std(signals[mask])
        if is_in:
            return self.fixed_in_std
        return self.fixed_out_std

    def run_attack(self:Self) -> MIAResult:
        """Runs the attack on the target model and dataset and assess privacy risks or data leakage.

        This method evaluates how the signal computed on the target model's output for a specific
        dataset compares to the same signal on the shadow models, to determine if the dataset was
        part of the model's training data or not.

        Returns
        -------
        Result(s) of the metric. An object containing the metric results, including predictions,
        true labels, and signal values.

        """
        n_audit_samples = self.shadow_models_signals.shape[1]
        score = np.zeros(n_audit_samples)  # List to hold the computed probability scores for each sample

        self.fixed_in_std = self.get_std(self.shadow_models_signals.flatten(), (~self.out_indices).flatten(), True, "fixed")
        self.fixed_out_std = self.get_std(self.shadow_models_signals.flatten(), self.out_indices.flatten(), False, "fixed")

        # Iterate over and extract signals for IN and OUT shadow models for each audit sample
        for i in tqdm(range(n_audit_samples), total=n_audit_samples, desc="Processing audit samples"):

            # Calculate the mean for OUT shadow model signals
            out_mask = self.out_indices[:,i]
            sm_signals = self.shadow_models_signals[:,i]

            out_mean = np.mean(sm_signals[out_mask])
            out_std = self.get_std(sm_signals, out_mask, False, self.var_calculation)

            # Get the signal from the target model for the current sample
            target_signal = self.target_signals[i]

            # Calculate the log probability density function value
            if self.online:
                in_mean = np.mean(sm_signals[~out_mask])
                in_std = self.get_std(sm_signals, ~out_mask, True, self.var_calculation)

                pr_in = norm.logpdf(target_signal, in_mean, in_std + 1e-30)
                pr_out = norm.logpdf(target_signal, out_mean, out_std + 1e-30)
            else:
                # Offline attack scores by how unlikely the target signal is under the
                # OUT distribution, matching the reference implementation
                # (tensorflow/privacy mi_lira_2021: score = logpdf(out), negated at ROC time).
                pr_in = 0
                pr_out = norm.logpdf(target_signal, out_mean, out_std + 1e-30)

            score[i] = (pr_in - pr_out)  # Append the calculated probability density value to the score list
            if np.isnan(score[i]):
                raise ValueError("Score is NaN")

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
