#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Implementation of the RMIA attack."""
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import nn

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.attacks.utils.utils import (
    gaussian_residual_probability,
    huber_energy,
    huber_residual_probability,
    laplace_residual_probability,
    softmax_logits,
)
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.signals.functional import mae, mse
from leakpro.signals.signal import ModelLogits
from leakpro.signals.signal_extractor import PytorchModel
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class AttackRMIA(AbstractMIA):
    """Implementation of the RMIA attack."""

    class AttackConfig(BaseModel):
        """Configuration for the RMIA attack."""

        model_config = ConfigDict(extra="forbid")
        num_shadow_models: int = Field(default=1,
                                       ge=1,
                                       description="Number of shadow models")
        temperature: float = Field(default=2.0,
                                   ge=0.0,
                                   description="Softmax temperature")
        training_data_fraction: float = Field(default=0.5,
                                              ge=0.0,
                                              le=1.0,
                                              description="Part of available attack data to use for shadow models")
        online: bool = Field(default=False,
                             description="Online vs offline attack")
        attack_data_fraction: float = Field(default=0.5,
                                            ge=0.0,
                                            le=1.0,
                                            description="Part of available attack data to use for estimating p(z). \
Going below 0.1 noticeably degrades the attack quality according to paper.")
        sigma: Optional[float] = Field(default=None,
                                       gt=0.0,
                                       description="Residual std for the likelihood used with continuous-output \
(forecasting/regression) models; mapped to the scale of the family implied by the training criterion \
(Gaussian for MSE, Laplace for L1, Huber for SmoothL1/Huber). None (default) estimates the scale from \
shadow-model residuals on the z-population. Ignored for classification models, where temperature applies instead.")
        # Parameters to be used with optuna
        gamma: float = Field(default=2.0,
                        ge=0.0,
                        description="Parameter to threshold LLRs",
                        json_schema_extra = {"optuna": {"type": "float", "low": 0.1, "high": 10, "log": True}})
        offline_a: float = Field(default=0.33,
                                 ge=0.0,
                                 le=1.0,
                                 description="Parameter to estimate the marginal p(x)",
                                 json_schema_extra = {"optuna": {"type": "float", "low": 0.0, "high": 1.0,"enabled_if": lambda model: not model.online}})  # noqa: E501

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
        """Initialize the RMIA attack.

        Args:
        ----
            handler (MIAHandler): The input handler object.
            configs (dict): Configuration parameters for the attack.

        """
        logger.info("Configuring the RMIA attack")
        # Initializes the pydantic object using the user-provided configs
        # This will ensure that the user-provided configs are valid
        self.configs = self.AttackConfig() if configs is None else self.AttackConfig(**configs)

        # Call the parent class constructor. It will check the configs.
        super().__init__(handler)

        # Assign the configuration parameters to the object
        for key, value in self.configs.model_dump().items():
            setattr(self, key, value)

        self.shadow_models = []
        self.epsilon = 1e-6
        self.shadow_models = None
        self.shadow_model_indices = None

        # Resolve the residual-likelihood family from the training criterion: each regression loss
        # implies a noise model (MSE -> Gaussian, L1 -> Laplace, SmoothL1/Huber -> Huber), and the
        # signal probability must use the matching likelihood. Classification losses keep the
        # softmax probability of the true class.
        criterion = self.handler.get_criterion()
        family_by_criterion = ((nn.MSELoss, "gaussian"), (nn.L1Loss, "laplace"),
                               (nn.SmoothL1Loss, "huber"), (nn.HuberLoss, "huber"))
        self.likelihood_family = next((family for cls, family in family_by_criterion if isinstance(criterion, cls)), None)
        self.is_regression = self.likelihood_family is not None
        # SmoothL1Loss(beta) is HuberLoss(delta=beta) scaled by 1/beta; the constant factor is
        # absorbed by the likelihood scale, so both map to the Huber family with the same threshold.
        if isinstance(criterion, nn.SmoothL1Loss):
            self._huber_delta = criterion.beta
        elif isinstance(criterion, nn.HuberLoss):
            self._huber_delta = criterion.delta
        else:
            self._huber_delta = None
        self._scale = None  # likelihood scale, resolved in prepare_attack when is_regression
        logger.info(f"RMIA signal probability mode: {self.likelihood_family or 'softmax'} "
                    f"(criterion: {type(criterion).__name__})")

        self.load_for_optuna = False
        self.attack_cache_folder_path = ShadowModelHandler().attack_cache_folder_path
        self.bayesian_optimization = False

    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "RMIA attack"
        reference_str = "Zarifzadeh, Sajjad, Philippe Cheng-Jie Marc Liu, and Reza Shokri. \
            Low-Cost High-Power Membership Inference by Boosting Relativity. (2023)."
        summary_str = "The RMIA attack is a membership inference attack based on the output logits of a black-box model."
        detailed_str = "The attack is executed according to: \
            1. A fraction of the population is sampled to compute the likelihood LR_z of p(z|theta) to p(z) for the target model.\
            2. The ratio is used to compute the likelihood ratio LR_x of p(x|theta) to p(x) for the target model. \
            3. The ratio LL_x/LL_z is viewed as a random variable (z is random) and used to classify in-members and out-members. \
            4. The attack is evaluated on an audit dataset to determine the attack performance."
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }

    def _get_z_logits(self:Self,
                      logits_cached: np.ndarray,
                      model,  # noqa: ANN001
                      z_indices: np.ndarray,
                      cached_mask: np.ndarray,
                      logit_row: dict) -> np.ndarray:
        """Return logits for z_indices, reading from cache where available and computing on-the-fly otherwise."""
        n = len(z_indices)
        logits = np.zeros((n, *logits_cached.shape[1:]))

        if cached_mask.any():
            rows = np.array([logit_row[int(idx)] for idx in z_indices[cached_mask]])
            logits[cached_mask] = logits_cached[rows]

        if (~cached_mask).any():
            # Shadow models are already PytorchModel wrappers; target model is a raw nn.Module.
            model_wrapped = model if isinstance(model, PytorchModel) else PytorchModel(model, self.handler.get_criterion())
            unc = np.array(ModelLogits()(
                [model_wrapped],
                self.handler,
                z_indices[~cached_mask],
            )).squeeze()
            logits[~cached_mask] = np.atleast_2d(unc)

        return logits

    def _resolve_scale(self:Self, shadow_logits_z: list, z_labels: np.ndarray) -> None:
        """Resolve the residual-likelihood scale used by the regression signal probability.

        With a user-provided sigma (residual std), it is mapped to the scale of the resolved
        family: sigma^2 for Gaussian, sigma/sqrt(2) for Laplace (Var = 2b^2), and sigma^2 for
        Huber (variance of the quadratic core). Otherwise the scale is estimated once from the
        shadow-model residuals on the z-population: the mean squared residual (Gaussian MLE),
        the mean absolute residual (Laplace MLE), or the mean Huber energy (heuristic — the
        Huber scale has no closed-form MLE). The same value is reused for the audit points in
        _run_attack so the scale is consistent across all probability ratios.

        Args:
        ----
            shadow_logits_z (list): Per-shadow-model predictions on the z-points.
            z_labels (np.ndarray): True outputs for the z-points, same shape as each prediction.

        """
        if self.sigma is not None:
            sigma_to_scale = {"gaussian": self.sigma ** 2,
                              "laplace": self.sigma / np.sqrt(2.0),
                              "huber": self.sigma ** 2}
            self._scale = sigma_to_scale[self.likelihood_family]
            return

        if self.likelihood_family == "gaussian":
            energies = [np.mean(mse(logits_z, z_labels)) for logits_z in shadow_logits_z]
        elif self.likelihood_family == "laplace":
            energies = [np.mean(mae(logits_z, z_labels)) for logits_z in shadow_logits_z]
        else:
            energies = [np.mean(huber_energy(logits_z, z_labels, self._huber_delta)) for logits_z in shadow_logits_z]

        self._scale = float(np.mean(energies))
        if self._scale <= 0.0:
            # Degenerate case: shadow models reproduce all z-targets exactly.
            self._scale = self.epsilon
        logger.info(f"Estimated {self.likelihood_family} residual scale = {self._scale:.6g} "
                    f"from {len(shadow_logits_z)} shadow model(s) on {len(z_labels)} z-points")

    def _signal_probability(self:Self, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute the per-point probability of the true output given a model.

        Classification: temperature softmax indexed at the true class (RMIA paper, Sec. 3).
        Regression/forecasting: residual likelihood matched to the training criterion's noise
        model — Gaussian exp(-MSE/(2*scale)) for MSE, Laplace exp(-MAE/scale) for L1, Huber
        for SmoothL1/Huber. The normalizing constants cancel in RMIA's ratios.

        Args:
        ----
            logits ( len(dataset) x ... ): Model outputs (class logits or forecasts).
            labels ( len(dataset) x ... ): True classes (int) or true outputs (float).

        Returns:
        -------
            Per-point probabilities of shape ( len(dataset), ).

        """
        if self.is_regression:
            if self._scale is None:
                raise RuntimeError("Residual likelihood scale is unresolved — prepare_attack must run first.")
            if self.likelihood_family == "gaussian":
                return gaussian_residual_probability(logits, labels, self._scale)
            if self.likelihood_family == "laplace":
                return laplace_residual_probability(logits, labels, self._scale)
            return huber_residual_probability(logits, labels, self._scale, self._huber_delta)
        return softmax_logits(logits, self.temperature)[np.arange(len(labels)), labels]

    def _prepare_shadow_models(self:Self) -> None:

        # Shadow models are trained on all data points in pairs.
        self.attack_data_indices = self.sample_indices_from_population(include_train_indices = True,
                                                                    include_test_indices = True)
        # train shadow models
        logger.info(f"Check for {self.num_shadow_models} shadow models (dataset: {len(self.attack_data_indices)} points)")
        self.shadow_model_indices = ShadowModelHandler().create_shadow_models(
            num_models = self.num_shadow_models,
            shadow_population = self.attack_data_indices,
            training_fraction = self.training_data_fraction,
            online = True)
        # load shadow models
        self.shadow_models, _ = ShadowModelHandler().get_shadow_models(self.shadow_model_indices)

        if self.online is False:
            self.out_indices = ~ShadowModelHandler().get_in_indices_mask(self.shadow_model_indices, self.audit_dataset["data"]).T

    def prepare_attack(self:Self) -> None:
        """Prepare Data needed for running the attack on the target model and dataset.

        Signals are computed on the auxiliary model(s) and dataset.
        """
        logger.info("Preparing shadow models for RMIA attack")

        # If we already have one run, we dont need to check for shadow models as logits are stored
        if not self.load_for_optuna:
            self._prepare_shadow_models()

            self.ground_truth = self.handler.get_labels(self.audit_dataset["data"])
            self.logits_theta = ShadowModelHandler().load_logits(name=f"target_{ShadowModelHandler().target_model_hash}")
            self.logits_shadow_models = []
            for indx in self.shadow_model_indices:
                self.logits_shadow_models.append(ShadowModelHandler().load_logits(indx=indx))

        # Sample z from full population per Algorithm 1 of the RMIA paper.
        # Build a cache lookup (global index → row) once — identical for target and all shadow models.
        all_cached = np.concatenate([
            np.asarray(self.handler.train_indices),
            np.asarray(self.handler.test_indices),
        ])
        logit_row = {int(idx): i for i, idx in enumerate(all_cached)}

        n_z = int(self.attack_data_fraction * len(self.attack_data_indices))
        z_indices = np.random.choice(self.attack_data_indices, size=n_z, replace=False)
        z_labels = self.handler.get_labels(z_indices)

        if self.is_regression != np.issubdtype(z_labels.dtype, np.floating):
            logger.warning(f"Label dtype ({z_labels.dtype}) does not match the task resolved from the "
                           f"criterion ({'regression' if self.is_regression else 'classification'}). "
                           "Check that the target metadata criterion matches the model output.")

        # cached_mask is True for z-points present in the logit cache (train+test).
        # Auxiliary population points not in the cache are computed on-the-fly.
        # For configs where f_train + f_test = 1.0 this mask is all-True and no forward pass is needed.
        cached_mask = np.array([int(idx) in logit_row for idx in z_indices])

        # Collect raw model outputs on the z-points for the target and all shadow models
        logits_z_theta = self._get_z_logits(
            self.logits_theta, self.handler.target_model, z_indices, cached_mask, logit_row)
        logits_z_shadow_models = [
            self._get_z_logits(sm_logits, sm, z_indices, cached_mask, logit_row)
            for sm, sm_logits in zip(self.shadow_models, self.logits_shadow_models)
        ]

        if self.is_regression:
            self._resolve_scale(logits_z_shadow_models, z_labels)

        # p(z | target model)
        p_z_given_theta = np.atleast_2d(self._signal_probability(logits_z_theta, z_labels))

        # p(z | each shadow model)
        p_z_given_shadow_models = np.array(
            [self._signal_probability(logits_z, z_labels) for logits_z in logits_z_shadow_models])

        # evaluate the marginal p(z)
        if self.online is True:
            p_z = np.mean(p_z_given_shadow_models, axis=0, keepdims=True)
        else:
            # create a mask that checks, for each point, if it was in the training set
            p_z = np.mean(p_z_given_shadow_models, axis=0)
            p_z = 0.5*((self.offline_a + 1) * p_z + (1-self.offline_a))

        self.ratio_z = p_z_given_theta / (p_z + self.epsilon)

    def _run_attack(self:Self) -> None:
        logger.info("Running RMIA online attack")

        # probability of the true output for each audit point given the target model
        n_audit_points = len(self.ground_truth)
        p_x_given_theta = np.atleast_2d(self._signal_probability(self.logits_theta, self.ground_truth))

        # same per shadow model, to compute the marginal p(x)
        p_x_given_shadow_models = np.array(
            [self._signal_probability(x, self.ground_truth) for x in self.logits_shadow_models])

        if self.online is True:
            p_x = np.mean(p_x_given_shadow_models, axis=0, keepdims=True)
        else:
            # compute the marginal p(x) from P_out and p_in where p_in = a*p_out+b
            masked_values = np.where(self.out_indices, p_x_given_shadow_models, np.nan)
            p_x_out = np.nanmean(masked_values, axis=0)
            p_x = 0.5*((self.offline_a + 1) * p_x_out + (1-self.offline_a))

        # compute the ratio of p(x|theta) to p(x)
        ratio_x = p_x_given_theta / (p_x + self.epsilon)

        # for each x, compute the score
        score = np.zeros((1, n_audit_points))
        for i in range(n_audit_points):
            likelihoods = ratio_x[0,i] / self.ratio_z
            score[0, i] = np.mean(likelihoods > self.gamma)

        # pick out the in-members and out-members signals
        in_members = self.audit_dataset["in_members"]
        out_members = self.audit_dataset["out_members"]
        self.in_member_signals = score[0,in_members].reshape(-1,1)
        self.out_member_signals = score[0,out_members].reshape(-1,1)

    def run_attack(self:Self) -> MIAResult:
        """Run the attack on the target model and dataset.

        Returns
        -------
            Result(s) of the metric.

        """
        # perform the attack
        self._run_attack()

        # set true labels for being in the training dataset
        true_labels = np.concatenate([np.ones(len(self.in_member_signals)), np.zeros(len(self.out_member_signals)),])
        signal_values = np.concatenate([self.in_member_signals, self.out_member_signals])

        # Ensure we use the stored quantities from now
        self.load_for_optuna = True

        # Save the results
        return MIAResult.from_full_scores(true_membership=true_labels,
                                          signal_values=signal_values,
                                          result_name="RMIA",
                                          metadata=self.configs.model_dump())

    def reset_attack(self: Self, config:BaseModel) -> None:
        """Reset attack to initial state."""

        # Assign the new configuration parameters to the object
        self.configs = config
        for key, value in config.model_dump().items():
            setattr(self, key, value)

        # new hyperparameters have been set, let's prepare the attack again
        self.prepare_attack()


