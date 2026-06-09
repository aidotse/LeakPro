#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Implementation of a multi-signal version of the LiRA attack."""

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator
from scipy.stats import norm
from tqdm import tqdm

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.signals import functional
from leakpro.signals.functional import SIGNAL_MEMBERSHIP_DIRECTION
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger

# Maps the class-style signal names accepted in the config to the functional signal names
# used as keys in SIGNAL_MEMBERSHIP_DIRECTION. (HopSkipJumpDistance has no functional twin and
# is not a valid MS-LiRA signal.)
_CLASS_TO_FUNCTIONAL = {
    "ModelLogits": "logits",
    "ModelRescaledLogits": "rescaled_logits",
    "ModelLoss": "loss",
    "Seasonality": "seasonality",
    "Trend": "trend",
    "MSE": "mse",
    "MAE": "mae",
    "SMAPE": "smape",
    "RescaledSMAPE": "rescaled_smape",
    "TS2Vec": "ts2vec",
    "DTW": "dtw",
    "MSM": "msm",
}


def _signal_membership_direction(signal_name: str) -> int:
    """Return +1 if a higher value of this signal indicates membership, -1 if lower does.

    Accepts both class-style names (config) and functional names. Raises ValueError for any
    signal whose direction is not declared, so a new signal must opt in explicitly.
    """
    functional_name = _CLASS_TO_FUNCTIONAL.get(signal_name, signal_name)
    try:
        return SIGNAL_MEMBERSHIP_DIRECTION[functional_name]
    except KeyError as e:
        raise ValueError(
            f"No membership direction defined for signal '{signal_name}'. Add an entry to "
            "SIGNAL_MEMBERSHIP_DIRECTION in leakpro/signals/functional.py (+1 if a higher value "
            "indicates membership, -1 if a lower value does)."
        ) from e


def _resolve_signal_fn(signal_name: str) -> callable:
    """Resolve a config signal name to its function in ``leakpro.signals.functional``.

    Accepts both class-style names (e.g. ``ModelRescaledLogits``) and functional names
    (e.g. ``rescaled_logits``) for config back-compat. The functional signal runs on cached
    logits, so it never re-queries a model (unlike the class-based ``Signal`` objects).
    Raises ValueError for an unknown signal.
    """
    functional_name = _CLASS_TO_FUNCTIONAL.get(signal_name, signal_name)
    if not hasattr(functional, functional_name):
        available = [f for f in dir(functional) if not f.startswith("_") and callable(getattr(functional, f))]
        raise ValueError(f"Unknown signal '{signal_name}'. Available functional signals: {available}")
    return getattr(functional, functional_name)


class AttackMSLiRA(AbstractMIA):
    """Implementation of a multi-signal version of the LiRA attack."""

    class AttackConfig(BaseModel):
        """Configuration for the MSLiRA attack."""

        signals: list[str] = Field(default=["ModelRescaledLogits"], description="What signals to use.")
        num_shadow_models: int = Field(default=2, ge=1, description="Number of shadow models")
        training_data_fraction: float = Field(default=0.5, ge=0.0, le=1.0, description="Part of available attack data to use for shadow models")  # noqa: E501
        online: bool = Field(default=False, description="Online vs offline attack")
        var_calculation: Literal["carlini", "individual_carlini", "fixed"] = Field(default="carlini", description="Variance estimation method to use [carlini, individual_carlini, fixed]")  # noqa: E501
        std_eps: float = Field(default=1e-30, ge=0.0, le=0.001, description="Small value to add to the standard deviations when estimating Gaussians (for numerical stability).")  # noqa: E501

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
        """Initialize the MSLiRA attack.

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

        # Membership direction per signal (+1 higher=member, -1 lower=member). Resolved first so
        # an unknown/undeclared signal fails with the direction error before any models are trained.
        self.signal_directions = np.array(
            [_signal_membership_direction(name) for name in self.signals]
        )

        # Functional signals run on cached logits (no per-model re-query); see prepare_attack.
        self.signal_fns = [_resolve_signal_fn(name) for name in self.signals]

    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "Multi-Signal Likelihood Ratio Attack"

        reference_str = "Johansson N., Olsson T., Nilsson D., Östman J., & Hoseini F. \
        Privacy Risks in Time Series Forecasting: User- and Record-Level Membership Inference. \
        arXiv:2509.04169, 2025."

        summary_str = "LiRA is a membership inference attack based on rescaled logits of a black-box model. \
        The multi-signal version extends LiRA to attack a model based on multiple signals extracted from the outputs."

        detailed_str = "The attack is executed according to: \
            1. A fraction of the target model dataset is sampled to be included (in-) or excluded (out-) \
            from the shadow model training dataset. \
            2. The attack signals are used to estimate Gaussian distributions for in and out members, \
            independently for each signal. \
            3. Probabilities are multiplied across the different signal distributions to obtain a joint membership probability. \
            4. The thresholds are used to classify in-members and out-members. \
            5. The attack is evaluated on an audit dataset to determine the attack performance."

        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }

    def prepare_attack(self:Self) -> None:
        """Prepares data to obtain metric on the target model and dataset, using signals computed on the auxiliary model/dataset.

        It selects a balanced subset of data samples from in-group and out-group members
        of the audit dataset, prepares the data for evaluation, and computes the signals
        for both shadow models and the target model.
        """

        # Fixed variance is used when the number of shadow models is below 32 (64, IN and OUT models)
        #       from (Membership Inference Attacks From First Principles)
        self.fix_var_threshold = 32

        self.attack_data_indices = self.sample_indices_from_population(include_train_indices = self.online,
                                                                       include_test_indices = self.online)

        self.shadow_model_indices = ShadowModelHandler().create_shadow_models(num_models = self.num_shadow_models,
                                                                              shadow_population =  self.attack_data_indices,
                                                                              training_fraction = self.training_data_fraction,
                                                                              online = self.online)

        self.shadow_models, _ = ShadowModelHandler().get_shadow_models(self.shadow_model_indices)

        logger.info("Create masks for all IN and OUT samples")
        self.in_indices_masks = ShadowModelHandler().get_in_indices_mask(self.shadow_model_indices, self.audit_dataset["data"])

        # Compute every signal once on CACHED logits. cache_logits/load_logits store and realign
        # logits to concat(train_indices, test_indices) == audit_dataset["data"], so the cached
        # rows are positionally aligned with the audit set and with get_labels below. This replaces
        # the old per-signal, per-model Signal() calls that re-ran every model on every signal.
        audit_indices = self.audit_dataset["data"]
        # get_labels returns the dataset targets: int64 class labels for classification, the
        # continuous target series for forecasting. One call serves every signal; each functional
        # signal interprets it (rescaled_logits/loss expect int64 labels; mse/mae/... expect a
        # series matching the (already squeezed) cached logits).
        targets = np.asarray(self.handler.get_labels(audit_indices)).squeeze()
        target_logits = ShadowModelHandler().load_logits(name="target")
        shadow_logits = [ShadowModelHandler().load_logits(indx=indx) for indx in self.shadow_model_indices]

        shadow_models_signals = []
        target_model_signals = []
        for signal_fn, signal_name in zip(self.signal_fns, self.signals):
            logger.info(f"Calculating {signal_name} for the target model")
            target_model_signals.append(signal_fn(target_logits, targets))

            logger.info(f"Calculating {signal_name} for all {self.num_shadow_models} shadow models")
            # (n_shadow_models, n_audit_points) -> (n_audit_points, n_shadow_models)
            per_model = np.array([signal_fn(logits, targets) for logits in shadow_logits])
            shadow_models_signals.append(per_model.T)

        # Stack signals to (n_audit_points, n_shadow_models, n_signals) / (n_audit_points, n_signals).
        # Computed over the full audit set; the online branch filters rows below.
        self.shadow_models_signals = np.stack(shadow_models_signals, axis=-1)
        self.target_model_signals = np.stack(target_model_signals, axis=-1)

        if self.online:
            # Exclude all audit points that have either no IN or OUT samples
            num_shadow_models_seen_points = np.sum(self.in_indices_masks, axis=1)
            mask = (num_shadow_models_seen_points > 0) & (num_shadow_models_seen_points < self.num_shadow_models)

            # Filter the audit data
            self.audit_data_indices = audit_indices[mask]
            self.in_indices_masks = self.in_indices_masks[mask, :]
            self.shadow_models_signals = self.shadow_models_signals[mask]
            self.target_model_signals = self.target_model_signals[mask]

            # Filter IN and OUT members
            self.in_members = np.arange(np.sum(mask[self.audit_dataset["in_members"]]))
            num_out_members = np.sum(mask[self.audit_dataset["out_members"]])
            self.out_members = np.arange(len(self.in_members), len(self.in_members) + num_out_members)

            assert len(self.audit_data_indices) == len(self.in_members) + len(self.out_members)

            if len(self.audit_data_indices) == 0:
                raise ValueError("No points in the audit dataset are used for the shadow models")

        else:
            self.audit_data_indices = audit_indices
            self.in_members = self.audit_dataset["in_members"]
            self.out_members = self.audit_dataset["out_members"]

            # Check offline attack for possible IN- sample(s)
            count_in_samples = np.count_nonzero(self.in_indices_masks)
            if count_in_samples > 0:
                logger.info(f"Some shadow model(s) contains {count_in_samples} IN samples in total for the model(s)")
                logger.info("This is not an offline attack!")

        # Orient every signal so that a HIGHER value means "more member-like" (multiply
        # "lower=member" signals by -1). The offline score is a one-sided Gaussian tail
        # (norm.logcdf), which is monotone increasing, so signals must agree on direction
        # before they can be summed across the signal axis. The online likelihood ratio is
        # invariant to this sign flip, so applying it here is safe for both modes.
        self.shadow_models_signals = self.shadow_models_signals * self.signal_directions
        self.target_model_signals = self.target_model_signals * self.signal_directions

    def get_std(self:Self, signals: list, mask: list, is_in: bool, var_calculation: str) -> np.ndarray:
        """A function to define what method to use for calculating variance for LiRA."""

        # Fixed/Global variance calculation.
        if var_calculation == "fixed":
            return self._fixed_variance(signals, mask, is_in)

        # Variance calculation as in the paper ( Membership Inference Attacks From First Principles )
        elif var_calculation == "carlini":  # noqa: RET505
            return self._carlini_variance(signals, mask, is_in)

        # Variance calculation as in the paper ( Membership Inference Attacks From First Principles )
        #   but check IN and OUT samples individualy
        elif var_calculation == "individual_carlini":
            return self._individual_carlini(signals, mask, is_in)

        # Unknown variance calculation
        else:
            raise NotImplementedError("Unknown variance calculation specified.")

    def _fixed_variance(self:Self, signals: list, mask: list, is_in: bool) -> np.ndarray:
        if is_in and not self.online:
            return np.array([None])
        return np.std(signals[mask], axis=0)

    def _carlini_variance(self:Self, signals: list, mask: list, is_in: bool) -> np.ndarray:
        if self.num_shadow_models >= self.fix_var_threshold*2:
                return np.std(signals[mask], axis=0)
        if is_in:
            return self.fixed_in_stds
        return self.fixed_out_stds

    def _individual_carlini(self:Self, signals: list, mask: list, is_in: bool) -> np.ndarray:
        if np.count_nonzero(mask) >= self.fix_var_threshold:
            return np.std(signals[mask], axis=0)
        if is_in:
            return self.fixed_in_stds
        return self.fixed_out_stds

    @staticmethod
    def _offline_score(target_signals: np.ndarray, out_means: np.ndarray, out_stds: np.ndarray,
                       std_eps: float) -> float:
        """Offline membership score: the one-sided multivariate Gaussian tail under the OUT model.

        Implements arXiv:2509.04169 Eq. (2): ``Pr(Z <= s), Z ~ N(mu_out, Sigma_out)`` which, with a
        diagonal covariance, is the product of per-signal univariate tail probabilities — i.e. the
        sum of per-signal ``logcdf``. Signals must already be oriented so higher = member. This is
        the multivariate generalization of offline LiRA (Carlini Eq. 4); for a single signal it
        reduces exactly to ``norm.logcdf(t, mu_out, sigma_out)``.
        """
        return float(norm.logcdf(target_signals, out_means, out_stds + std_eps).sum())

    @staticmethod
    def _online_score(target_signals: np.ndarray, in_means: np.ndarray, in_stds: np.ndarray,
                      out_means: np.ndarray, out_stds: np.ndarray, std_eps: float) -> float:
        """Online membership score: summed per-signal log-likelihood ratio (Eq. 1, diagonal Sigma)."""
        in_prs = norm.logpdf(target_signals, in_means, in_stds + std_eps)
        out_prs = norm.logpdf(target_signals, out_means, out_stds + std_eps)
        return float((in_prs - out_prs).sum())

    def run_attack(self:Self) -> MIAResult:
        """Runs the attack on the target model and dataset and assess privacy risks or data leakage.

        This method evaluates how the signals extracted from the target model's output for a specific dataset
        compares to the signals of output of shadow models to determine if the dataset was part of the
        model's training data or not.

        Returns
        -------
        Result(s) of the metric. An object containing the metric results, including predictions,
        true labels, and membership scores.

        """
        n_audit_samples = self.shadow_models_signals.shape[0]
        score = np.zeros(n_audit_samples)  # List to hold the computed probability scores for each sample

        shadow_models_signals_flattened = self.shadow_models_signals.reshape(-1, self.shadow_models_signals.shape[-1])  # flatten only first two dimensions (samples, shadow models)  # noqa: E501
        self.fixed_in_stds = self.get_std(shadow_models_signals_flattened, self.in_indices_masks.flatten(), True, "fixed")
        self.fixed_out_stds = self.get_std(shadow_models_signals_flattened, (~self.in_indices_masks).flatten(), False, "fixed")

        # Iterate over and extract signals for IN and OUT shadow models for each audit sample
        for i, (shadow_models_signals, mask) in tqdm(enumerate(zip(self.shadow_models_signals, self.in_indices_masks)),
                            total=n_audit_samples,
                            desc="Processing audit samples"):

            target_signals = self.target_model_signals[i]

            # Compute OUT statistics
            out_means = np.mean(shadow_models_signals[~mask], axis=0)
            out_stds = self.get_std(shadow_models_signals, ~mask, False, self.var_calculation)

            if self.online:
                # Online: summed per-signal log-likelihood ratio (paper Eq. 1)
                in_means = np.mean(shadow_models_signals[mask], axis=0)
                in_stds = self.get_std(shadow_models_signals, mask, True, self.var_calculation)
                score[i] = self._online_score(target_signals, in_means, in_stds,
                                              out_means, out_stds, self.std_eps)
            else:
                # Offline: one-sided multivariate Gaussian tail under OUT (paper Eq. 2)
                score[i] = self._offline_score(target_signals, out_means, out_stds, self.std_eps)

            if np.isnan(score[i]):
                raise ValueError("Score is NaN")

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
                                    result_name="MS-LiRA",
                                    metadata=self.configs.model_dump())
