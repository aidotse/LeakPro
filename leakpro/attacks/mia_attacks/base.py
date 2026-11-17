#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Implementation of the BASE attack."""

from collections.abc import Iterator
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field
from scipy.special import logsumexp
from tqdm import tqdm

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.robustness import RobustnessConfig, save_robustness_results
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.attacks.utils.utils import softmax_logits
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.signals.signal import ModelLogits
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger
from leakpro.utils.seed import seed_everything


class BASEConfig(BaseModel):
    """Configuration for the BASE attack."""

    num_shadow_models: int = Field(default=1, ge=1, description="Number of shadow models")
    temperature: float = Field(default=2.0, ge=0.0, description="Softmax temperature")
    training_data_fraction: float = Field(default=0.5, gt=0.0, lt=1.0, description="Part of available attack data to use for shadow models. Must be < 1: at 1 every shadow model trains on every point, leaving no OUT reference models.")  # noqa: E501
    online: bool = Field(default=False, description="Online vs offline attack")
    offline_scale_factor: float = Field(default=0.33,
                                        description="Rescale the LogSumExp threshold to compensate for the lack of in-models",
                                        json_schema_extra = {"optuna": {"type": "float", "low": 0.0, "high": 1.0,"enabled_if": lambda model: not model.online}})  # noqa: E501

class AttackBASE(AbstractMIA):
    """Implementation of the BASE attack."""

    AttackConfig = BASEConfig # required config for attack

    def __init__(self:Self,
                 handler: AbstractInputHandler,
                 configs: dict
                 ) -> None:
        """Initialize the BASE attack.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.
            configs (dict): Configuration parameters for the attack.

        """
        logger.info("Configuring the BASE attack")
        # Initializes the pydantic object using the user-provided configs
        # This will ensure that the user-provided configs are valid
        self.configs = BASEConfig() if configs is None else BASEConfig(**configs)

        # Call the parent class constructor. It will check the configs.
        super().__init__(handler)

        # Assign the configuration parameters to the object
        for key, value in self.configs.model_dump().items():
            setattr(self, key, value)

        self.shadow_models = []
        self.epsilon = 1e-6
        self.shadow_models = None
        self.shadow_model_indices = None
        self.attack_cache_folder_path = ShadowModelHandler().attack_cache_folder_path


    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "BASE attack"
        reference_str = "Lassila, Marcus, Johan Östman, and Khac-Hoang Ngo. Practical Bayes-Optimal Membership Inference Attacks. arXiv preprint arXiv:2505.24089 (2025)."  # noqa: E501
        summary_str = "A loss based attack with a log-sum-exponential approximation of the threshold"
        detailed_str = "The attack is executed according to: \
            1. Compute the loss value of the target model on the target node, L(target_model, target_point).\
            2. Compute the LogSumExp over the negative loss value over N shadow models on the target point. \
            3. The score is -L(target_model, target_point) - LogSumExp(-L(shadow_model, target_point))."
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }

    def _log_conf(self:Self, logits: np.ndarray, ground_truth_indices: np.ndarray) -> np.ndarray:
        """Log-confidence of the correct class, for either multi-class or single-logit (binary) logits.

        Raw logits can't be fed straight into scipy's log_softmax when the model has a single
        output unit (binary classification): log_softmax over one column is always 0, and indexing
        with a ground-truth label of 1 raises an IndexError. softmax_logits() already knows how to
        expand a single logit into [1-p, p] via sigmoid before applying temperature scaling, so we
        reuse it here (same helper RMIA uses) and take the log ourselves.
        """
        n_points = logits.shape[0]
        probs = softmax_logits(logits, self.temperature)
        return np.log(probs + self.epsilon)[np.arange(n_points), ground_truth_indices]

    def prepare_attack(self:Self, seed: Optional[int] = None) -> None:
        """Prepare data needed for running the attack on the target model and dataset.

        Signals are computed on the auxiliary model(s) and dataset.

        Args:
            seed: Optional seed for model preparation.

        """
        if seed is not None:
            seed_everything(seed)
        logger.info("Preparing shadow models for BASE attack")

        # Get all available indices for attack dataset, if self.online = True, include training and test data
        self.attack_data_indices = self.sample_indices_from_population(include_train_indices = True,
                                                                       include_test_indices = True)

        # train shadow models
        logger.info(f"Check for {self.num_shadow_models} shadow models (dataset: {len(self.attack_data_indices)} points)")
        self.shadow_model_indices = ShadowModelHandler().create_shadow_models(
            num_models = self.num_shadow_models,
            shadow_population = self.attack_data_indices,
            training_fraction = self.training_data_fraction,
        )
        # load shadow models
        self.shadow_models, _ = ShadowModelHandler().get_shadow_models(self.shadow_model_indices)
        if self.online is False:
            self.out_indices = ~ShadowModelHandler().get_in_indices_mask(self.shadow_model_indices, self.audit_dataset["data"]).T

    def _check_out_models_available(self:Self, n_out_models:np.ndarray) -> None:
        """Verify every scored point has at least one OUT shadow model.

        The offline threshold is ``logsumexp(out_logits) - log(n_out_models)``, the log mean
        confidence over the models that did not train on the point. That is well defined for any
        positive number of OUT models, so the only real requirement is that the count is non-zero.

        This replaces an assert that demanded exactly ``num_shadow_models // 2`` OUT models per
        point. That encoded the old fixed half-and-half split and was wrong twice over: it rejected
        every ``training_data_fraction`` other than 0.5, and it already failed for an odd
        ``num_shadow_models`` at fraction 0.5, where the counts legitimately straddle the half.

        Args:
        ----
            n_out_models (np.ndarray): Per-point count of shadow models that did not train on the point.

        Raises:
        ------
            ValueError: If any scored point has no OUT shadow model.

        """
        n_without = int(np.sum(n_out_models == 0))
        if n_without > 0:
            raise ValueError(
                f"Offline BASE needs at least one OUT shadow model per audit point, but {n_without} of "
                f"{n_out_models.size} points are in every shadow model. With num_shadow_models="
                f"{self.num_shadow_models} and training_data_fraction={self.training_data_fraction}, "
                f"each point is expected in about {self.num_shadow_models * self.training_data_fraction:.1f} "
                "model(s). Move the fraction towards 0.5, or raise num_shadow_models."
            )

    def score_samples(self:Self, dataloader:list, audit_indices:list) -> np.ndarray:
        """Score samples in the dataloader.

        Args:
        ----
            dataloader (list): A dataloader to score samples from.
            audit_indices (list): The indices of the audit dataset.

        Returns:
        -------
            np.ndarray: The scores for the samples in the dataloader.

        """
        n_points = len(dataloader.dataset)
        ground_truth_indices = dataloader.dataset.targets.numpy()
        assert n_points == len(ground_truth_indices), "Number of points and labels must be the same"
        logger.info(f"Scoring {n_points} points with BASE attack")
        logits_target = np.array(ModelLogits()([self.target_model], None, None, dataloader)).squeeze(axis=0)
        log_conf_target = self._log_conf(logits_target, ground_truth_indices)
        # run points through shadow models and collect the log confidence values
        logits_sm = []
        for m in tqdm(self.shadow_models, desc="Scoring with shadow models"):
            logits_sm.append( np.array(ModelLogits()([m], self.handler, None, dataloader)).squeeze(axis=0) )
        log_conf_shadow_models = np.array([self._log_conf(x, ground_truth_indices) for x in logits_sm])

        out_indices = None
        if not self.online:
            index_map = np.array([np.where(self.audit_dataset["data"] == val)[0][0] for val in audit_indices])
            out_indices = self.out_indices[:, index_map]
        return self._score_signals(log_conf_target, log_conf_shadow_models, out_indices)

    def run_attack(self:Self) -> MIAResult:
        """Run the attack on the target model and dataset.

        Returns
        -------
            Result(s) of the metric.

        """
        # perform the attack
        online_or_offline = "online" if self.online else "offline"
        attack_info = f"Running {online_or_offline} BASE attack with {self.num_shadow_models} shadow models"
        logger.info(attack_info)

        signals = self._get_cached_signals()
        score = self._score_signals(signals[0], signals[1:], self.out_indices if not self.online else None)

        # pick out the in-members and out-members signals
        in_members = self.audit_dataset["in_members"]
        out_members = self.audit_dataset["out_members"]
        self.in_member_signals = score[in_members].reshape(-1,1)
        self.out_member_signals = score[out_members].reshape(-1,1)

        # set true labels for being in the training dataset
        true_labels = np.concatenate([np.ones(len(self.in_member_signals)),np.zeros(len(self.out_member_signals)),])
        signal_values = np.concatenate([self.in_member_signals, self.out_member_signals])

        # compute ROC, TP, TN etc
        return MIAResult.from_full_scores(true_membership=true_labels,
                                    signal_values=signal_values,
                                    result_name="BASE",
                                    metadata=self.configs.model_dump(),
                                    audit_indices=self.scored_population_indices())

    def _get_cached_signals(self) -> np.ndarray:
        """Read target and shadow log-confidence values in the cached audit-record order."""
        # Load the logits for the target model and shadow models
        ground_truth_indices = self.handler.get_labels(self.audit_dataset["data"])
        logits_target = ShadowModelHandler().load_logits(name=f"target_{ShadowModelHandler().target_model_hash}")
        logits_shadow_models = []
        for indx in self.shadow_model_indices:
            logits_shadow_models.append(ShadowModelHandler().load_logits(indx=indx))

        # collect the log confidence output of the correct class (which is the negative cross-entropy loss)
        log_conf_target = self._log_conf(logits_target, ground_truth_indices)

        # run points through shadow models and collect the log confidence values
        log_conf_shadow_models = np.array([self._log_conf(x, ground_truth_indices) for x in logits_shadow_models])

        return np.concatenate((log_conf_target[None], log_conf_shadow_models))

    def _score_signals(self, log_conf_target: np.ndarray, log_conf_shadow_models: np.ndarray,
                       out_indices: np.ndarray) -> np.ndarray:
        """Evaluate BASE on the supplied target and reference log confidences."""
        if self.online is True:
            threshold = logsumexp(log_conf_shadow_models, axis=0) - np.log(self.num_shadow_models)
        else:
            n_out_models = np.sum(out_indices, axis=0)
            self._check_out_models_available(n_out_models)

            out_logits = np.where(out_indices, log_conf_shadow_models, -np.inf)
            threshold = logsumexp(out_logits, axis=0) - np.log(n_out_models)

        return log_conf_target - threshold if self.online else log_conf_target - self.offline_scale_factor * threshold

    def run_robustness(self, n_runs: int, seed: Optional[int] = None,
                       output_dir: Optional[str] = None) -> list[MIAResult]:
        """Rotate cached log confidences and OUT masks through BASE's ordinary scorer.

        The original target runs first, followed by distinct shadow targets.
        Every target is excluded from its references and uses its recorded membership.
        """
        config = RobustnessConfig(n_runs=n_runs, seed=seed)
        config.validate_attack(self)
        self.prepare_attack(seed=seed)
        metadata = ShadowModelHandler().get_shadow_model_metadata(self.shadow_model_indices)
        training_indices = [self.handler.train_indices, *[model.train_indices for model in metadata]]
        membership = np.array([np.isin(self.audit_dataset["data"], indices) for indices in training_indices])
        training_sizes = [len(indices) for indices in training_indices]
        signals = self._get_cached_signals()
        targets = [0, *np.random.default_rng(seed).permutation(np.arange(1, len(signals)))[:config.n_runs - 1]]

        def rotations() -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
            for target in targets:
                references = np.delete(np.arange(len(signals)), target)
                scores = self._score_signals(signals[target], signals[references], ~membership[references])
                yield target, references, scores, np.ones(signals.shape[1], dtype=bool)

        return save_robustness_results(self, {"audit": signals}, membership, training_sizes, targets,
                                       rotations(), seed, output_dir)
