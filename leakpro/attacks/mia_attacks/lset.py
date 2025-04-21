"""Implementation of the LSET attack."""

import numpy as np
from pydantic import BaseModel, Field
from scipy.special import logsumexp

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.attacks.utils.utils import softmax_logits
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.signals.signal import ModelLogits
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class LSETConfig(BaseModel):
    """Configuration for the LSET attack."""

    num_shadow_models: int = Field(default=1, ge=1, description="Number of shadow models")
    temperature: float = Field(default=2.0, ge=0.0, description="Softmax temperature")
    training_data_fraction: float = Field(default=0.5, ge=0.0, le=1.0, description="Part of available attack data to use for shadow models")  # noqa: E501
    online: bool = Field(default=False, description="Online vs offline attack")


class AttackLSET(AbstractMIA):
    """Implementation of the LSET attack."""

    AttackConfig = LSETConfig # required config for attack

    def __init__(self:Self,
                 handler: AbstractInputHandler,
                 configs: dict
                 ) -> None:
        """Initialize the LSET attack.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.
            configs (dict): Configuration parameters for the attack.

        """
        logger.info("Configuring the LSET attack")
        # Initializes the pydantic object using the user-provided configs
        # This will ensure that the user-provided configs are valid
        self.configs = LSETConfig() if configs is None else LSETConfig(**configs)

        # Call the parent class constructor. It will check the configs.
        super().__init__(handler)

        # Assign the configuration parameters to the object
        for key, value in self.configs.model_dump().items():
            setattr(self, key, value)

        self.shadow_models = []
        self.signal = ModelLogits()
        self.epsilon = 1e-6
        self.shadow_models = None
        self.shadow_model_indices = None


    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "LSET attack"
        reference_str = "No reference"
        summary_str = "A simple loss based attack with a log-sum-exponential approximation of the threshold"
        detailed_str = "The attack is executed according to: \
            1. Compute the loss value of the target model on the target node, L(target_model, target_point).\
            2. Compute the LogSumExp over the negative loss value over N shadow models on the target point, LogSumExp(-L(shadow_model, target_point)). \
            3. The score is -L(target_model, target_point) - LogSumExp(-L(shadow_model, target_point))."
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
        logger.info("Preparing shadow models for LSET attack")

        # Get all available indices for attack dataset, if self.online = True, include training and test data
        self.attack_data_indices = self.sample_indices_from_population(include_train_indices = self.online,
                                                                       include_test_indices = self.online)

        # train shadow models
        logger.info(f"Check for {self.num_shadow_models} shadow models (dataset: {len(self.attack_data_indices)} points)")
        self.shadow_model_indices = ShadowModelHandler().create_shadow_models(
            num_models = self.num_shadow_models,
            shadow_population = self.attack_data_indices,
            training_fraction = self.training_data_fraction,
            online = self.online)
        # load shadow models
        self.shadow_models, _ = ShadowModelHandler().get_shadow_models(self.shadow_model_indices)

    def run_attack(self:Self) -> MIAResult:
        """Run the attack on the target model and dataset.

        Returns
        -------
            Result(s) of the metric.

        """
        # perform the attack
        if self.online:
            logger.info("Running LSET online attack")
            audit_data_indices, in_members, out_members = self._filter_audit_data_for_online_attack(self.shadow_model_indices)
        else:
            logger.info("Running LSET offline attack")
            audit_data_indices = self.audit_dataset["data"]
            in_members = self.audit_dataset["in_members"]
            out_members = self.audit_dataset["out_members"]

        n_audit_points = len(audit_data_indices)
        ground_truth_indices = self.handler.get_labels(audit_data_indices)
        assert np.issubdtype(ground_truth_indices.dtype, np.integer)

        # run target points through real model to get logits
        logits_target = np.array(self.signal([self.target_model], self.handler, audit_data_indices)).squeeze(axis=0)
        # collect the log confidence output of the correct class (which is the negative cross-entropy loss)
        log_conf_target = np.log(softmax_logits(logits_target, self.temperature)[np.arange(n_audit_points),ground_truth_indices])

        # run points through shadow models and collect the log confidence values
        logits_shadow_models = self.signal(self.shadow_models, self.handler, self.audit_dataset["data"])
        log_conf_shadow_models = np.array([np.log(softmax_logits(x, self.temperature)[np.arange(n_audit_points),ground_truth_indices]) for x in logits_shadow_models])

        # the threshold is relative to a Monte Carlo approximation to the logarithm of the expected exponential of the negative loss, over shadow models
        # subtracting the log(num_shadow_models) is not necessary if we sweep thresholds, but should be there if we want to get the probabilities using sigmoid
        threshold = logsumexp(log_conf_shadow_models, axis=0) - np.log(self.num_shadow_models)

        score = log_conf_target - threshold

        # pick out the in-members and out-members signals
        self.in_member_signals = score[in_members].reshape(-1,1)
        self.out_member_signals = score[out_members].reshape(-1,1)

        # set true labels for being in the training dataset
        true_labels = np.concatenate([np.ones(len(self.in_member_signals)),np.zeros(len(self.out_member_signals)),])
        signal_values = np.concatenate([self.in_member_signals, self.out_member_signals])

        # compute ROC, TP, TN etc
        return MIAResult.from_full_scores(true_membership=true_labels,
                                    signal_values=signal_values,
                                    result_name="LSET",
                                    metadata=self.configs.model_dump())
