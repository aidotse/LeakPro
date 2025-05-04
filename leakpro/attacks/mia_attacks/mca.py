"""Implementation of the MCA attack."""

import numpy as np
from pydantic import BaseModel, Field

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.attacks.utils.utils import softmax_logits
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.signals.signal import ModelLogits
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class MCAConfig(BaseModel):
    """Configuration for the MCA attack."""

    num_shadow_models: int = Field(default=1, ge=1, description="Number of shadow models")
    temperature: float = Field(default=2.0, ge=0.0, description="Softmax temperature")
    training_data_fraction: float = Field(default=0.5, ge=0.0, le=1.0, description="Part of available attack data to use for shadow models")  # noqa: E501
    online: bool = Field(default=False, description="Online vs offline attack")


class AttackMCA(AbstractMIA):
    """Implementation of the MCA attack."""

    AttackConfig = MCAConfig # required config for attack

    def __init__(self:Self,
                 handler: AbstractInputHandler,
                 configs: dict
                 ) -> None:
        """Initialize the MCA attack.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.
            configs (dict): Configuration parameters for the attack.

        """
        logger.info("Configuring the MCA attack")
        # Initializes the pydantic object using the user-provided configs
        # This will ensure that the user-provided configs are valid
        self.configs = MCAConfig() if configs is None else MCAConfig(**configs)

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
        title_str = "MCA attack"
        reference_str = "No reference"
        summary_str = "A heuristic mean confidence difference attack."
        detailed_str = "The attack is executed according to: \
            1. Compute the logits of the target model \
            2. Compute the expected value of the logits using shadow models \
            3. Set the score to the difference of the logits and expected logits."
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
        logger.info("Preparing shadow models for MCA attack")
        # Check number of shadow models that are available

        # sample dataset to compute histogram
        logger.info("Preparing attack data for training the MCA attack")

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
        logger.info("Running MCA attack")
        if self.online:
            audit_data_indices, in_members, out_members = self._filter_audit_data_for_online_attack(self.shadow_model_indices)
        else:
            audit_data_indices = self.audit_dataset["data"]
            in_members = self.audit_dataset["in_members"]
            out_members = self.audit_dataset["out_members"]

        n_audit_points = len(audit_data_indices)
        ground_truth_indices = self.handler.get_labels(audit_data_indices).astype(int)

        # run target points through real model to get logits
        logits_target = np.array(self.signal([self.target_model], self.handler, audit_data_indices)).squeeze(axis=0)
        # collect the log confidence output of the correct class (which is the negative cross-entropy loss)
        unnormalized_conf_target = softmax_logits(logits_target, self.temperature)[np.arange(n_audit_points),ground_truth_indices]

        # run points through shadow models and collect the log confidence values
        logits_shadow_models = self.signal(self.shadow_models, self.handler, audit_data_indices)
        unnormalized_conf_shadow_models = np.array([softmax_logits(x, self.temperature)[np.arange(n_audit_points),ground_truth_indices] for x in logits_shadow_models])  # noqa: E501
        threshold = unnormalized_conf_shadow_models.mean(axis=0)

        score = unnormalized_conf_target / threshold

        # pick out the in-members and out-members signals
        self.in_member_signals = score[in_members].reshape(-1,1)
        self.out_member_signals = score[out_members].reshape(-1,1)

        # set true labels for being in the training dataset
        true_labels = np.concatenate([np.ones(len(self.in_member_signals)),np.zeros(len(self.out_member_signals)),])
        signal_values = np.concatenate([self.in_member_signals, self.out_member_signals])

        return MIAResult.from_full_scores(true_membership=true_labels,
                                    signal_values=signal_values,
                                    result_name="MCA",
                                    metadata=self.configs.model_dump())
