"""Implementation of the MCA attack."""

import numpy as np
from pydantic import BaseModel, Field

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.signals.signal import ModelLogits
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class MCAConfig(BaseModel):
    """Configuration for the MCA attack."""

    num_shadow_models: int = Field(default=1, ge=1, description="Number of shadow models")
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


    def _online_attack(self:Self) -> None:
        logger.info("Running MCA online attack")

        # STEP 1: find out which audit data points can actually be audited
        # find the shadow models that are trained on what points in the audit dataset
        in_indices_mask = ShadowModelHandler().get_in_indices_mask(self.shadow_model_indices, self.audit_dataset["data"]).T
        # filter out the points that no shadow model has seen and points that all shadow models have seen
        num_shadow_models_seen_points = np.sum(in_indices_mask, axis=0)
        # make sure that the audit points are included in the shadow model training (but not all)
        mask = (num_shadow_models_seen_points > 0) & (num_shadow_models_seen_points < self.num_shadow_models)

        # STEP 2: Select datapoints that are auditable
        audit_data_indices = self.audit_dataset["data"][mask]
        # find out how many in-members survived the filtering
        in_members = np.arange(np.sum(mask[self.audit_dataset["in_members"]]))
        # find out how many out-members survived the filtering
        num_out_members = np.sum(mask[self.audit_dataset["out_members"]])
        out_members = np.arange(len(in_members), len(in_members) + num_out_members)
        ground_truth_indices = self.handler.get_labels(audit_data_indices).astype(int)

        assert len(audit_data_indices) == len(ground_truth_indices)
        assert len(audit_data_indices) == len(in_members) + len(out_members)

        if len(audit_data_indices) == 0:
            raise ValueError("No points in the audit dataset are used for the shadow models")

        logger.info(f"Number of points in the audit dataset that are used for online attack: {len(audit_data_indices)}")

        # STEP 3: Run the attack
        # run points through target model to get logits
        logits_target = np.array(self.signal([self.target_model], self.handler, audit_data_indices)).squeeze(axis=0)
        n_audit_points = len(audit_data_indices)
        # collect the unnormilized output of the correct class (which is the negative cross-entropy loss)
        unnormilized_conf_target = logits_target[np.arange(n_audit_points),ground_truth_indices]

        # run points through shadow models and collect the unnormilized confidence values
        logits_shadow_models = self.signal(self.shadow_models, self.handler, audit_data_indices)
        unnormilized_conf_shadow_models = np.array([x[np.arange(n_audit_points),ground_truth_indices] for x in logits_shadow_models])  # noqa: E501

        threshold = unnormilized_conf_shadow_models.mean(axis=0)

        score = unnormilized_conf_target - threshold

        # pick out the in-members and out-members signals
        self.in_member_signals = score[in_members].reshape(-1,1)
        self.out_member_signals = score[out_members].reshape(-1,1)

    def _offline_attack(self:Self) -> None:
        logger.info("Running MCA offline attack")

        # run target points through real model to get logits
        logits_target = np.array(self.signal([self.target_model], self.handler, self.audit_dataset["data"])).squeeze(axis=0)
        ground_truth_indices = self.handler.get_labels(self.audit_dataset["data"])
        assert np.issubdtype(ground_truth_indices.dtype, np.integer)

        n_audit_points = len(self.audit_dataset["data"])
        # collect the log confidence output of the correct class (which is the negative cross-entropy loss)
        unnormilized_conf_target = logits_target[np.arange(n_audit_points),ground_truth_indices]

        # run points through shadow models and collect the log confidence values
        logits_shadow_models = self.signal(self.shadow_models, self.handler, self.audit_dataset["data"])
        unnormilized_conf_shadow_models = np.array([x[np.arange(n_audit_points),ground_truth_indices] for x in logits_shadow_models])  # noqa: E501
        threshold = unnormilized_conf_shadow_models.mean(axis=0)

        in_members = self.audit_dataset["in_members"]
        out_members = self.audit_dataset["out_members"]

        score = unnormilized_conf_target - threshold

        # pick out the in-members and out-members signals
        self.in_member_signals = score[in_members].reshape(-1,1)
        self.out_member_signals = score[out_members].reshape(-1,1)

    def run_attack(self:Self) -> MIAResult:
        """Run the attack on the target model and dataset.

        Returns
        -------
            Result(s) of the metric.

        """
        # perform the attack
        if self.online is True:
            self._online_attack()
        else:
            self._offline_attack()

        # set true labels for being in the training dataset
        true_labels = np.concatenate(
            [
                np.ones(len(self.in_member_signals)),
                np.zeros(len(self.out_member_signals)),
            ]
        )
        signal_values = np.concatenate([self.in_member_signals, self.out_member_signals])

        return MIAResult.from_full_scores(true_membership=true_labels,
                                    signal_values=signal_values,
                                    result_name="MCA",
                                    metadata=self.configs.model_dump())
