"""Implementation of the LSET Laplace attack."""

import numpy as np
from laplace import Laplace
from pydantic import BaseModel, Field

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.attacks.utils.utils import softmax_logits
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.signals.signal import ModelLogits
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class LSETLaplaceConfig(BaseModel):
    """Configuration for the LSET Laplace attack."""

    temperature: float = Field(default=2.0, ge=0.0, description="Softmax temperature")
    training_data_fraction: float = Field(default=0.5, ge=0.0, le=1.0, description="Part of available attack data to use for shadow model")  # noqa: E501
    online: bool = Field(default=False, description="Online vs offline attack")


class AttackLSETLaplace(AbstractMIA):
    """Implementation of the LSET Laplace attack."""

    AttackConfig = LSETLaplaceConfig # required config for attack

    def __init__(self:Self,
                 handler: AbstractInputHandler,
                 configs: dict
                 ) -> None:
        """Initialize the LSET Laplace attack.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.
            configs (dict): Configuration parameters for the attack.

        """
        logger.info("Configuring the LSET Laplace attack")
        # Initializes the pydantic object using the user-provided configs
        # This will ensure that the user-provided configs are valid
        self.configs = LSETLaplaceConfig() if configs is None else LSETLaplaceConfig(**configs)

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
        self.la = None


    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "LSET Laplace attack"
        reference_str = "No reference"
        summary_str = "A simple loss based attack with a log-sum-exponential approximation of the threshold, using the Laplace approximation."  # noqa: E501
        detailed_str = "The attack is executed according to: \
            1. Compute the log confidence values of the target samples using the target model \
            2. Use the Laplace approximation to get the expected log confidence value. \
            3. Return the difference of the target models log confidence values and the expected log confidence values."
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
        logger.info("Preparing shadow model for LSET Laplace attack")
        # Check number of shadow models that are available

        # sample dataset to compute histogram
        logger.info("Preparing attack data for training the LSET Laplace attack")

        # Get all available indices for attack dataset, if self.online = True, include training and test data
        self.attack_data_indices = self.sample_indices_from_population(include_train_indices = self.online,
                                                                       include_test_indices = self.online)

        # train shadow models
        logger.info(f"Check for 1 shadow model (dataset: {len(self.attack_data_indices)} points)")
        self.shadow_model_indices = ShadowModelHandler().create_shadow_models(
            num_models = 1,
            shadow_population = self.attack_data_indices,
            training_fraction = self.training_data_fraction,
            online = self.online)
        # load shadow models
        self.shadow_models, _ = ShadowModelHandler().get_shadow_models(self.shadow_model_indices)
        train_indices = ShadowModelHandler().get_shadow_model_metadata(self.shadow_model_indices)[0].train_indices
        train_loader = self.handler.get_dataloader(train_indices, shuffle=False)
        self.la = Laplace(self.shadow_models[0].model_obj,
            "classification",
            subset_of_weights="last_layer",
            hessian_structure="kron")
        self.la.fit(train_loader)
        self.la.optimize_prior_precision()

    def _online_attack(self:Self) -> None:
        logger.info("Running LSET Laplace online attack")

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
        # collect the log confidence output of the correct class (which is the negative cross-entropy loss)
        log_conf_target = np.log(softmax_logits(logits_target, self.temperature)[np.arange(n_audit_points),ground_truth_indices])

        data_loader = self.handler.get_dataloader(audit_data_indices, shuffle=False)
        assert self._is_shuffling(data_loader) is False, "DataLoader must not shuffle data to maintain order of indices"

        # Get predictions using laplace approximation
        pred_la = []
        for x, _ in data_loader:
            pred_la.extend(self.la(x, pred_type="glm", link_approx="probit"))
        pred_la = np.vstack(pred_la)
        ref_log_confs = np.log(pred_la[np.arange(n_audit_points),ground_truth_indices])

        score = log_conf_target - ref_log_confs

        # pick out the in-members and out-members signals
        self.in_member_signals = score[in_members].reshape(-1,1)
        self.out_member_signals = score[out_members].reshape(-1,1)

    def _offline_attack(self:Self) -> None:
        logger.info("Running LSET Laplace offline attack")

        # run target points through real model to get logits
        logits_target = np.array(self.signal([self.target_model], self.handler, self.audit_dataset["data"])).squeeze(axis=0)
        ground_truth_indices = self.handler.get_labels(self.audit_dataset["data"])
        assert np.issubdtype(ground_truth_indices.dtype, np.integer)

        n_audit_points = len(self.audit_dataset["data"])
        # collect the log confidence output of the correct class (which is the negative cross-entropy loss)
        log_conf_target = np.log(softmax_logits(logits_target, self.temperature)[np.arange(n_audit_points),ground_truth_indices])

        data_loader = self.handler.get_dataloader(self.audit_dataset["data"], shuffle=False)

        # Get predictions using laplace approximation
        pred_la = []
        for x, _ in data_loader:
            pred_la.extend(self.la(x, pred_type="glm", link_approx="probit"))
        pred_la = np.vstack(pred_la)
        ref_log_confs = np.log(pred_la[np.arange(n_audit_points),ground_truth_indices])

        in_members = self.audit_dataset["in_members"]
        out_members = self.audit_dataset["out_members"]

        score = log_conf_target - ref_log_confs

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
        signal_values = np.concatenate(
            [self.in_member_signals, self.out_member_signals]
        )

        # compute ROC, TP, TN etc
        return MIAResult.from_full_scores(true_membership=true_labels,
                                    signal_values=signal_values,
                                    result_name="LSET_Laplace",
                                    metadata=self.configs.model_dump())
