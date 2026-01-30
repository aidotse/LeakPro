"""Implementation of the RMIA-Direct attack."""
import os
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator
from scipy.stats import norm
from tqdm import tqdm

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.attacks.utils.utils import softmax_logits
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.signals.signal import get_signal_from_name
from leakpro.utils.import_helper import Self, Tuple
from leakpro.utils.logger import logger


class AttackRMIADirect(AbstractMIA):
    """Implementation of the RMIA-Direct attack."""

    class AttackConfig(BaseModel):
        """Configuration for the RMIA attack."""

        
        signal_name: str = Field(default="ModelRescaledLogits", description="What signal to use.")
        num_shadow_models: int = Field(default=2,
                                       ge=2,
                                       description="Number of shadow models")
        training_data_fraction: float = Field(default=0.5,
                                              ge=0.0,
                                              le=1.0,
                                              description="Part of available attack data to use for shadow models")
        attack_data_fraction: float = Field(default=0.1,
                                            ge=0.0,
                                            le=1.0,
                                            description="Part of available attack data to use for attack")
        online: bool = Field(default=False,
                             description="Online vs offline attack")
        # Parameters to be used with optuna
        gamma: float = Field(default=2.0,
                        ge=0.0,
                        description="Parameter to threshold LLRs")


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
        logger.info("Configuring the RMIA-Direct attack")
        # Initializes the pydantic object using the user-provided configs
        # This will ensure that the user-provided configs are valid
        self.configs = self.AttackConfig() if configs is None else self.AttackConfig(**configs)

        # Call the parent class constructor. It will check the configs.
        super().__init__(handler)

        # Assign the configuration parameters to the object
        for key, value in self.configs.model_dump().items():
            setattr(self, key, value)

        if self.online is False:
            raise ValueError("Only online RMIA-Direct is defined.")

        self.shadow_models = []
        self.signal = get_signal_from_name(self.signal_name)
        self.epsilon = 1e-6
        self.shadow_models = None
        self.shadow_model_indices = None

    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "RMIA-Direct attack"
        reference_str = "Zarifzadeh, Sajjad, Philippe Cheng-Jie Marc Liu, and Reza Shokri. \
            Low-Cost High-Power Membership Inference by Boosting Relativity. (2023)."
        summary_str = "RMIA-Direct performs membership inference by computing pairwise likelihood \
rati                   between a target point x and random population samples z."
        detailed_str = "RMIA-Direct performs membership inference by computing pairwise likelihood \
                        ratios between a target point x and random population samples z. The likelihood\
                        ratio compares Pr(x|θ)/Pr(x) against Pr(z|θ)/Pr(z), where Pr(·) is estimated using\
                        reference (OUT) models. The attack score is the fraction of population samples z\
                        for which the likelihood ratio exceeds a threshold γ."
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

        logger.info("Preparing shadow models for RMIA-Direct attack")
        # Check number of shadow models that are available

        # sample dataset to compute histogram
        logger.info("Preparing attack data for training the RMIA-Direct attack")

        # Get all available indices for attack dataset, if self.online = True, include training and test data
        self.attack_data_indices = self.sample_indices_from_population(include_aux_indices = not self.online,
                                                                       include_train_indices = self.online,
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
        logger.info("Running RMIA online attack")

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
        
        assert len(audit_data_indices) == len(in_members) + len(out_members)

        if len(audit_data_indices) == 0:
            raise ValueError("No points in the audit dataset are used for the shadow models")
        
        in_indices_mask = in_indices_mask[:,mask]
        n_audit_points = len(audit_data_indices)

        logger.info(f"Number of points in the audit dataset that are used for online attack: {len(audit_data_indices)}")

        # STEP 3: Run the attack
        # run audit points through target and shadow models to get logits
        x_logits_target_model = np.array(self.signal([self.target_model], self.handler, audit_data_indices)).squeeze()
        x_logits_shadow_models = np.array(self.signal(self.shadow_models, self.handler, audit_data_indices))
        
        # Make a "random sample" to compute p(z) for points in attack dataset on the OUT shadow models for each audit point
        self.attack_data_index = self.sample_indices_from_population(include_aux_indices= not self.online,
                                                                     include_train_indices = self.online,
                                                                     include_test_indices = self.online)
        if len(self.attack_data_index) == 0:
            raise ValueError("There are no auxilliary points to use for the attack.")
        n_attack_points = int(self.attack_data_fraction * len(self.attack_data_index))

        # subsample the attack data based on the fraction
        logger.info(f"Subsampling attack data from {len(self.attack_data_index)} points")
        self.attack_data_index = np.random.choice(
            self.attack_data_index,
            n_attack_points,
            replace=False
        )
        logger.info(f"Number of attack data points after subsampling: {len(self.attack_data_index)}")
        assert len(self.attack_data_index) == n_attack_points

        # Run sampled attack points through target and shadow models
        attack_data_in_indices_mask = ShadowModelHandler().get_in_indices_mask(self.shadow_model_indices, self.attack_data_index).T
        z_logits_target_model = np.array(self.signal([self.target_model], self.handler, self.attack_data_index)).squeeze()
        z_logits_shadow_models = np.array(self.signal(self.shadow_models, self.handler, self.attack_data_index))

        score = np.zeros(n_audit_points)
        for i in tqdm(range(n_audit_points), desc="Calculating likelihoods"):
            x_mask = ~attack_data_in_indices_mask & in_indices_mask[:, i:i+1] # shape (num_shadow_models, num_z)
            z_mask = attack_data_in_indices_mask & ~in_indices_mask[:, i:i+1] # shape (num_shadow_models, num_z)

            valid_z = (np.sum(x_mask, axis=0) != 0) & (np.sum(z_mask, axis=0) != 0)
            num_valid_z = np.sum(valid_z)
            x_mask = x_mask[:, valid_z] # shape (num_shadow_models, num_valid_z)
            z_mask = z_mask[:, valid_z] # shape (num_shadow_models, num_valid_z)

            
            #logger.info(x_mask.shape, z_mask.shape)

            x_sm = x_logits_shadow_models[:, i:i+1].repeat(num_valid_z, axis=1) # shape (num_shadow_models, num_valid_z)
            x_tgt = x_logits_target_model[i:i+1].repeat(num_valid_z) # shape (num_valid_z,)

            z_sm = z_logits_shadow_models[:, valid_z] # shape (num_shadow_models, num_valid_z)
            z_tgt = z_logits_target_model[valid_z] # shape (num_valid_z,)

            #print(x_sm.shape, x_tgt.shape, z_sm.shape, z_tgt.shape)

            x_ratio_numer = norm.logpdf(x_tgt, np.mean(x_sm, where=x_mask, axis=0), np.std(x_sm, where=x_mask, axis=0) + self.epsilon)
            x_ratio_denom = norm.logpdf(x_tgt, np.mean(x_sm, where=z_mask, axis=0), np.std(x_sm, where=z_mask, axis=0) + self.epsilon)
            z_ratio_numer = norm.logpdf(z_tgt, np.mean(z_sm, where=x_mask, axis=0), np.std(z_sm, where=x_mask, axis=0) + self.epsilon)
            z_ratio_denom = norm.logpdf(z_tgt, np.mean(z_sm, where=z_mask, axis=0), np.std(z_sm, where=z_mask, axis=0) + self.epsilon)
            log_pr = x_ratio_numer + z_ratio_numer - x_ratio_denom - z_ratio_denom
            log_pr = log_pr[~np.isnan(log_pr)]
            score[i] = np.mean(log_pr > self.gamma)

        # pick out the in-members and out-members signals
        self.in_member_signals = score[in_members].reshape(-1,1)
        self.out_member_signals = score[out_members].reshape(-1,1)
        
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

        return MIAResult.from_full_scores(true_membership=true_labels,
                                    signal_values=signal_values,
                                    result_name="RMIA-Direct",
                                    metadata=self.configs.model_dump())


