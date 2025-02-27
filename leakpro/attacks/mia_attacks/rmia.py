"""Implementation of the RMIA attack."""
import os

import numpy as np
from pydantic import BaseModel, Field, model_validator

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.attacks.utils.utils import softmax_logits
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.metrics.attack_result import MIAResult
from leakpro.signals.signal import ModelLogits
from leakpro.utils.import_helper import Self, Tuple
from leakpro.utils.logger import logger


class AttackRMIA(AbstractMIA):
    """Implementation of the RMIA attack."""

    class AttackConfig(BaseModel):
        """Configuration for the RMIA attack."""

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
        attack_data_fraction: float = Field(default=0.1,
                                            ge=0.0,
                                            le=1.0,
                                            description="Part of available attack data to use for attack")
        online: bool = Field(default=False,
                             description="Online vs offline attack")
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

        if self.online is False and self.population_size == self.audit_size:
            raise ValueError("The audit dataset is the same size as the population dataset. \
                    There is no data left for the shadow models.")

        self.shadow_models = []
        self.signal = ModelLogits()
        self.epsilon = 1e-6
        self.shadow_models = None
        self.shadow_model_indices = None

        # Folder to store intermediate results
        self.attack_folder_path = "leakpro_output/attacks/rmia"
        os.makedirs(self.attack_folder_path, exist_ok=True)
        self.load_for_optuna = False


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

    def _prepare_shadow_models(self:Self) -> None:

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

    def _prepare_offline_aux_attack_logits(self:Self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare the logits for the offline attack on the auxiliary dataset."""

        # subsample the attack data based on the fraction
        logger.info(f"Subsampling attack data from {len(self.attack_data_indices)} points")
        n_points = int(self.attack_data_fraction * len(self.attack_data_indices))
        chosen_attack_data_indices = np.random.choice(self.attack_data_indices, n_points, replace=False)
        logger.info(f"Number of attack data points after subsampling: {len(chosen_attack_data_indices)}")

        # get the true label indices
        z_true_labels = self.handler.get_labels(chosen_attack_data_indices)
        assert np.issubdtype(z_true_labels.dtype, np.integer)
        f_z_true_labels = f"{self.attack_folder_path}/z_true_labels.npy"
        np.save(f_z_true_labels, z_true_labels)

        # run points through real model to collect the logits
        logits_theta = np.array(self.signal([self.target_model], self.handler, chosen_attack_data_indices))

        # Store the ratio of p(z|theta) to p(z) for the audit dataset to be used in other optuna
        f_logits_theta = f"{self.attack_folder_path}/logits_theta.npy"
        np.save(f_logits_theta, logits_theta)

        # run points through shadow models and collect the logits
        logits_shadow_models = self.signal(self.shadow_models, self.handler, chosen_attack_data_indices)
        f_logits_sm = f"{self.attack_folder_path}/logits_shadow_models.npy"
        np.save(f_logits_sm, logits_shadow_models)

        return logits_theta, logits_shadow_models, z_true_labels

    def prepare_attack(self:Self) -> None:
        """Prepare Data needed for running the attack on the target model and dataset.

        Signals are computed on the auxiliary model(s) and dataset.
        """
        logger.info("Preparing shadow models for RMIA attack")

        # If we already have one run, we dont need to check for shadow models as logits are stored
        if not self.load_for_optuna:
            self._prepare_shadow_models()

        # compute quantities that are not touching the audit dataset
        if self.online is False:
            # compute the ratio of p(z|theta) (target model) to p(z)=sum_{theta'} p(z|theta') (shadow models)
            # for all points in the attack dataset output from signal: # models x # data points x # classes

            if not self.load_for_optuna:
                logits_theta, logits_shadow_models, z_true_labels = self._prepare_offline_aux_attack_logits()
            else:
                logits_theta = np.load(f"{self.attack_folder_path}/logits_theta.npy")
                logits_shadow_models = np.load(f"{self.attack_folder_path}/logits_shadow_models.npy")
                z_true_labels = np.load(f"{self.attack_folder_path}/z_true_labels.npy")

            # collect the softmax output of the correct class
            n_attack_points = len(z_true_labels)
            p_z_given_theta = softmax_logits(logits_theta, self.temperature)[:,np.arange(n_attack_points),z_true_labels]

            # collect the softmax output of the correct class for each shadow model
            sm_logits_shadow_models = [softmax_logits(x, self.temperature) for x in logits_shadow_models]
            p_z_given_shadow_models = np.array([x[np.arange(n_attack_points),z_true_labels] for x in sm_logits_shadow_models])

            # evaluate the marginal p(z)
            p_z = np.mean(p_z_given_shadow_models, axis=0) if len(self.shadow_models) > 1 else p_z_given_shadow_models.squeeze()
            p_z = 0.5*((self.offline_a + 1) * p_z + (1-self.offline_a))

            self.ratio_z = p_z_given_theta / (p_z + self.epsilon)

    def _prepare_online_audit_logits(self:Self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare the logits for the online attack on the audit dataset."""
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

        out_model_indices = ~in_indices_mask[:,mask]

        logger.info(f"Number of points in the audit dataset that are used for online attack: {len(audit_data_indices)}")

        # run points through target model to get logits
        logits_theta = np.array(self.signal([self.target_model], self.handler, audit_data_indices))
        # run points through shadow models to get logits
        logits_shadow_models = self.signal(self.shadow_models, self.handler, audit_data_indices)

        f_logits_theta = f"{self.attack_folder_path}/logits_audit_theta.npy"
        np.save(f_logits_theta, logits_theta)
        f_logits_sm = f"{self.attack_folder_path}/logits_audit_shadow_models.npy"
        np.save(f_logits_sm, logits_shadow_models)
        f_ground_truth_indices = f"{self.attack_folder_path}/ground_truth_indices.npy"
        np.save(f_ground_truth_indices, ground_truth_indices)
        f_out_model_indices = f"{self.attack_folder_path}/out_model_indices.npy"
        np.save(f_out_model_indices, out_model_indices)
        f_in_members = f"{self.attack_folder_path}/in_members.npy"
        np.save(f_in_members, in_members)
        f_out_members = f"{self.attack_folder_path}/out_members.npy"
        np.save(f_out_members, out_members)

        return logits_theta, logits_shadow_models, ground_truth_indices, out_model_indices, in_members, out_members

    def _prepare_online_aux_attack_logits(self:Self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
               # Make a "random sample" to compute p(z) for points in attack dataset on the OUT shadow models for each audit point
        self.attack_data_index = self.sample_indices_from_population(include_train_indices = False,
                                                                     include_test_indices = False)
        if len(self.attack_data_index) == 0:
            raise ValueError("There are no auxilliary points to use for the attack.")

        # subsample the attack data based on the fraction
        logger.info(f"Subsampling attack data from {len(self.attack_data_index)} points")
        self.attack_data_index = np.random.choice(
            self.attack_data_index,
            int(self.attack_data_fraction * len(self.attack_data_index)),
            replace=False
        )
        logger.info(f"Number of attack data points after subsampling: {len(self.attack_data_index)}")

        # get the true label indices
        z_true_labels = self.handler.get_labels(self.attack_data_index).astype(int)
        assert np.issubdtype(z_true_labels.dtype, np.integer)

        # run points through real model to collect the logits
        logits_target_model = np.array(self.signal([self.target_model], self.handler, self.attack_data_index))
        # run points through shadow models and collect the logits
        logits_shadow_models = self.signal(self.shadow_models, self.handler, self.attack_data_index)

        f_logits_target_model = f"{self.attack_folder_path}/logits_aux_target_model.npy"
        np.save(f_logits_target_model, logits_target_model)
        f_logits_shadow_models = f"{self.attack_folder_path}/logits_aux_shadow_models.npy"
        np.save(f_logits_shadow_models, logits_shadow_models)
        f_z_true_labels = f"{self.attack_folder_path}/z_true_labels.npy"
        np.save(f_z_true_labels, z_true_labels)

        return logits_target_model, logits_shadow_models, z_true_labels

    def _prepare_offline_audit_logits(self:Self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare the logits for the offline attack on the audit dataset."""

        # run target points through real model to get logits
        logits_theta = np.array(self.signal([self.target_model], self.handler, self.audit_dataset["data"]))
        f_logits_theta = f"{self.attack_folder_path}/logits_audit_theta.npy"
        np.save(f_logits_theta, logits_theta)

        # run points through shadow models and collect the logits
        logits_shadow_models = self.signal(self.shadow_models, self.handler, self.audit_dataset["data"])
        f_logits_sm = f"{self.attack_folder_path}/logits_audit_shadow_models.npy"
        np.save(f_logits_sm, logits_shadow_models)

        return logits_theta, logits_shadow_models

    def _online_attack(self:Self) -> None:
        logger.info("Running RMIA online attack")
        if not self.load_for_optuna:
            logits_theta, logits_shadow_models, ground_truth_indices, out_model_indices, in_members, out_members = self._prepare_online_audit_logits()  # noqa: E501
        else:
            logits_theta = np.load(f"{self.attack_folder_path}/logits_audit_theta.npy")
            logits_shadow_models = np.load(f"{self.attack_folder_path}/logits_audit_shadow_models.npy")
            ground_truth_indices = np.load(f"{self.attack_folder_path}/ground_truth_indices.npy")
            out_model_indices = np.load(f"{self.attack_folder_path}/out_model_indices.npy")
            in_members = np.load(f"{self.attack_folder_path}/in_members.npy")
            out_members = np.load(f"{self.attack_folder_path}/out_members.npy")

        # STEP 3: Run the attack
        # collect the softmax output of the correct class
        n_audit_points = len(ground_truth_indices)
        p_x_given_target_model = softmax_logits(logits_theta, self.temperature)[:,np.arange(n_audit_points),ground_truth_indices]

        # run points through shadow models, colelct logits and compute p(x)
        sm_shadow_models = [softmax_logits(x, self.temperature) for x in logits_shadow_models]
        p_x_given_shadow_models = np.array([x[np.arange(n_audit_points),ground_truth_indices] for x in sm_shadow_models])

        p_x = np.mean(p_x_given_shadow_models, axis=0) if len(self.shadow_models) > 1 else p_x_given_shadow_models.squeeze()
        # compute the ratio of p(x|theta) to p(x)
        ratio_x = p_x_given_target_model / (p_x + self.epsilon)

        if not self.load_for_optuna:
            logits_target_model, logits_shadow_models, z_true_labels = self._prepare_online_aux_attack_logits()
        else:
            logits_target_model = np.load(f"{self.attack_folder_path}/logits_aux_target_model.npy")
            logits_shadow_models = np.load(f"{self.attack_folder_path}/logits_aux_shadow_models.npy")
            z_true_labels = np.load(f"{self.attack_folder_path}/z_true_labels.npy")

        # collect the softmax output of the correct class
        n_attack_points = len(self.attack_data_index)
        p_z_given_target_model = softmax_logits(logits_target_model, self.temperature)[:,np.arange(n_attack_points),z_true_labels]

        # collect the softmax output of the correct class for each shadow model
        sm_shadow_models = [softmax_logits(x, self.temperature) for x in logits_shadow_models]
        p_z_given_shadow_models = np.array([x[np.arange(n_attack_points),z_true_labels] for x in sm_shadow_models])

        # evaluate the marginal p(z) by averaging over the OUT models
        p_z = np.zeros((n_audit_points, len(self.attack_data_index)))
        for i in range(n_audit_points):
            model_mask = out_model_indices[:,i]
            p_z[i] = np.mean(p_z_given_shadow_models[model_mask, :], axis=0)
        ratio_z = p_z_given_target_model / (p_z + self.epsilon)

        # for each x, compute the score
        likelihoods = ratio_x.T / ratio_z
        score = np.mean(likelihoods > self.gamma, axis=1)

        # pick out the in-members and out-members signals
        self.in_member_signals = score[in_members].reshape(-1,1)
        self.out_member_signals = score[out_members].reshape(-1,1)

    def _offline_attack(self:Self) -> None:
        logger.info("Running RMIA offline attack")

        if not self.load_for_optuna:
            logits_theta, logits_shadow_models = self._prepare_offline_audit_logits()
        else:
            logits_theta = np.load(f"{self.attack_folder_path}/logits_audit_theta.npy")
            logits_shadow_models = np.load(f"{self.attack_folder_path}/logits_audit_shadow_models.npy")

        # collect the softmax output of the correct class
        ground_truth_indices = self.handler.get_labels(self.audit_dataset["data"])
        assert np.issubdtype(ground_truth_indices.dtype, np.integer)

        n_audit_points = len(self.audit_dataset["data"])
        p_x_given_target_model = softmax_logits(logits_theta, self.temperature)[:,np.arange(n_audit_points),ground_truth_indices]

        # collect the softmax output of the correct class for each shadow model
        # Stack to dimension # models x # data points
        sm_shadow_models = [softmax_logits(x, self.temperature) for x in logits_shadow_models]
        p_x_given_shadow_models = np.array([x[np.arange(n_audit_points),ground_truth_indices] for x in sm_shadow_models])

        # evaluate the marginal p_out(x) by averaging the output of the shadow models
        p_x_out = np.mean(p_x_given_shadow_models, axis=0) if len(self.shadow_models) > 1 else p_x_given_shadow_models.squeeze()

        # compute the marginal p(x) from P_out and p_in where p_in = a*p_out+b
        p_x = 0.5*((self.offline_a + 1) * p_x_out + (1-self.offline_a))

        # compute the ratio of p(x|theta) to p(x)
        ratio_x = p_x_given_target_model / (p_x + self.epsilon)

        # for each x, compare it with the ratio of all z points
        likelihoods = ratio_x.T / self.ratio_z

        in_members = self.audit_dataset["in_members"]
        out_members = self.audit_dataset["out_members"]

        score = np.mean(likelihoods > self.gamma, axis=1)

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

        # create thresholds
        min_signal_val = np.min(np.concatenate([self.in_member_signals, self.out_member_signals]))
        max_signal_val = np.max(np.concatenate([self.in_member_signals, self.out_member_signals]))
        thresholds = np.linspace(min_signal_val, max_signal_val, 1000)

        member_preds = np.greater(self.in_member_signals, thresholds).T
        non_member_preds = np.greater(self.out_member_signals, thresholds).T

        # what does the attack predict on test and train dataset
        predictions = np.concatenate([member_preds, non_member_preds], axis=1)
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

        # Ensure we use the stored quantities from now
        self.load_for_optuna = True

        # compute ROC, TP, TN etc
        return MIAResult(
            predicted_labels=predictions,
            true_labels=true_labels,
            predictions_proba=None,
            signal_values=signal_values,
        )

    def reset_attack(self: Self, config:BaseModel) -> None:
        """Reset attack to initial state."""

        # Assign the new configuration parameters to the object
        for key, value in config.model_dump().items():
            setattr(self, key, value)

        # new hyperparameters have been set, let's prepare the attack again
        self.prepare_attack()


