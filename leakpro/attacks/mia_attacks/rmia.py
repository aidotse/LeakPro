"""Implementation of the RMIA attack."""
import numpy as np
import torch
from pydantic import BaseModel, Field, model_validator

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.attacks.utils.utils import softmax_logits
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.utils.import_helper import Self
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
        online: bool = Field(default=False,
                             description="Online vs offline attack")
        # Parameters to be used with optuna
        z_data_sample_fraction: float = Field(default=0.5,
                                                ge=0.0,
                                                le=1.0,
                                                description="Part of available attack data to use for estimating p(z)",)
        gamma: float = Field(default=2.0,
                        ge=0.0,
                        description="Parameter to threshold LLRs",
                        json_schema_extra = {"optuna": {"type": "float", "low": 0.1, "high": 10, "log": True}})
        offline_a: float = Field(default=0.33,
                                 ge=0.0,
                                 le=1.0,
                                 description="Parameter to estimate the marginal p(x)",
                                 json_schema_extra = {"optuna": {"type": "float", "low": 0.0, "high": 1.0,"enabled_if": lambda model: not model.online}})  # noqa: E501

        # Vectorized-path options (minimal & safe defaults)
        vectorized: bool = Field(
            default=False,
            description="Enable the vectorized RMIA fast-path (keeps classic path when False)."
        )

        vec_batch: int = Field(
            default=1024,
            ge=1,
            description=(
                "Number of audit points to process per inner batch in the vectorized path. "
                "Lower this if you have limited memory; keep larger for speed."
            ),
        )

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

            self.ground_truth_indices = self.handler.get_labels(self.audit_dataset["data"])
            self.logits_theta = ShadowModelHandler().load_logits(name="target")
            self.logits_shadow_models = []
            for indx in self.shadow_model_indices:
                self.logits_shadow_models.append(ShadowModelHandler().load_logits(indx=indx))

            # suggest calculating here for use when calculating p_z_given_* and p_x_given_* later 
            self.shadow_models_inmask = ShadowModelHandler().get_in_indices_mask(self.shadow_model_indices, self.audit_dataset["data"])
            self.target_gtlabel_probs = rmia_get_gtlprobs(self.logits_theta, self.ground_truth_indices, self.temperature) # check dims 
            self.shadow_gtl_probs = np.vstack([
                rmia_get_gtlprobs(logits, self.ground_truth_indices, self.temperature) for logits in self.logits_shadow_models
            ]) 

        # collect the softmax output of the correct class
        n_attack_points = self.z_data_sample_fraction * len(self.handler.population) # points to compute p(z)

        # pick random indices sampled from the attack data
        z_indices = np.random.choice(self.attack_data_indices, size=int(n_attack_points), replace=False)
        z_labels = self.handler.get_labels(z_indices)

        p_z_given_theta = softmax_logits(self.logits_theta, self.temperature)[z_indices,z_labels]  # noqa: E501
        p_z_given_theta = np.atleast_2d(p_z_given_theta)

        # collect the softmax output of the correct class for each shadow model
        sm_logits_shadow_models = [softmax_logits(x, self.temperature) for x in self.logits_shadow_models]
        p_z_given_shadow_models = np.array([x[z_indices,z_labels] for x in sm_logits_shadow_models])  # noqa: E501

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

        # collect the softmax output of the correct class
        n_audit_points = len(self.ground_truth_indices)
        p_x_given_theta = softmax_logits(self.logits_theta, self.temperature)[np.arange(n_audit_points),self.ground_truth_indices]
        p_x_given_theta = np.atleast_2d(p_x_given_theta)

        # run points through shadow models, colelct logits and compute p(x)
        sm_shadow_models = [softmax_logits(x, self.temperature) for x in self.logits_shadow_models]
        p_x_given_shadow_models = np.array([x[np.arange(n_audit_points),self.ground_truth_indices] for x in sm_shadow_models])

        if self.online is True:
            p_x = np.mean(p_x_given_shadow_models, axis=0, keepdims=True)
        else:
            # compute the marginal p(x) from P_out and p_in where p_in = a*p_out+b
            masked_values = np.where(self.out_indices, p_x_given_shadow_models, np.nan)
            p_x_out = np.nanmean(masked_values, axis=0)
            p_x = 0.5*((self.offline_a + 1) * p_x_out + (1-self.offline_a))

        # compute the ratio of p(x|theta) to p(x)
        ratio_x = p_x_given_theta / (p_x + self.epsilon)

        if self.configs.vectorized:
            #ratio_x = rmia_prep_ratio(target_gtlprobs, shadow_gtlprobs, shadow_inmasks, x_indices, online, offline_a, epsilon)
            #ratio_z = rmia_prep_ratio(target_gtlprobs, shadow_gtlprobs, shadow_inmasks, z_indices, online, offline_a, epsilon)
            logger.info(f"Vectorized RMIA takes in ratio_x with shape {ratio_x.shape}")
            logger.info(f"Vectorized RMIA takes in ratio_z with shape {self.ratio_z.shape}")
            score = rmia_use_torch(ratio_x, self.ratio_z, self.gamma, self.configs.vec_batch).reshape(1,-1)
            logger.info(f"Vectorized RMIA produces score with shape {score.shape}")
        else:
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



##### some helper functions used in the vectorized code of rmia #####

def rmia_get_gtlprobs(logits, labels, temperature=1.0, select = None):
    select = np.arange(len(labels)) if select is None else select
    assert len(select) == len(labels)
    assert logits.shape[0] > max(select)
    assert logits.shape[1] > max(labels)
    return softmax_logits(logits, temperature)[select,labels]

def rmia_prep_ratio(target_gtlprobs, shadow_gtlprobs, shadow_inmasks, indices=None, online=True, offline_a=0.33, epsilon=1e-12):
    if indices is None:
        prob_target = target_gtlprobs
        prob_shadow = shadow_gtlprobs
        mask_shadow = shadow_inmasks
    else:
        prob_target = target_gtlprobs[indices, :]
        prob_shadow = shadow_gtlprobs[indices, :]
        mask_shadow = shadow_inmasks[indices, :]

    if online:
        prob_prior = np.mean(prob_shadow, axis=1, keepdims=True)
    else:
        prob_in_masked = np.where(mask_shadow, np.nan, prob_shadow)
        prob_prior_out = np.nanmean(prob_in_masked, axis=1, keepdims=True)
        prob_prior = 0.5 * ((offline_a + 1) * prob_prior_out + (1 - offline_a))

    ratio = prob_target / (prob_prior + epsilon)
    return ratio

def rmia_use_torch(ratio_x, ratio_z, gamma=1.0, batch_size=1000, use_gpu_if_available=True):

    # Check that inputs are not matrices or higher-dimensional
    if np.prod(ratio_x.shape) != len(np.ravel(ratio_x)):
        raise ValueError("ratio_x must be a 1D array or a row/column vector.")
    if np.prod(ratio_z.shape) != len(np.ravel(ratio_z)):
        raise ValueError("ratio_z must be a 1D array or a row/column vector.")

    device = torch.device('cuda' if use_gpu_if_available and torch.cuda.is_available() else 'cpu')
    ratio_x_t = torch.from_numpy(np.ravel(ratio_x)).to(torch.float32).to(device).unsqueeze(1)  # shape: (n_x, 1)
    ratio_z_t = torch.from_numpy(np.ravel(ratio_z)).to(torch.float32).to(device).unsqueeze(1)  # shape: (n_z, 1)
    gamma_t = torch.tensor(gamma, dtype=torch.float32, device=device)

    n_audit_points = ratio_x_t.shape[0]
    score_list = []

    for start_idx in range(0, n_audit_points, batch_size):
        end_idx = min(start_idx + batch_size, n_audit_points)
        batch_x_t = ratio_x_t[start_idx:end_idx]  # shape: (batch_size, 1)
        likelihoods = batch_x_t / ratio_z_t.T     # shape: (batch_size, n_z)
        score_batch = torch.mean((likelihoods > gamma_t).float(), dim=1)
        score_list.append(score_batch)

    score_tensor = torch.cat(score_list)  # shape: (n_x,)
    return score_tensor.cpu().numpy()

def rmia_vectorised(target_gtlprobs, shadow_gtlprobs, shadow_inmasks=None, online=True, offline_a=0.33,
                    x_indices=None, z_indices=None, gamma=1.0, epsilon=1e-12, batch_size=1000, use_gpu_if_available=True):

    device = torch.device('cuda' if use_gpu_if_available and torch.cuda.is_available() else 'cpu')
    print("Available device:", device)

    if len(target_gtlprobs.shape) == 1:
        target_gtlprobs = target_gtlprobs.reshape(-1, 1)

    n_population = target_gtlprobs.shape[0]
    n_shadow_models = shadow_gtlprobs.shape[1]
    print("shadow_gtlprobs", shadow_gtlprobs.shape)
    assert shadow_gtlprobs.shape[0] == n_population

    if shadow_inmasks is None:
        shadow_inmasks = np.zeros_like(shadow_gtlprobs, dtype=bool)

    assert shadow_inmasks.shape == shadow_gtlprobs.shape
    print("shadow_inmasks", shadow_inmasks.shape)

    ratio_x = rmia_prep_ratio(target_gtlprobs, shadow_gtlprobs, shadow_inmasks, x_indices, online, offline_a, epsilon)
    ratio_z = rmia_prep_ratio(target_gtlprobs, shadow_gtlprobs, shadow_inmasks, z_indices, online, offline_a, epsilon)

    print("ratio_x", ratio_x.shape)
    print("ratio_z", ratio_z.shape)

    return rmia_use_torch(ratio_x, ratio_z, gamma, batch_size, use_gpu_if_available)


