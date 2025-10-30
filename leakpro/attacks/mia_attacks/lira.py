"""Implementation of the LiRA attack."""

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator
from scipy.stats import norm
from tqdm import tqdm

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.utils.import_helper import Self


class AttackLiRA(AbstractMIA):
    """Implementation of the LiRA attack."""

    class AttackConfig(BaseModel):
        """Configuration for the LiRA attack."""

        num_shadow_models: int = Field(default=1, ge=1, description="Number of shadow models")
        training_data_fraction: float = Field(default=0.5, ge=0.0, le=1.0, description="Part of available attack data to use for shadow models")  # noqa: E501
        online: bool = Field(default=False, description="Online vs offline attack")
        var_calculation: Literal["carlini", "individual_carlini", "fixed"] = Field(default="carlini", description="Variance estimation method to use [carlini, individual_carlini, fixed]")  # noqa: E501
        vectorized: bool = Field(default=False, description="Compute shadow-model scores in a single vectorized pass (faster) instead of per-sample loop (safer).")

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
        """Initialize the LiRA attack.

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

        self.shadow_models = []

    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "Likelihood Ratio Attack"

        reference_str = "Carlini N, et al. Membership Inference Attacks From First Principles"

        summary_str = "LiRA is a membership inference attack based on rescaled logits of a black-box model"

        detailed_str = "The attack is executed according to: \
            1. A fraction of the target model dataset is sampled to be included (in-) or excluded (out-) \
            from the shadow model training dataset. \
            2. The rescaled logits are used to estimate Gaussian distributions for in and out members \
            3. The thresholds are used to classify in-members and out-members. \
            4. The attack is evaluated on an audit dataset to determine the attack performance."

        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }

    def rescale_logits(self:Self, logits: np.ndarray, true_label:np.ndarray) -> np.ndarray:
        """Rescale the logits to a range of [0, 1].

        Args:
            logits (np.ndarray): The logits to be rescaled.
            true_label (np.ndarray): The true labels for the logits.

        Returns:
            np.ndarray: The rescaled logits.

        """
        if logits.shape[1] == 1:
            def sigmoid(z:np.ndarray) -> np.ndarray:
                return 1/(1 + np.exp(-z))
            positive_class_prob = sigmoid(logits).reshape(-1, 1)
            predictions = np.concatenate([1 - positive_class_prob, positive_class_prob], axis=1)
        else:
            predictions = logits - np.max(logits, axis=1, keepdims=True)
            predictions = np.exp(predictions)
            predictions = predictions/np.sum(predictions,axis=1, keepdims=True)

        count = predictions.shape[0]
        y_true = predictions[np.arange(count), true_label]
        predictions[np.arange(count), true_label] = 0

        y_wrong = np.sum(predictions, axis=1)
        output_signals = np.log(y_true+1e-45) - np.log(y_wrong+1e-45)
        return output_signals  # noqa: RET504

    def prepare_attack(self:Self)->None:
        """Prepares data to obtain metric on the target model and dataset, using signals computed on the auxiliary model/dataset.

        It selects a balanced subset of data samples from in-group and out-group members
        of the audit dataset, prepares the data for evaluation, and computes the logits
        for both shadow models and the target model.
        """

        # Fixed variance is used when the number of shadow models is below 32 (64, IN and OUT models)
        #       from (Membership Inference Attacks From First Principles)
        self.fix_var_threshold = 32

        self.attack_data_indices = self.sample_indices_from_population(include_train_indices = True,
                                                                       include_test_indices = True)

        self.shadow_model_indices = ShadowModelHandler().create_shadow_models(num_models = self.num_shadow_models,
                                                                              shadow_population =  self.attack_data_indices,
                                                                              training_fraction = self.training_data_fraction,
                                                                              online = self.online)

        self.shadow_models, _ = ShadowModelHandler().get_shadow_models(self.shadow_model_indices)

        self.out_indices = ~ShadowModelHandler().get_in_indices_mask(self.shadow_model_indices, self.audit_dataset["data"]).T

        true_labels = self.handler.get_labels(self.audit_dataset["data"])
        self.target_logits = ShadowModelHandler().load_logits(name="target")
        self.shadow_models_logits = []
        for indx in self.shadow_model_indices:
            self.shadow_models_logits.append(ShadowModelHandler().load_logits(indx=indx))

        self.shadow_models_logits = np.array([self.rescale_logits(x, true_labels) for x in self.shadow_models_logits])
        self.target_logits = self.rescale_logits(self.target_logits, true_labels)

    def run_attack(self:Self) -> MIAResult:
        """Runs the attack on the target model and dataset and assess privacy risks or data leakage.

        This method evaluates how the target model's output (logits) for a specific dataset
        compares to the output of shadow models to determine if the dataset was part of the
        model's training data or not.

        Returns
        -------
        Result(s) of the metric. An object containing the metric results, including predictions,
        true labels, and signal values.

        """
        # Convert shape (M, N) -> (N, M) and out_mask to in_mask
        shadow_models_logits = self.shadow_models_logits.T
        in_indices = (~self.out_indices).T
        
        # Decides which score calculation method should be used
        if(self.vectorized):
            score = lira_vectorized(shadow_models_logits, in_indices,
                                   self.target_logits, self.var_calculation,
                                   self.online, self.fix_var_threshold)
        else:
            score = lira_iterative(shadow_models_logits, in_indices,
                                   self.target_logits, self.var_calculation,
                                   self.online, self.fix_var_threshold)

        # Split the score array into two parts based on membership: in (training) and out (non-training)
        in_members = self.audit_dataset["in_members"]
        out_members = self.audit_dataset["out_members"]
        self.in_member_signals = score[in_members].reshape(-1,1)  # Scores for known training data members
        self.out_member_signals = score[out_members].reshape(-1,1)  # Scores for non-training data members

        # Prepare true labels array, marking 1 for training data and 0 for non-training data
        true_membership = np.concatenate(
            [np.ones(len(self.in_member_signals)), np.zeros(len(self.out_member_signals))]
        )

        # Combine all signal values for further analysis
        signal_values = np.concatenate([self.in_member_signals, self.out_member_signals])

        # Return a result object containing predictions, true labels, and the signal values for further evaluation
        return MIAResult.from_full_scores(true_membership=true_membership,
                                    signal_values=signal_values,
                                    result_name="LiRA",
                                    metadata=self.configs.model_dump())

def validate_inputs(shadow_models_logits: np.ndarray, target_logits: np.ndarray, shadow_inmask: np.ndarray):
    if(np.all(shadow_inmask) or not np.any(shadow_inmask)):
        raise ValueError("shadow_inmask cannot be purely True or False.")
    if shadow_inmask.shape != shadow_models_logits.shape:
        raise ValueError("shadow_inmask must have the same shape as shadow_models_logits")
    if shadow_models_logits.ndim != 2:
        raise ValueError("shadow_models_logits must be a 2D array (n_samples, n_shadow_models)")
    if target_logits.ndim != 1:
        raise ValueError("target_logits must be a 1D array")
    if len(target_logits) != shadow_models_logits.shape[0]:
        raise ValueError( f"target_logits length ({target_logits.shape[0]}) must match " 
                         f"the number of audit samples ({shadow_models_logits.shape[0]})")

def fixed_std(shadow_models_logits, shadow_inmask, online):
    """Compute per-sample IN and OUT standard deviations using fixed variance mode."""

    assert np.sum(~shadow_inmask) > 30, "Too few OUT logits to compute fixed std"
    if online:
        assert np.sum(shadow_inmask) > 30, "Too few IN logits to compute fixed std"

    out_std = np.nanstd(np.where(~shadow_inmask, shadow_models_logits, np.nan))
    in_std  = np.nanstd(np.where(shadow_inmask,  shadow_models_logits, np.nan)) if online else np.nan
    return np.array([in_std]), np.array([out_std])

#-------------------------------
# Vectorized LiRA implementation
#-------------------------------

def lira_vectorized(shadow_models_logits: np.ndarray, shadow_inmask: np.ndarray, 
                    target_logits: np.ndarray, var_calculation: str = "carlini",
                    online: bool = False, fix_var_threshold: int = 32) -> np.ndarray:
    """
    Compute LiRA membership inference scores in a fully vectorized manner.

    This function evaluates the likelihood that each sample in the target dataset
    was part of the training set by comparing its logits with shadow model logits.
    The computation is fully vectorized for efficiency.

    Parameters
    ----------
    shadow_models_logits : np.ndarray
        Array of shape (N, M), N = n_samples, M = num_shadow_models, containing logits from shadow models.
    shadow_inmask: np.ndarray
        Boolean array of shape (N, M) indicating which shadow models
        correspond to IN (training) samples.
    target_logits : np.ndarray
        Array of shape (N,) containing logits from the target model.
    var_calculation : str, default "carlini"
        Method to calculate variance. Options: 'fixed', 'carlini', 'individual_carlini'.
    online : bool, default False
        Whether to compute online LiRA scores (uses in-sample logpdf) or offline scores.
    fix_var_threshold : int, default 32
        Threshold for the number of shadow models required to compute sample-specific variance.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples,) containing the LiRA scores for each audit sample.
    """
    validate_inputs(shadow_models_logits, target_logits, shadow_inmask)

    n_samples, num_shadow_models = shadow_models_logits.shape

    # Computes the fixed in and out variances
    fixed_in_std, fixed_out_std = fixed_std(shadow_models_logits.flatten(), shadow_inmask.flatten(), online)

    # Vectorized mean calculation, if a slice is empty it will produce an
    # runtime warning, but the calculation will still produce the correct result
    out_means = np.nanmean(np.where(~shadow_inmask, shadow_models_logits, np.nan), axis=1)
    in_means = np.nanmean(np.where(shadow_inmask, shadow_models_logits, np.nan), axis=1)

    # Cast to lowercase
    var_calc = var_calculation.lower()

    # Base std case (fixed)
    out_stds = np.full(n_samples, fixed_out_std[0])
    in_stds = np.full(n_samples, fixed_in_std[0])

    if(var_calc == "fixed"):
        pass # Base case already handled

    elif(var_calc == "carlini"):
        if(num_shadow_models >= fix_var_threshold*2):
            out_stds = np.nanstd(np.where(~shadow_inmask, shadow_models_logits, np.nan), axis=1)
            in_stds  = np.nanstd(np.where(shadow_inmask,  shadow_models_logits, np.nan), axis=1)

    elif(var_calc == "individual_carlini"):
        # Count contributing shadow models per sample
        out_counts = np.sum(~shadow_inmask, axis=1)
        in_counts  = np.sum(shadow_inmask, axis=1)

        # Compute std only where enough shadow models exist, otherwise fallback to fixed
        in_mask  = in_counts  >= fix_var_threshold
        out_mask = out_counts >= fix_var_threshold

        # Use global fixed std as default, overwrite only where enough shadow models
        in_stds[in_mask]  = np.nanstd(np.where(shadow_inmask[in_mask, :], shadow_models_logits[in_mask, :], np.nan), axis=1)
        out_stds[out_mask] = np.nanstd(np.where(~shadow_inmask[out_mask, :], shadow_models_logits[out_mask, :], np.nan), axis=1)

    else:
        raise ValueError(f"Unknown var_calculation: {var_calculation!r}")

    # Vectorized logpdf
    pr_out = norm.logpdf(target_logits, out_means, out_stds + 1e-30) if online else -norm.logcdf(target_logits, out_means, out_stds + 1e-30)
    pr_in  = norm.logpdf(target_logits, in_means, in_stds + 1e-30) if online else np.zeros(n_samples)

    # Final LiRA score per audit sample
    scores = pr_in - pr_out
    
    # Debug helper
    if np.any(np.isnan(scores)):
        nan_idx = np.where(np.isnan(scores))[0]
        raise ValueError(f"NaN in vectorized scores at indices {nan_idx.tolist()}")

    return scores

#-------------------------------
# Iterative LiRA implementation
#-------------------------------

def lira_iterative(shadow_models_logits: np.ndarray, shadow_inmask: np.ndarray,
                   target_logits: np.ndarray, var_calculation: str = "carlini",
                   online: bool = False, fix_var_threshold: int = 32) -> np.ndarray:

    """
    Compute LiRA membership inference scores using an iterative approach.

    This function evaluates the likelihood that each sample in the target dataset
    was part of the training set by comparing its logits with shadow model logits.
    Unlike the vectorized version, this implementation iterates over each sample,
    computing per-sample means and variances.
    Parameters
    ----------
    shadow_models_logits : np.ndarray
        Array of shape (N, M), N = n_samples, M = num_shadow_models, containing logits from shadow models.
    shadow_inmask: np.ndarray
        Boolean array of shape (N, M) indicating which shadow models
        correspond to IN (training) samples.
    target_logits : np.ndarray
        Array of shape (N,) containing logits from the target model.
    var_calculation : str, default "carlini"
        Method to calculate variance. Options: 'fixed', 'carlini', 'individual_carlini'.
    online : bool, default False
        Whether to compute online LiRA scores (uses in-sample logpdf) or offline scores.
    fix_var_threshold : int, default 32
        Threshold for the number of shadow models required to compute sample-specific variance.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples,) containing the LiRA scores for each audit sample.
    """
    out_indices = ~shadow_inmask

    validate_inputs(shadow_models_logits, target_logits, shadow_inmask)

    n_samples, num_shadow_models = shadow_models_logits.shape
    score = np.zeros(n_samples)  # List to hold the computed probability scores for each sample

    # Computes the fixed in and out variances
    fixed_in_std, fixed_out_std = fixed_std(shadow_models_logits.flatten(), shadow_inmask.flatten(), online)

    # Iterate over and extract logits for IN and OUT shadow models for each audit sample
    for i in tqdm(range(n_samples), total=n_samples, desc="Processing audit samples"):

        # Calculate the mean for OUT shadow model logits
        out_mask = out_indices[i,:]
        sm_logits = shadow_models_logits[i,:]

        out_mean = np.mean(sm_logits[out_mask])
        out_std = get_std(sm_logits, out_mask, False, num_shadow_models,
                          var_calculation, fixed_in_std, fixed_out_std, fix_var_threshold)

        # Get the logit from the target model for the current sample
        target_logit = target_logits[i]

        # Calculate the log probability density function value
        if online:
            in_mean = np.mean(sm_logits[~out_mask])
            in_std = get_std(sm_logits, ~out_mask, True, num_shadow_models,
                             var_calculation, fixed_in_std, fixed_out_std, fix_var_threshold)

            pr_in = norm.logpdf(target_logit, in_mean, in_std + 1e-30)
            pr_out = norm.logpdf(target_logit, out_mean, out_std + 1e-30)
        else:
            pr_in = 0
            pr_out = -norm.logcdf(target_logit, out_mean, out_std + 1e-30)

        score[i] = (pr_in - pr_out)  # Append the calculated probability density value to the score list
        if np.isnan(score[i]):
            raise ValueError("Score is NaN")

    return score

def get_std(logits: np.ndarray, mask: np.ndarray, is_in: bool,
             num_shadow_models, var_calculation: str, fixed_in_std,
             fixed_out_std, fix_var_threshold) -> np.ndarray:
    """A function to define what method to use for calculating variance for LiRA."""

    # Cast to lowercase
    var_calc = var_calculation.lower()

    # Fixed/Global variance calculation.
    if var_calc == "fixed":
        # We flip the mask as this specific function takes the shadow_inmask
        return fixed_std(logits, ~mask, is_in)

    # Variance calculation as in the paper ( Membership Inference Attacks From First Principles )
    elif var_calc == "carlini":
        return carlini_variance(logits, mask, is_in, num_shadow_models,
                                fixed_in_std, fixed_out_std, fix_var_threshold)

    # Variance calculation as in the paper ( Membership Inference Attacks From First Principles )
    #   but check IN and OUT samples individualy
    elif var_calc == "individual_carlini":
        return individual_carlini(logits, mask, is_in, fixed_in_std,
                                  fixed_out_std, fix_var_threshold)

    return np.array([None])

def carlini_variance(logits: np.ndarray, mask: np.ndarray, is_in: bool,
                      num_shadow_models, fixed_in_std,
                      fixed_out_std, fix_var_threshold) -> np.ndarray:
    if num_shadow_models >= fix_var_threshold*2:
            return np.std(logits[mask])
    return fixed_in_std if is_in else fixed_out_std

def individual_carlini(logits: np.ndarray, mask: np.ndarray, is_in: bool,
                        fixed_in_std, fixed_out_std, fix_var_threshold) -> np.ndarray:
    if np.count_nonzero(mask) >= fix_var_threshold:
        return np.std(logits[mask])
    return fixed_in_std if is_in else fixed_out_std
