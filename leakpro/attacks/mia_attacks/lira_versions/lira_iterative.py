import numpy as np
from scipy.stats import norm
from tqdm import tqdm

def lira_iterative(shadow_models_logits,
                      target_logits,
                      in_indices_masks,
                      var_calculation,
                      online,
                      fix_var_threshold = 32
                      ) -> np.array:  
    """
    Compute LiRA scores in an iterative manner.

    Expects:
      - self.shadow_models_logits: numpy array shape (N, M) where N = audit samples, M = shadow models logits per sample
      - self.in_indices_masks: boolean array shape (N, M), True for IN shadow models, False for OUT
      - self.target_logits: numpy array shape (N,)
      - self.online: bool (if False, pr_in is zero)
    """

    if shadow_models_logits.ndim != 2:
        raise ValueError("shadow_models_logits must be a 2D array (n_samples, n_shadow_models)")
    if in_indices_masks.shape != shadow_models_logits.shape:
        raise ValueError("in_indices_masks must have the same shape as shadow_models_logits")
    if target_logits.ndim != 1 or target_logits.shape[0] != shadow_models_logits.shape[0]:
        raise ValueError("target_logits must be a 1D array with length equal to number of samples")

    n_audit_samples, num_shadow_models = shadow_models_logits.shape
    score = np.zeros(n_audit_samples)  # List to hold the computed probability scores for each sample

    # Iterate over and extract logits for IN and OUT shadow models for each audit sample
    for i, (shadow_models_logits, mask) in tqdm(enumerate(zip(shadow_models_logits, in_indices_masks)),
                                                total=len(shadow_models_logits),
                                                desc="Processing audit samples"):

        # Calculate the mean for OUT shadow model logits
        out_mean = np.mean(shadow_models_logits[~mask])
        out_std = get_std(shadow_models_logits, ~mask, False, var_calculation, online, fix_var_threshold, num_shadow_models)

        # Get the logit from the target model for the current sample
        target_logit = target_logits[i]

        # Calculate the log probability density function value
        pr_out = norm.logpdf(target_logit, out_mean, out_std + 1e-30)

        if online:
            in_mean = np.mean(shadow_models_logits[mask])
            in_std = get_std(shadow_models_logits, mask, True, var_calculation, online, fix_var_threshold, num_shadow_models)

            pr_in = norm.logpdf(target_logit, in_mean, in_std + 1e-30)
        else:
            pr_in = 0

        score[i] = (pr_in - pr_out)  # Append the calculated probability density value to the score list
        if np.isnan(score[i]):
            raise ValueError("Score is NaN")
    return score


def get_std(logits: np.ndarray, mask: np.ndarray, is_in: bool, var_calculation: str, online: bool, fix_var_threshold, num_shadow_models) -> np.ndarray:
    """A function to define what method to use for calculating variance for LiRA."""

    # Cast to lowercase
    var_calc = var_calculation.lower()

    # Fixed/Global variance calculation.
    if var_calc== "fixed":
        return fixed_variance(logits, mask, is_in, online)

    # Variance calculation as in the paper ( Membership Inference Attacks From First Principles )
    elif var_calc== "carlini":
        return _carlini_variance(logits, mask, is_in, fix_var_threshold, num_shadow_models)

    # Variance calculation as in the paper ( Membership Inference Attacks From First Principles )
        #   but check IN and OUT samples individualy
    elif var_calc== "individual_carlini":
        return _individual_carlini(logits, mask, is_in, fix_var_threshold)

    return np.array([None])

def fixed_variance(logits: np.ndarray, mask: np.ndarray, is_in: bool, online: bool) -> np.ndarray:
    if is_in and not online:
        return np.array([None])
    return np.std(logits[mask])

def _carlini_variance(logits: np.ndarray, mask: np.ndarray, is_in: bool, fix_var_threshold, num_shadow_models) -> np.ndarray:
    fixed_in_std, fixed_out_std = compute_fixed_stds(logits, mask)

    if num_shadow_models >= fix_var_threshold*2:
            return np.std(logits[mask])
    if is_in:
        return fixed_in_std
    return fixed_out_std

def _individual_carlini(logits: np.ndarray, mask: np.ndarray, is_in: bool, fix_var_threshold) -> np.ndarray:
    fixed_in_std, fixed_out_std = compute_fixed_stds(logits, mask)

    if np.count_nonzero(mask) >= fix_var_threshold:
        return np.std(logits[mask])

    return fixed_in_std if is_in else fixed_out_std

def compute_fixed_stds(logits: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """
    Compute fallback fixed IN/OUT standard deviations for one sample.

    Parameters
    ----------
    logits : np.ndarray
        1D array of logits for all shadow models for a given sample.
    mask : np.ndarray
        Boolean mask of same length as logits, True for IN shadow models.

    Returns
    -------
    tuple[float, float]
        (fixed_in_std, fixed_out_std)
    """
    in_vals  = logits[mask]
    out_vals = logits[~mask]

    fixed_in_std  = np.std(in_vals)  if in_vals.size  > 0 else 0.0
    fixed_out_std = np.std(out_vals) if out_vals.size > 0 else 0.0

    return fixed_in_std, fixed_out_std

