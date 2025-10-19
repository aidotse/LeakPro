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
    Compute LiRA (Likelihood Ratio Attack) scores iteratively for each sample.

    Parameters
    ----------
    shadow_models_logits : np.ndarray
        Array of shape (N, M), logits from shadow models for each audit sample.
    target_logits : np.ndarray
        Array of shape (N,), logits from the target model.
    in_indices_masks : np.ndarray
        Boolean array of shape (N, M), True where the shadow model saw the sample (IN).
    var_calculation : str
        Variance calculation method: "fixed", "carlini", or "individual_carlini".
    online : bool
        Whether to include IN likelihoods when computing scores.
    fix_var_threshold : int, optional
        Minimum number of shadow models to compute local variance (default: 32).

    Returns
    -------
    np.ndarray
        Array of shape (N,) containing per-sample LiRA scores.
    """

    if shadow_models_logits.ndim != 2:
        raise ValueError("shadow_models_logits must be a 2D array (n_samples, n_shadow_models)")
    if in_indices_masks.shape != shadow_models_logits.shape:
        raise ValueError("in_indices_masks must have the same shape as shadow_models_logits")
    if target_logits.ndim != 1 or target_logits.shape[0] != shadow_models_logits.shape[0]:
        raise ValueError("target_logits must be a 1D array with length equal to number of samples")

    n_audit_samples, num_shadow_models = shadow_models_logits.shape
    score = np.zeros(n_audit_samples)  # List to hold the computed probability scores for each sample

    fixed_in_std, fixed_out_std = compute_fixed_stds(shadow_models_logits, in_indices_masks)

    # Iterate over and extract logits for IN and OUT shadow models for each audit sample
    for i, (shadow_models_logits, mask) in tqdm(enumerate(zip(shadow_models_logits, in_indices_masks)),
                                                total=len(shadow_models_logits),
                                                desc="Processing audit samples"):

        # Calculate the mean for OUT shadow model logits
        out_mean = np.mean(shadow_models_logits[~mask])
        out_std = get_std(shadow_models_logits, ~mask, False,
                          var_calculation, online, fix_var_threshold,
                          num_shadow_models, fixed_in_std, fixed_out_std)

        # Get the logit from the target model for the current sample
        target_logit = target_logits[i]

        # Calculate the log probability density function value
        pr_out = norm.logpdf(target_logit, out_mean, out_std + 1e-30)

        if online:
            in_mean = np.mean(shadow_models_logits[mask])
            in_std = get_std(shadow_models_logits, mask, True,
                             var_calculation, online, fix_var_threshold,
                             num_shadow_models, fixed_in_std, fixed_out_std)

            pr_in = norm.logpdf(target_logit, in_mean, in_std + 1e-30)
        else:
            pr_in = 0

        score[i] = (pr_in - pr_out)  # Append the calculated probability density value to the score list
        if np.isnan(score[i]):
            raise ValueError("Score is NaN")
    return score


def get_std(logits: np.ndarray, mask: np.ndarray, is_in: bool,
            var_calculation: str, online: bool, fix_var_threshold,
            num_shadow_models, fixed_in_std, fixed_out_std) -> np.ndarray:
    """A function to define what method to use for calculating variance for LiRA."""

    # Cast to lowercase
    var_calc = var_calculation.lower()

    # Fixed/Global variance calculation.
    if var_calc== "fixed":
        return fixed_variance(logits, mask, is_in, online)

    # Variance calculation as in the paper ( Membership Inference Attacks From First Principles )
    elif var_calc== "carlini":
        return _carlini_variance(logits, mask, is_in, fix_var_threshold,
                                 num_shadow_models, fixed_in_std, fixed_out_std)

    # Variance calculation as in the paper ( Membership Inference Attacks From First Principles )
    #   but check IN and OUT samples individualy
    elif var_calc== "individual_carlini":
        return _individual_carlini(logits, mask, is_in, fix_var_threshold,
                                   fixed_in_std, fixed_out_std)

    return np.array([None])

def fixed_variance(logits: np.ndarray, mask: np.ndarray, is_in: bool, online: bool) -> np.ndarray:
    if is_in and not online:
        return np.array([None])
    return np.std(logits[mask])

def _carlini_variance(logits: np.ndarray, mask: np.ndarray, is_in: bool,
                      fix_var_threshold, num_shadow_models, fixed_in_std,
                      fixed_out_std) -> np.ndarray:

    if num_shadow_models >= fix_var_threshold*2:
            return np.std(logits[mask])
    if is_in:
        return fixed_in_std
    return fixed_out_std

def _individual_carlini(logits: np.ndarray, mask: np.ndarray, is_in: bool,
                        fix_var_threshold, fixed_in_std, fixed_out_std) -> np.ndarray:

    if np.count_nonzero(mask) >= fix_var_threshold:
        return np.std(logits[mask])

    return fixed_in_std if is_in else fixed_out_std

def compute_fixed_stds(logits: np.ndarray, masks: np.ndarray) -> tuple[float, float]:
    """Compute global fallback IN/OUT standard deviations across all samples."""

    in_vals  = logits[masks]
    out_vals = logits[~masks]

    return np.std(in_vals), np.std(out_vals)
