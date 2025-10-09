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
    n_audit_samples = shadow_models_logits.shape[0]
    score = np.zeros(n_audit_samples)  # List to hold the computed probability scores for each sample

    # Iterate over and extract logits for IN and OUT shadow models for each audit sample
    for i, (shadow_models_logits, mask) in tqdm(enumerate(zip(shadow_models_logits, in_indices_masks)),
                                                total=len(shadow_models_logits),
                                                desc="Processing audit samples"):

        # Calculate the mean for OUT shadow model logits
        out_mean = np.mean(shadow_models_logits[~mask])
        out_std = get_std(shadow_models_logits, ~mask, False, var_calculation)

        # Get the logit from the target model for the current sample
        target_logit = target_logits[i]

        # Calculate the log probability density function value
        pr_out = norm.logpdf(target_logit, out_mean, out_std + 1e-30)

        if online:
            in_mean = np.mean(shadow_models_logits[mask])
            in_std = get_std(shadow_models_logits, mask, True, var_calculation)

            pr_in = norm.logpdf(target_logit, in_mean, in_std + 1e-30)
        else:
            pr_in = 0

        score[i] = (pr_in - pr_out)  # Append the calculated probability density value to the score list
        if np.isnan(score[i]):
            raise ValueError("Score is NaN")
    return score


def get_std(logits: list, mask: list, is_in: bool, var_calculation: str) -> np.ndarray:
    """A function to define what method to use for calculating variance for LiRA."""

    # Fixed/Global variance calculation.
    if var_calculation == "fixed":
        return fixed_variance(logits, mask, is_in)

    # Variance calculation as in the paper ( Membership Inference Attacks From First Principles )
    if var_calculation == "carlini":
        return _carlini_variance(logits, mask, is_in)

    # Variance calculation as in the paper ( Membership Inference Attacks From First Principles )
    #   but check IN and OUT samples individualy
    if var_calculation == "individual_carlini":
        return _individual_carlini(logits, mask, is_in)

    return np.array([None])

def fixed_variance(logits: list, mask: list, is_in: bool) -> np.ndarray:
    if is_in and not self.online:
        return np.array([None])
    return np.std(logits[mask])

def _carlini_variance(logits: list, mask: list, is_in: bool) -> np.ndarray:
    if self.num_shadow_models >= self.fix_var_threshold*2:
            return np.std(logits[mask])
    if is_in:
        return self.fixed_in_std
    return self.fixed_out_std

def _individual_carlini(logits: list, mask: list, is_in: bool) -> np.ndarray:
    if np.count_nonzero(mask) >= self.fix_var_threshold:
        return np.std(logits[mask])
    if is_in:
        return self.fixed_in_std
    return self.fixed_out_std
