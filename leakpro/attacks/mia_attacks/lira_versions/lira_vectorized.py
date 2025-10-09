import numpy as np
from scipy.stats import norm

def lira_vectorized(shadow_models_logits,
                      target_logits,
                      in_indices_masks,
                      var_calculation,
                      online,
                      fix_var_threshold = 32
                      ) -> np.array:  
    """
    Compute LiRA scores in a fully vectorized manner.

    Parameters
    ----------
    shadow_models_logits : np.ndarray
        Array of shape (N, M), where N = audit samples and M = shadow models.
    target_logits : np.ndarray
        Array of shape (N,), logits of the target model for audit samples.
    in_indices_masks : np.ndarray
        Boolean mask of shape (N, M). True for IN (member) shadow models.
    var_calculation : str
        One of {"fixed", "carlini", "individual_carlini"} determining variance method.
    online : bool
        If False, PR_in is set to zero.
    fix_var_threshold : int, optional
        Threshold for fixed variance, default 32 (from
        "Membership Inference Attacks from First Principles", Carlini et al., 2022).

    Returns
    -------
    np.ndarray
        LiRA scores per audit sample.
    """
    
    if shadow_models_logits.ndim != 2:
        raise ValueError("shadow_models_logits must be a 2D array (n_samples, n_shadow_models)")
    if in_indices_masks.shape != shadow_models_logits.shape:
        raise ValueError("in_indices_masks must have the same shape as shadow_models_logits")
    if target_logits.ndim != 1 or target_logits.shape[0] != shadow_models_logits.shape[0]:
        raise ValueError("target_logits must be a 1D array with length equal to number of samples")

    n_samples, num_shadow_models = shadow_models_logits.shape

    # Computes the fixed in and out variances
    fixed_in_std, fixed_out_std = compute_fixed_stds(shadow_models_logits, in_indices_masks)

    # Vectorized mean calculation
    out_means = np.nanmean(np.where(~in_indices_masks, shadow_models_logits, np.nan), axis=1)
    in_means = np.nanmean(np.where(in_indices_masks, shadow_models_logits, np.nan), axis=1)

    # Replace NaNs in means with 0.0
    out_means = np.where(np.isnan(out_means), 0.0, out_means)
    in_means  = np.where(np.isnan(in_means),  0.0, in_means)

    # Cast to lowercase
    var_calc = var_calculation.lower()

    if(var_calc== "fixed"):
        in_stds, out_stds = vectorized_fixed_variance(in_indices_masks,
                                                      shadow_models_logits,
                                                      online)

    elif(var_calc == "carlini"):
        in_stds, out_stds = vectorized_carlini_variance(in_indices_masks,
                                                        shadow_models_logits,
                                                        fix_var_threshold,
                                                        online,
                                                        num_shadow_models,
                                                        fixed_out_std,
                                                        fixed_in_std)

    elif(var_calc == "individual_carlini"):
        in_stds, out_stds = vectorized_individual_carlini(in_indices_masks,
                                                          shadow_models_logits,
                                                          fix_var_threshold,
                                                          online,
                                                          fixed_out_std,
                                                          fixed_in_std)
    else:
        raise ValueError(f"Unknown var_calculation: {var_calculation!r}")

    # Vectorized logpdf
    pr_out = norm.logpdf(target_logits, out_means, out_stds + 1e-30)
    pr_in  = norm.logpdf(target_logits, in_means, in_stds + 1e-30) if online else np.zeros(n_samples)

    # Final LiRA score per audit sample
    scores = pr_in - pr_out
    
    # Debug helper
    if np.any(np.isnan(scores)):
        nan_idx = np.where(np.isnan(scores))[0]
        raise ValueError(f"NaN in vectorized scores at indices {nan_idx.tolist()}")

    return scores

def vectorized_fixed_variance(in_indices_masks, shadow_models_logits, online) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-sample IN and OUT standard deviations using fixed variance mode.

    Parameters
    ----------
    in_indices_masks : np.ndarray
        Boolean mask for IN samples, shape (N, M).
    shadow_models_logits : np.ndarray
        Shadow model logits, shape (N, M).
    online : bool
        If False, IN variances are set to zero.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (in_stds, out_stds) arrays of shape (N,).
    """

    out_stds = np.nanstd(np.where(~in_indices_masks, shadow_models_logits, np.nan), axis=1)
    in_stds  = np.nanstd(np.where(in_indices_masks,  shadow_models_logits, np.nan), axis=1)

    if not online:
        in_stds[:] = 0.0

    return in_stds, out_stds

def vectorized_carlini_variance(in_indices_masks,
                                shadow_models_logits,
                                fix_var_threshold,
                                online,
                                num_shadow_models,
                                fixed_out_std,
                                fixed_in_std
                                ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-sample IN/OUT standard deviations using Carlini variance method.

    Parameters
    ----------
    in_indices_masks : np.ndarray
        Boolean mask for IN samples, shape (N, M).
    shadow_models_logits : np.ndarray
        Shadow model logits, shape (N, M).
    fix_var_threshold : int
        Threshold for variance switching.
    online : bool
        If False, IN variances are set to zero.
    num_shadow_models : int
        Total number of shadow models.
    fixed_out_std, fixed_in_std : float
        Precomputed fixed variances for fallback.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (in_stds, out_stds) arrays of shape (N,).
    """

    n_samples = shadow_models_logits.shape[0]
    if num_shadow_models >= fix_var_threshold * 2:
        out_stds = np.nanstd(np.where(~in_indices_masks, shadow_models_logits, np.nan), axis=1)
        in_stds  = np.nanstd(np.where(in_indices_masks,  shadow_models_logits, np.nan), axis=1)
    else:
        out_stds = np.full(n_samples, fixed_out_std)
        in_stds  = np.full(n_samples, fixed_in_std) if online else np.zeros(n_samples)

    return in_stds, out_stds

def vectorized_individual_carlini(in_indices_masks,
                                shadow_models_logits,
                                fix_var_threshold,
                                online,
                                fixed_out_std,
                                fixed_in_std
                                ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-sample IN/OUT standard deviations using individual Carlini variance.

    Variance is computed per audit sample; samples with fewer than
    fix_var_threshold contributing shadow models use fixed variance instead.

    Parameters
    ----------
    in_indices_masks : np.ndarray
        Boolean mask for IN samples, shape (N, M).
    shadow_models_logits : np.ndarray
        Shadow model logits, shape (N, M).
    fix_var_threshold : int
        Minimum number of shadow models for local variance.
    online : bool
        If False, IN variances are set to zero.
    fixed_out_std, fixed_in_std : float
        Global fixed variance fallbacks.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (in_stds, out_stds) arrays of shape (N,).
    """

    out_stds = np.nanstd(np.where(~in_indices_masks, shadow_models_logits, np.nan), axis=1)
    in_stds  = np.nanstd(np.where(in_indices_masks,  shadow_models_logits, np.nan), axis=1)

    out_counts = np.sum(~in_indices_masks, axis=1)
    in_counts  = np.sum(in_indices_masks,  axis=1)

    out_stds = np.where(out_counts >= fix_var_threshold, out_stds, fixed_out_std)
    in_stds  = np.where(in_counts  >= fix_var_threshold, in_stds,  fixed_in_std)

    if not online:
        in_stds[:] = 0.0

    return in_stds, out_stds

def compute_fixed_stds(shadow_models_logits, in_indices_masks):
    """
    Compute global fixed in and out standard deviations

    Returns
    -------
    tuple[float, float]
        If there are no IN or no OUT values, the corresponding std is 0.0.
    """
    in_vals  = shadow_models_logits[in_indices_masks]
    out_vals = shadow_models_logits[~in_indices_masks]
    fixed_in_std = np.std(in_vals) if in_vals.size > 0 else 0.0
    fixed_out_std = np.std(out_vals) if out_vals.size > 0 else 0.0

    return fixed_in_std, fixed_out_std
