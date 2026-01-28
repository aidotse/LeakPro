import numpy as np
import matplotlib.pyplot as plt

def undo_permutations_and_concatenate(arrays, permutations):
    """
    Undo the row-wise permutations for each array and concatenate them along axis=1.

    Parameters:
    - arrays: list of 2D numpy arrays of shape (no_points, no_models_i)
    - permutations: list of 1D numpy arrays of shape (no_points,) representing the permutation indices

    Returns:
    - concatenated_array: 2D numpy array of shape (no_points, sum(no_models_i))
    """
    unpermuted_arrays = []
    for array, perm in zip(arrays, permutations):
        # Create an inverse permutation
        #print("permutation", perm)
        inverse_perm = np.argsort(perm)
        #print("inverseperm", inverse_perm)
        # Undo the permutation
        unpermuted_array = array[inverse_perm]
        unpermuted_arrays.append(unpermuted_array)

    # Concatenate along axis=1
    concatenated_array = np.concatenate(unpermuted_arrays, axis=1)
    return concatenated_array

def plot_results(results, labels = None, smarts = None):
    fig, axs = plt.subplots(1,2,figsize=(12,4))
    for ax, flag in zip(axs,[True,False]):
        for i, (fpr, tpr) in enumerate(results):
            label = None if labels is None else labels[i]
            smart = "-" if smarts is None else smarts[i]                
            if flag:
                ax.plot(fpr, tpr, smart, label = label)
            else:
                ax.loglog(fpr, tpr, smart, label = label)
    
        ax.plot(fpr, fpr, 'r--', label="random")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC Curve")
        ax.legend()
    
    plt.show()

from glob import glob
import os
import hashlib

def load_results(glob_prompt = "robust/output/signals", signal_name = "rescaled_logits.npy", verbose = False):

    if isinstance(glob_prompt, str):
        glob_prompts = [glob_prompt]
    elif isinstance(glob_prompt, list):
        glob_prompts = glob_prompt
    else:
        raise ValueError("glob_prompt must be a string or a list of strings.")
    
    audit_data_indices_list = []
    shadow_models_logits_list = [] 
    in_indices_masks_list = []
    number_shadow_model_list = []


    all_dirs = []
    for prompt in glob_prompts:
        matched_dirs = glob(prompt)
        all_dirs.extend(matched_dirs)

    for dirpath in sorted(all_dirs, key=os.path.getmtime):
        if verbose: print("signal directory", dirpath)
        audit_data_indices = np.load(dirpath+"/audit_data_indices.npy")
        #print("audit_data_indices",audit_data_indices.shape)
        audit_data_indices_list.append(audit_data_indices)
        shadow_models_logits = np.load(dirpath+"/"+signal_name)
        #print("shadow_models_logits",shadow_models_logits.shape)
        shadow_models_logits_list.append(shadow_models_logits)
        in_indices_masks = np.load(dirpath+"/in_indices_masks.npy")
        #print("in_indices_masks",in_indices_masks.shape)
        in_indices_masks_list.append(in_indices_masks)
        if verbose:
            number_shadow_model_list.append(in_indices_masks.shape[1])
            print(hashlib.sha256(audit_data_indices.tobytes()).hexdigest()[:16], \
                hashlib.sha256(in_indices_masks.tobytes()).hexdigest()[:16], \
                in_indices_masks.shape)
            print()

    total_no_runs = len(audit_data_indices_list)
    unique_no_runs = len(set([tuple(item) for item in audit_data_indices_list]))
    if unique_no_runs != total_no_runs:
        print(f"NOTE: audit sets are not unique: {unique_no_runs} < {total_no_runs}")

    shadow_models_logits = undo_permutations_and_concatenate(shadow_models_logits_list, audit_data_indices_list)
    in_indices_masks = undo_permutations_and_concatenate(in_indices_masks_list, audit_data_indices_list)

    total_no_masks = in_indices_masks.shape[1]
    no_unique_masks = len(set([tuple(in_indices_masks[:, i]) for i in range(total_no_masks)]))
    if total_no_masks != no_unique_masks:
        print(f"ERROR: masks are not unique: {no_unique_masks} < {total_no_masks}")
    else:
        print(f"NOTE: all masks are unique as expected: {total_no_masks} total")

    return shadow_models_logits, in_indices_masks


import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
#import warnings
import matplotlib.pyplot as plt

def integrate_over_range(x, y, x0, x1):
    """
    Integrates y(x) over the range [x0, x1] using trapezoid rule.
    Interpolates or extrapolates y0 and y1 using linear interpolation with flat extrapolation.
    
    Parameters:
        x (array-like): Input x values.
        y (array-like): Input y values.
        x0 (float): Lower bound of integration.
        x1 (float): Upper bound of integration.
    
    Returns:
        float: Estimated integral of y(x) over [x0, x1].
    """
    # Convert inputs to numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # Check for NaNs
    assert not np.isnan(x).any(), "x contains NaN values"
    assert not np.isnan(y).any(), "y contains NaN values"

    # Warn if extrapolation is necessary
    if x1 > np.max(x):
        msg = "x1 is greater than the maximum value in x. Extrapolation will be used."
        print("Note that", msg) # warnings.warn(msg)
    if x0 < np.min(x):
        msg = "x0 is smaller than the minimum value in x. Extrapolation will be used."
        print("Note that", msg) # warnings.warn(msg)
    
    # Create interpolator with flat extrapolation
    interpolator = interp1d(x, y, kind='linear', bounds_error=False, fill_value=(y[0], y[-1]))

    # Interpolate/extrapolate y0 and y1
    y0 = interpolator(x0)
    y1 = interpolator(x1)

    # Mask x and y to exclude values outside (x0, x1)
    mask = (x > x0) & (x < x1)
    x_spliced = x[mask]
    y_spliced = y[mask]

    # Prepend x0/y0 and append x1/y1
    x_amended = np.concatenate(([x0], x_spliced, [x1]))
    y_amended = np.concatenate(([y0], y_spliced, [y1]))

    # Compute integral using trapezoid rule
    integral = trapezoid(y_amended, x_amended)

    return integral

#x = np.linspace(-1, 0.99, 100)
#y = np.exp(-x**2)
#result = integrate_over_range(x, y, 0, 1)
#print("Integral over [0, 1]:", result)

def privacy_tail_AUC_ratio(fpr, tpr, tail = 0.001):
    # Sort fpr and tpr together
    sorted_indices = np.argsort(fpr)
    x = fpr[sorted_indices]
    y = tpr[sorted_indices]

    # Remove left duplicates
    filter_indices = np.append(x[:-1] != x[1:], True)
    x = x[filter_indices]
    y = y[filter_indices]

    # Calculate tail areas and ratio
    tail_AUC_target = integrate_over_range(x, y, 0.0, tail)
    tail_AUC_random = 0.5*tail**2
    return (tail_AUC_target/tail_AUC_random).item()

# privacy_tail_AUC_ratio(fpr, tpr)

def average_privacy_shortfall_ratio(fpr, tpr, tail = 0.001):
    # Sort fpr and tpr together
    sorted_indices = np.argsort(fpr)
    x = fpr[sorted_indices]
    y = tpr[sorted_indices]

    # Remove left duplicates
    filter_indices = np.append(x[:-1] != x[1:], True)
    x = x[filter_indices]
    y = y[filter_indices]

    # Remove zero FPR. Limit to tail
    filter_indices = (x > 0.0) & (x <= tail)
    x = x[filter_indices]
    y = y[filter_indices]

    privacy_risk_ratio = y/x
    return privacy_risk_ratio.mean().item()

# average_privacy_shortfall_ratio(fpr, tpr)

from sklearn.linear_model import LinearRegression

def regress_privacy_shortfall_ratio(fpr, tpr, tail = 0.001, plot = False, verbose = False):
    # Sort fpr and tpr together
    sorted_indices = np.argsort(fpr)
    x = fpr[sorted_indices]
    y = tpr[sorted_indices]

    # Remove left duplicates
    filter_indices = np.append(x[:-1] != x[1:], True)
    x = x[filter_indices]
    y = y[filter_indices]

    # Remove zero FPR. Limit to tail
    filter_indices = (x > 0.0) & (x <= tail)
    x = x[filter_indices]
    y = y[filter_indices]

    # Log-log transformation
    mask = (x > 0.0) & (y > 0.0) & (x <= 0.001) 
    log_x = np.log10(x[mask]).reshape(-1, 1)
    log_y = np.log10(y[mask])
    
    # Linear regression in log-log space
    model = LinearRegression()
    model.fit(log_x, log_y)

    if plot:
        # Predict log_y and convert back to original scale
        log_y_pred = model.predict(log_x)
        y_pred = 10 ** log_y_pred
        
        # Plot original data and fitted line
        plt.figure(figsize=(8, 6))
        plt.loglog(fpr, tpr, 'bo', label='Original Data')
        plt.loglog(10**log_x, y_pred, 'r-', label='Fitted Line')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Log-Log Linear Regression')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.show()        

    beta, alpha = model.coef_, 10**model.intercept_

    if verbose:
        print(f"alpha = {alpha}, beta = {beta}")
    
    return (alpha/beta*tail**(beta-1)).item()

# regress_privacy_shortfall_ratio(fpr, tpr, plot=True)

#from vectorized_mia import vectorized_lira_score
from leakpro.attacks.mia_attacks.lira import lira_vectorized as vectorized_lira_score
from sklearn.metrics import roc_curve

def hold_one_out_sampling(shadow_logits, shadow_inmask, target_logits = None, target_inmask = None):
    noneflag = False 
    if target_logits is None:
        assert target_inmask is None
        noneflag = True
    
    i_range = np.arange(shadow_logits.shape[1])
    results = []
    for i in i_range:
        ii = np.setdiff1d(i_range,i)
        if noneflag:
            target_logits, target_inmask = shadow_logits[:,i], shadow_inmask[:,i]
        score = vectorized_lira_score(shadow_logits[:,ii], shadow_inmask[:,ii], target_logits, target_inmask)
        mask = ~np.isnan(score)
        #if not all(mask):
        #    print("number of nan scores:", np.isnan(score).sum())
        fpr, tpr, thresholds =  roc_curve(target_inmask[mask], score[mask])        
        results.append((fpr, tpr))
    return results


def bootstrap_sampling(K, M, shadow_logits, shadow_inmask, target_logits = None, target_inmask = None, replace = True, vec_mia_fun = vectorized_lira_score):
    noneflag = False 
    if target_logits is None:
        assert target_inmask is None
        noneflag = True
    elif target_logits.ndim == 1:
        assert target_inmask.ndim == 1
        target_logits = target_logits.reshape([-1,1])
        target_inmask = target_inmask.reshape([-1,1])
    else:
        assert target_logits.ndim == 2
        assert target_inmask.ndim == 2
    
    no_models = shadow_logits.shape[1] 
    ii_models = np.arange(no_models)
    results = []
    for m in range(M):
        if noneflag:
            i_target = np.random.randint(no_models)
            target_logits, target_inmask = shadow_logits[:,[i_target]], shadow_inmask[:,[i_target]]
            ii_remain = np.setdiff1d(ii_models,i_target)
        else:
            ii_remain = ii_models
        ii_sample = np.random.choice(ii_remain, K, replace)
        j = np.random.randint(target_logits.shape[1])
        #print(j, target_logits.shape, target_inmask.shape)
        score = vec_mia_fun(target_logits[:,j], shadow_logits[:,ii_sample], shadow_inmask[:,ii_sample])
        mask = ~np.isnan(score)
        #if not all(mask):
        #    print("number of nan scores:", np.isnan(score).sum())
        fpr, tpr, thresholds =  roc_curve(target_inmask[mask,j], score[mask])        
        results.append((fpr, tpr))
    return results

def interpolate_unique(fpr0, fpr, tpr, extrapolate=np.nan):
    # Sort fpr and tpr together
    sorted_indices = np.argsort(fpr)
    x = fpr[sorted_indices]
    y = tpr[sorted_indices]

    # Remove left duplicates
    filter_indices = np.append(x[:-1] != x[1:], True)
    x = x[filter_indices]
    y = y[filter_indices]

    return np.interp(fpr0, x, y, left=extrapolate, right=extrapolate)

def confidence_bands(tprs, conf=95):
    return [
        np.percentile(np.vstack(tprs), conf/2+50, axis=0),
        np.mean(np.vstack(tprs), axis=0),
        np.percentile(np.vstack(tprs), 50-conf/2, axis=0)
    ]

def interp_conf(results, conf=95):
    fpr0 = sorted(np.unique(np.concat([fpr for (fpr, tpr) in results])))
    test = [interpolate_unique(fpr0, fpr, tpr) for (fpr, tpr) in results]
    conf = [(fpr0,tpr) for tpr in confidence_bands(test, conf)]
    return conf

from scipy.stats import norm

def vectorized_lira_score(target_logits, shadow_logits, shadow_inmask, fix_var_th = 30, ddof = 1):
    no_points, no_models = shadow_logits.shape
    assert shadow_inmask.shape == (no_points, no_models), f"ERROR: shadow_inmask shape ({shadow_inmask.shape}) != shadow_logits shape ({no_points, no_models})"
    assert target_logits.shape == (no_points,), f"ERROR: target_logits shape mismatch with shadow models {target_logits.shape} != ({no_points},)"
    #assert target_inmask.shape == (no_points,), f"ERROR: target_inmask shape mismatch with shadow models {target_inmask.shape} != ({no_points},)"
    
    # prepare for vectorized operations    
    shadow_logits_in = shadow_logits.copy()
    shadow_logits_in[~shadow_inmask] = np.nan
    #print(shadow_logits_in[shadow_inmask], shadow_logits_in[~shadow_inmask])
    shadow_logits_out = shadow_logits.copy()
    shadow_logits_out[shadow_inmask] = np.nan
    #print(shadow_logits_out[shadow_inmask], shadow_logits_out[~shadow_inmask])

    # calculate global variances
    in_counts = shadow_inmask.sum(axis=1)
    assert in_counts.sum() > 30, "ERROR: too few valid data points to calculate global IN variance"
    prior_var_in = np.nanvar(shadow_logits_in, ddof=ddof)
    out_counts = no_models - in_counts
    assert out_counts.sum() > 30, "ERROR: too few valid data points to calculate global OUT variance"
    prior_var_out = np.nanvar(shadow_logits_out, ddof=ddof)

    # implement variance calculation according to Carlini -- ddof should also matter here
    var_in = prior_var_in * np.ones_like(target_logits)
    if any(in_counts > fix_var_th):
        var_in[in_counts > fix_var_th] = np.nanvar(shadow_logits_in[in_counts > fix_var_th,:], axis=1, ddof=ddof)
    var_out = prior_var_out * np.ones_like(target_logits)
    if any(out_counts > fix_var_th):
        var_out[out_counts > fix_var_th] = np.nanvar(shadow_logits_out[out_counts > fix_var_th,:], axis=1, ddof=ddof)

    # implement mean calculation - must not be empty slice
    mean_in = np.nan * np.ones_like(target_logits)
    if any(in_counts > 0):
        mean_in[in_counts > 0] = np.nanmean(shadow_logits_in[in_counts > 0,:], axis=1)
    mean_out = np.nan * np.ones_like(target_logits)
    if any(out_counts > 0):
        mean_out[out_counts > 0] = np.nanmean(shadow_logits_out[out_counts > 0,:], axis=1)
    
    # implement LIRA score calculation
    logpr_in = norm.logpdf(target_logits, mean_in, np.sqrt(var_in) + 1e-30)
    logpr_out = norm.logpdf(target_logits, mean_out, np.sqrt(var_out) + 1e-30)
    score = logpr_in - logpr_out
    
    return score

    
