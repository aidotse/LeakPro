import numpy as np

def nd(target, pred): # Normalized deviation
    numerator = np.sum(np.abs(target - pred))
    denominator = np.sum(np.abs(target))
    return numerator / denominator if denominator != 0 else np.nan

def mse(target, pred):
    return np.mean(np.square(target - pred))

def rmse(target, pred):
    return np.sqrt(mse(target, pred))

def nrmse(target, pred, variant = "mean_abs"):
    numerator = rmse(target, pred)
    denominator = {
        "min_max": np.max(target) - np.min(target),
        "mean_abs": np.mean(np.abs(target)),
        "std": np.std(target)}[variant]
    return numerator / denominator if denominator != 0 else np.nan

def rmsle(target, pred):
    return np.sqrt(np.mean(np.square(np.log1p(target) - np.log1p(pred))))

def mae(target, pred):
    return np.mean(np.abs(target - pred))

def mape(target, pred):
    mask = target != 0
    return np.mean(np.abs((target[mask] - pred[mask]) / target[mask])) * 100 if np.any(mask) else np.nan

def smape(target, pred):
    denominator = (np.abs(target) + np.abs(pred)) + 1e-30
    return np.mean(np.abs(target - pred) / denominator)

def r2(target, pred):
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan