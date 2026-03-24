"""Helpers for various likelihood-based losses.

Ported from the original Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np
import torch as th
from torch.nn import functional


def normal_kl(
    mean1: th.Tensor | float,
    logvar1: th.Tensor | float,
    mean2: th.Tensor | float,
    logvar2: th.Tensor | float,
) -> th.Tensor:
    """Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x: th.Tensor) -> th.Tensor:
    """Approximate the cumulative distribution function of the standard normal."""
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(
    x: th.Tensor,
    *,
    means: th.Tensor,
    log_scales: th.Tensor,
) -> th.Tensor:
    """Compute the log-likelihood of a Gaussian distribution discretizing to an image.

    Args:
    ----
        x: Target images rescaled to the range `[-1, 1]`.
        means: Gaussian mean tensor.
        log_scales: Gaussian log-standard-deviation tensor.

    Returns:
    -------
        Tensor of log probabilities with the same shape as `x`.

    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

def topk_loss(out: th.Tensor, iden: th.Tensor, k: int) -> th.Tensor:
    """Compute the top-k loss.

    Args:
    ----
        out (Tensor): The output logits from the model.
        iden (Tensor): The ground truth class indices.
        k (int): The number of top incorrect classes to consider.

    Returns:
    -------
        Tensor: The computed top-k loss.

    """
    assert out.shape[0] == iden.shape[0]
    iden = iden.unsqueeze(1)
    real = out.gather(1, iden).squeeze(1)
    if k == 0:
        return -1 * real.mean()
    tmp_out = th.scatter(out, dim=1, index=iden, src=-th.ones_like(iden) * 1000.0)
    margin = th.topk(tmp_out, k=k)[0]
    return -1 * real.mean() + margin.mean()

def p_reg_loss(feature_t: th.Tensor, classes: th.Tensor, p_reg: th.Tensor) -> th.Tensor:
    """Compute the p_reg loss.

    Args:
    ----
        feature_t (Tensor): The feature tensor.
        classes (Tensor): The class indices.
        p_reg (Tensor): The regularization tensor.

    Returns:
    -------
        Tensor: The computed p_reg loss.

    """
    fea_reg = p_reg[classes]
    return functional.mse_loss(feature_t, fea_reg)
