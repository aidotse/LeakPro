"""Modular and vetorized implementation of the Robust MIA (RMIA) attack.


    To run legacy behaviour(similar to the previous code): instantiate 
    AttackRMIA with {"vectorized": False, "use_all_z": True}; 
    the scoring reduces to the same loops.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np
from pydantic import BaseModel, Field, model_validator

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.attacks.utils.utils import softmax_logits
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.signals.signal import ModelLogits
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


__all__ = ["AttackRMIA", "AttackRMIAModular", "RMIAEvaluationBundle", "vectorized_rmia_score", "rmia_roc"]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

_BUNDLE_EPS = 1e-6

def _sigmoid(x: np.ndarray) -> np.ndarray:
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    z = np.zeros_like(x, dtype=np.float64)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    out = np.empty_like(x, dtype=np.float64)
    out[pos_mask] = 1.0 / (1.0 + z[pos_mask])
    out[neg_mask] = z[neg_mask] / (1.0 + z[neg_mask])
    return out


def _make_rng(rng: Optional[np.random.Generator | int]) -> np.random.Generator:
    if isinstance(rng, (int, np.integer)) or rng is None:
        return np.random.default_rng(rng)
    return rng


def _sample_reference_indices(
    *,
    num_target: int,
    num_reference: int,
    sample_size: int,
    rng: np.random.Generator,
    cache: Dict[str, np.ndarray],
    cache_key: str,
    exclude_self: bool,
) -> np.ndarray:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    sample_size = min(sample_size, max(1, num_reference))
    cached = cache.get(cache_key)
    if cached is not None and cached.shape == (num_target, sample_size):
        return cached

    indices = np.empty((num_target, sample_size), dtype=np.int64)

    if exclude_self and num_reference == num_target:
        base = np.arange(num_reference)
        for i in range(num_target):
            pool = np.delete(base, i)
            if pool.size == 0:
                indices[i, :] = i
                continue
            replace = sample_size > pool.size
            indices[i, :] = rng.choice(pool, size=sample_size, replace=replace)
    else:
        replace = sample_size > num_reference
        indices[:] = rng.choice(num_reference, size=(num_target, sample_size), replace=replace)

    cache[cache_key] = indices
    return indices


def _apply_offline_affine(prior: np.ndarray, offline_a: Optional[float]) -> np.ndarray:
    if offline_a is None:
        return prior
    return 0.5 * ((offline_a + 1.0) * prior + (1.0 - offline_a))


def _compute_prior_from_bundle(
    bundle: "RMIAEvaluationBundle",
    *,
    use_out_only: bool,
    eps: float,
    score_scale: float,
    offline_a: Optional[float] = None,
    custom_out_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    probs = bundle.shadow_probs
    if custom_out_mask is not None:
        if custom_out_mask.shape != (bundle.num_models, bundle.num_points):
            raise ValueError("custom_out_mask shape mismatch")
        mask = custom_out_mask.T.astype(bool)
    elif use_out_only:
        mask = bundle.out_mask
    else:
        mask = np.ones_like(bundle.out_mask, dtype=bool)

    sums = (probs * mask).sum(axis=1)
    counts = mask.sum(axis=1)
    prior = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums, dtype=np.float64),
        where=counts > 0,
    )

    need_fallback = counts == 0
    if np.any(need_fallback):
        prior[need_fallback] = bundle.global_mean[need_fallback]

    prior = _apply_offline_affine(prior, offline_a)
    prior = np.clip(prior * float(score_scale), eps, 1.0 - eps)
    return prior


def vectorized_rmia_score(
    audit_bundle: "RMIAEvaluationBundle",
    *,
    reference_bundle: Optional["RMIAEvaluationBundle"] = None,
    gamma: float = 1.0,
    z_sample_size: int = 512,
    rng: Optional[np.random.Generator | int] = None,
    use_all_z: bool = False,
    batch_points: int = 1024,
    eps: float = 1e-6,
    score_scale: float = 1.0,
    offline_a: Optional[float] = None,
    exclude_self: bool = False,
) -> np.ndarray:
    rng = _make_rng(rng)

    p_theta_x = np.clip(audit_bundle.target_probs, eps, 1.0 - eps)
    prior_x = _compute_prior_from_bundle(
        audit_bundle,
        use_out_only=False,
        eps=eps,
        score_scale=score_scale,
        offline_a=offline_a,
    )
    ratio_x = p_theta_x / (prior_x + eps)

    ref_bundle = reference_bundle or audit_bundle
    p_theta_z = np.clip(ref_bundle.target_probs, eps, 1.0 - eps)
    prior_z = _compute_prior_from_bundle(
        ref_bundle,
        use_out_only=True,
        eps=eps,
        score_scale=score_scale,
        offline_a=offline_a,
    )
    ratio_z_all = p_theta_z / (prior_z + eps)

    num_points = audit_bundle.num_points
    num_reference = ref_bundle.num_points
    gamma_f = float(gamma)

    if gamma_f <= 0.0:
        # All ratios exceed a non-positive gamma, so return the trivial all-ones result without heavy work.
        return np.ones(num_points, dtype=np.float64)

    if use_all_z:
        # Sort once and binary-search each threshold; avoids materialising num_points × num_reference arrays.
        ratio_z_sorted = np.sort(ratio_z_all)
        thresholds = ratio_x / gamma_f
        counts = np.searchsorted(ratio_z_sorted, thresholds, side="right")
        return counts.astype(np.float64) / float(num_reference)

    cache_key = f"z_samples_{num_reference}_{z_sample_size}_{exclude_self}"
    z_indices = _sample_reference_indices(
        num_target=num_points,
        num_reference=num_reference,
        sample_size=z_sample_size,
        rng=rng,
        cache=audit_bundle.extra,
        cache_key=cache_key,
        exclude_self=exclude_self and (audit_bundle is ref_bundle),
    )
    ratio_z = ratio_z_all[z_indices]

    scores = np.zeros(num_points, dtype=np.float64)
    idx_start = 0
    while idx_start < num_points:
        idx_end = min(num_points, idx_start + batch_points)
        chunk_ratio_x = ratio_x[idx_start:idx_end][:, None]
        # Multiply once per batch to reuse cached samples and prevent extra temporary arrays from division.
        comp = chunk_ratio_x >= gamma_f * ratio_z[idx_start:idx_end]
        scores[idx_start:idx_end] = comp.mean(axis=1)
        idx_start = idx_end

    return scores


def rmia_roc(
    scores: np.ndarray,
    target_inmask: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the ROC curve for RMIA scores."""
    if target_inmask is None:
        raise ValueError("target_inmask is required to compute ROC")
    y = target_inmask.astype(bool)
    if thresholds is None:
        uniq = np.unique(scores)
        thresholds = np.concatenate(([-1.0], uniq, [2.0]))
    tpr, fpr = [], []
    positives = max(1, np.sum(y))
    negatives = max(1, np.sum(~y))
    for thr in thresholds:
        pred = scores >= thr
        tp = np.sum(pred & y)
        fp = np.sum(pred & (~y))
        tpr.append(tp / positives)
        fpr.append(fp / negatives)
    return np.asarray(fpr), np.asarray(tpr), thresholds


# ---------------------------------------------------------------------------
# Data bundles
# ---------------------------------------------------------------------------


@dataclass
class RMIAEvaluationBundle:
    """Container for pre-computed probabilities and masks used by RMIA.
    It ensures the shape of the probability arrays/masks correct and also clipped; 
    it precomputes out_mask and global_mean, and supports copying with cached extras."""

    target_probs: np.ndarray
    shadow_probs: np.ndarray
    shadow_inmask: np.ndarray
    target_inmask: Optional[np.ndarray] = None
    extra: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.target_probs = np.clip(
            np.asarray(self.target_probs, dtype=np.float64),
            _BUNDLE_EPS,
            1.0 - _BUNDLE_EPS,
        )
        shadow_probs = np.asarray(self.shadow_probs, dtype=np.float64)
        if shadow_probs.ndim != 2:
            raise ValueError("shadow_probs must be 2-D with shape (points, models)")
        self.shadow_probs = np.clip(shadow_probs, _BUNDLE_EPS, 1.0 - _BUNDLE_EPS)

        self.shadow_inmask = np.asarray(self.shadow_inmask, dtype=bool)
        if self.shadow_inmask.shape != self.shadow_probs.shape:
            raise ValueError("shadow_inmask shape must match shadow_probs")

        self.num_points, self.num_models = self.shadow_probs.shape
        self.out_mask = ~self.shadow_inmask
        self.global_mean = self.shadow_probs.mean(axis=1)

        if self.target_probs.shape[0] != self.num_points:
            raise ValueError("target_probs length must match number of points")

        if self.target_inmask is not None and self.target_inmask.shape != (self.num_points,):
            raise ValueError("target_inmask must align with number of points")

    def copy(self) -> "RMIAEvaluationBundle":
        return RMIAEvaluationBundle(
            target_probs=self.target_probs.copy(),
            shadow_probs=self.shadow_probs.copy(),
            shadow_inmask=self.shadow_inmask.copy(),
            target_inmask=None if self.target_inmask is None else self.target_inmask.copy(),
            extra={k: (v.copy() if hasattr(v, "copy") else v) for k, v in self.extra.items()},
        )


# ---------------------------------------------------------------------------
# Modular RMIA attack
# ---------------------------------------------------------------------------


class AttackRMIA(AbstractMIA):
    """Modular RMIA attack with logits-only entry points."""

    shadow_handler_cls = ShadowModelHandler

    class AttackConfig(BaseModel):
        """Configuration for the modular RMIA attack."""

        num_shadow_models: int = Field(default=1, ge=1, description="Number of shadow models")
        temperature: float = Field(default=2.0, ge=0.0, description="Softmax temperature")
        training_data_fraction: float = Field(
            default=0.5,
            ge=0.0,
            le=1.0,
            description="Fraction of attack data used for shadow model training",
        )
        attack_data_fraction: float = Field(
            default=0.1,
            ge=0.0,
            le=1.0,
            description="Fraction of attack data used for reference logits",
        )
        online: bool = Field(default=False, description="Toggle between online and offline RMIA")
        gamma: float = Field(
            default=2.0,
            ge=0.0,
            description="Likelihood-ratio threshold",
            json_schema_extra={"optuna": {"type": "float", "low": 0.1, "high": 10, "log": True}},
        )
        offline_a: float = Field(
            default=0.33,
            ge=0.0,
            le=1.0,
            description="Affine calibration for offline marginal p(x)",
            json_schema_extra={
                "optuna": {
                    "type": "float",
                    "low": 0.0,
                    "high": 1.0,
                    "enabled_if": lambda model: not model.online,
                }
            },
        )
        z_sample_size: int = Field(
            default=512,
            ge=1,
            description="Number of reference points drawn per audit example when sampling",
        )
        use_all_z: bool = Field(
            default=True,
            description="Use the full reference set instead of sampling z points",
        )
        score_scale: float = Field(
            default=1.0,
            description="Scaling factor applied to prior estimates before ratio computation",
        )
        vectorized: bool = Field(
            default=True,
            description="Use vectorised likelihood computations (disable to fall back to looped variant)",
        )

        @model_validator(mode="after")
        def check_num_shadow_models_if_online(self) -> Self:
            if self.online and self.num_shadow_models < 2:
                raise ValueError("When online is True, num_shadow_models must be >= 2")
            return self

    def __init__(
        self: Self,
        handler: MIAHandler,
        configs: Optional[dict] = None,
        *,
        signal: Optional[ModelLogits] = None,
    ) -> None:
        logger.info("Configuring the modular RMIA attack")
        self.configs = self.AttackConfig() if configs is None else self.AttackConfig(**configs)
        super().__init__(handler)

        for key, value in self.configs.model_dump().items():
            setattr(self, key, value)

        if not self.online and self.population_size == len(self.audit_dataset["data"]):
            raise ValueError(
                "The audit dataset exhausts the population; no points remain for shadow models."
            )

        self.signal = signal or ModelLogits()
        self.epsilon = 1e-6

        self.shadow_models: Optional[list] = None
        self.shadow_model_indices: Optional[np.ndarray] = None

        self.attack_cache_folder_path = f"leakpro_output/attack_cache/{self.attack_id}"
        os.makedirs(self.attack_cache_folder_path, exist_ok=True)

        self.load_for_optuna = False
        self._precomputed_aux_bundle: Optional[RMIAEvaluationBundle] = None
        self._precomputed_audit_bundle: Optional[RMIAEvaluationBundle] = None
        self.aux_bundle: Optional[RMIAEvaluationBundle] = None
        self.audit_bundle: Optional[RMIAEvaluationBundle] = None
        self.vectorized = self.configs.vectorized

    # ------------------------------------------------------------------
    # Descriptive helpers
    # ------------------------------------------------------------------

    def description(self: Self) -> dict:
        title_str = "RMIA attack (modular)"
        reference_str = (
            "Zarifzadeh, Sajjad, Philippe Cheng-Jie Marc Liu, and Reza Shokri. "
            "Low-Cost High-Power Membership Inference by Boosting Relativity. (2023)."
        )
        summary_str = (
            "The RMIA attack is a likelihood-ratio membership inference attack that operates on logits "
            "from target and shadow models. Supports a vectorised scorer or a legacy looped variant via config."
        )
        detailed_str = (
            "This modular implementation separates shadow model training from scoring so that pre-computed "
            "logits can be supplied without retraining models."
        )
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }

    # ------------------------------------------------------------------
    # Public API extensions
    # ------------------------------------------------------------------

    def set_precomputed_bundles(
        self: Self,
        *,
        audit_bundle: Optional[RMIAEvaluationBundle] = None,
        aux_bundle: Optional[RMIAEvaluationBundle] = None,
    ) -> None:
        """Inject pre-computed logits for the audit and reference datasets."""
        self._precomputed_audit_bundle = audit_bundle
        self._precomputed_aux_bundle = aux_bundle
        if audit_bundle is not None:
            logger.info("Audit logits bundle provided; skipping model evaluations for audit data")
        if aux_bundle is not None:
            logger.info("Auxiliary logits bundle provided; skipping shadow model evaluations for reference data")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_shadow_handler(self: Self) -> ShadowModelHandler:
        return type(self).shadow_handler_cls(self.handler)

    def _prepare_shadow_models(self: Self) -> None:
        attack_indices = self.sample_indices_from_population(
            include_train_indices=self.online,
            include_test_indices=self.online,
        )
        self.attack_data_indices = attack_indices
        handler = self._get_shadow_handler()
        self.shadow_model_indices = handler.create_shadow_models(
            num_models=self.num_shadow_models,
            shadow_population=self.attack_data_indices,
            training_fraction=self.training_data_fraction,
            online=self.online,
        )
        self.shadow_models, _ = handler.get_shadow_models(self.shadow_model_indices)

    def _bundle_from_logits(
        self: Self,
        target_logits: np.ndarray,
        shadow_logits: np.ndarray,
        labels: np.ndarray,
        *,
        target_membership: Optional[np.ndarray] = None,
        shadow_inmask: Optional[np.ndarray] = None,
        extra: Optional[Dict[str, np.ndarray]] = None,
    ) -> RMIAEvaluationBundle:
        """Convert raw logits into an :class:`RMIAEvaluationBundle`."""
        labels = labels.astype(int)
        n_points = labels.shape[0]
        target_probs = softmax_logits(target_logits, self.temperature)
        if target_probs.ndim != 3:
            raise ValueError("Expected target_logits to yield shape (num_models, num_points, num_classes)")
        target_correct = target_probs[0, np.arange(n_points), labels]

        shadow_probs = softmax_logits(shadow_logits, self.temperature)
        if shadow_probs.ndim != 3:
            raise ValueError("Expected shadow_logits to yield shape (num_models, num_points, num_classes)")
        correct_shadow = shadow_probs[:, np.arange(n_points), labels]
        shadow_correct = correct_shadow.T  # (num_points, num_models)
        if shadow_inmask is None:
            shadow_inmask = np.zeros_like(shadow_correct, dtype=bool)
        else:
            shadow_inmask = shadow_inmask.astype(bool)

        extra = {} if extra is None else extra
        return RMIAEvaluationBundle(
            target_probs=target_correct,
            shadow_probs=shadow_correct,
            shadow_inmask=shadow_inmask,
            target_inmask=None if target_membership is None else target_membership.astype(bool),
            extra=extra,
        )

    def _collect_offline_aux_bundle(self: Self) -> RMIAEvaluationBundle:
        logger.info("Collecting auxiliary logits for offline RMIA")
        if not hasattr(self, "attack_data_indices"):
            self.attack_data_indices = self.sample_indices_from_population()
        n_points = max(1, int(self.attack_data_fraction * len(self.attack_data_indices)))
        chosen_indices = np.random.choice(self.attack_data_indices, n_points, replace=False)

        labels = self.handler.get_labels(chosen_indices)
        target_logits = np.array(self.signal([self.target_model], self.handler, chosen_indices))
        shadow_logits = self.signal(self.shadow_models, self.handler, chosen_indices)

        bundle = self._bundle_from_logits(target_logits, shadow_logits, labels)
        np.save(f"{self.attack_cache_folder_path}/offline_aux_target_probs.npy", bundle.target_probs)
        np.save(f"{self.attack_cache_folder_path}/offline_aux_shadow_probs.npy", bundle.shadow_probs)
        np.save(f"{self.attack_cache_folder_path}/offline_aux_shadow_inmask.npy", bundle.shadow_inmask)
        return bundle

    def _collect_offline_audit_bundle(self: Self) -> RMIAEvaluationBundle:
        logger.info("Collecting audit logits for offline RMIA")
        audit_indices = self.audit_dataset["data"]
        labels = self.handler.get_labels(audit_indices)
        target_logits = np.array(self.signal([self.target_model], self.handler, audit_indices))
        shadow_logits = self.signal(self.shadow_models, self.handler, audit_indices)
        membership = np.zeros(len(audit_indices), dtype=bool)
        membership[self.audit_dataset["in_members"]] = True

        bundle = self._bundle_from_logits(target_logits, shadow_logits, labels, target_membership=membership)
        np.save(f"{self.attack_cache_folder_path}/offline_audit_target_probs.npy", bundle.target_probs)
        np.save(f"{self.attack_cache_folder_path}/offline_audit_shadow_probs.npy", bundle.shadow_probs)
        np.save(f"{self.attack_cache_folder_path}/offline_audit_shadow_inmask.npy", bundle.shadow_inmask)
        np.save(f"{self.attack_cache_folder_path}/offline_audit_membership.npy", membership)
        return bundle

    def _collect_online_audit_bundle(self: Self) -> RMIAEvaluationBundle:
        logger.info("Collecting audit logits for online RMIA")
        handler = self._get_shadow_handler()
        indices_mask = handler.get_in_indices_mask(self.shadow_model_indices, self.audit_dataset["data"]).T
        num_seen = np.sum(indices_mask, axis=0)
        mask = (num_seen > 0) & (num_seen < self.num_shadow_models)
        audit_indices = self.audit_dataset["data"][mask]
        if audit_indices.size == 0:
            raise ValueError("No audit points satisfy the online RMIA requirements")

        membership_mask = np.zeros(mask.sum(), dtype=bool)
        membership_mask[: np.sum(mask[self.audit_dataset["in_members"]])] = True

        labels = self.handler.get_labels(audit_indices)
        target_logits = np.array(self.signal([self.target_model], self.handler, audit_indices))
        shadow_logits = self.signal(self.shadow_models, self.handler, audit_indices)
        shadow_inmask = indices_mask[:, mask].T
        out_model_indices = (~indices_mask[:, mask]).astype(bool)

        bundle = self._bundle_from_logits(
            target_logits,
            shadow_logits,
            labels,
            target_membership=membership_mask,
            shadow_inmask=shadow_inmask,
            extra={"out_model_indices": out_model_indices},
        )

        np.save(f"{self.attack_cache_folder_path}/online_audit_target_probs.npy", bundle.target_probs)
        np.save(f"{self.attack_cache_folder_path}/online_audit_shadow_probs.npy", bundle.shadow_probs)
        np.save(f"{self.attack_cache_folder_path}/online_audit_shadow_inmask.npy", bundle.shadow_inmask)
        np.save(f"{self.attack_cache_folder_path}/online_out_model_indices.npy", out_model_indices)
        np.save(f"{self.attack_cache_folder_path}/online_audit_membership.npy", membership_mask)
        return bundle

    def _collect_online_aux_bundle(self: Self) -> RMIAEvaluationBundle:
        logger.info("Collecting auxiliary logits for online RMIA")
        aux_indices = self.sample_indices_from_population()
        n_points = max(1, int(self.attack_data_fraction * len(aux_indices)))
        chosen_indices = np.random.choice(aux_indices, n_points, replace=False)

        labels = self.handler.get_labels(chosen_indices)
        target_logits = np.array(self.signal([self.target_model], self.handler, chosen_indices))
        shadow_logits = self.signal(self.shadow_models, self.handler, chosen_indices)

        bundle = self._bundle_from_logits(target_logits, shadow_logits, labels)
        np.save(f"{self.attack_cache_folder_path}/online_aux_target_probs.npy", bundle.target_probs)
        np.save(f"{self.attack_cache_folder_path}/online_aux_shadow_probs.npy", bundle.shadow_probs)
        np.save(f"{self.attack_cache_folder_path}/online_aux_shadow_inmask.npy", bundle.shadow_inmask)
        return bundle

    def prepare_attack(self: Self) -> None:
        logger.info("Preparing RMIA attack inputs")
        if self._precomputed_audit_bundle is not None and self._precomputed_aux_bundle is not None:
            self.audit_bundle = self._precomputed_audit_bundle.copy()
            self.aux_bundle = self._precomputed_aux_bundle.copy()
            return

        if not self.load_for_optuna:
            self._prepare_shadow_models()

        if self.online:
            self.audit_bundle = self._collect_online_audit_bundle()
            self.aux_bundle = self._collect_online_aux_bundle()
        else:
            self.aux_bundle = self._collect_offline_aux_bundle()
            self.audit_bundle = self._collect_offline_audit_bundle()

    # ------------------------------------------------------------------
    # Scoring routines
    # ------------------------------------------------------------------

    def _score_offline_vectorized(self: Self) -> np.ndarray:
        if self.audit_bundle is None or self.aux_bundle is None:
            raise ValueError("Audit/auxiliary bundles must be prepared before scoring")

        return vectorized_rmia_score(
            self.audit_bundle,
            reference_bundle=self.aux_bundle,
            z_sample_size=self.z_sample_size,
            gamma=self.gamma,
            use_all_z=self.use_all_z,
            eps=self.epsilon,
            score_scale=self.score_scale,
            offline_a=self.offline_a,
            exclude_self=True,
        )

    def _score_offline_looped(self: Self) -> np.ndarray:
        audit = self.audit_bundle
        aux = self.aux_bundle
        if audit is None or aux is None:
            raise ValueError("Audit/auxiliary bundles must be prepared before scoring")

        p_x_given_target = np.clip(audit.target_probs, self.epsilon, 1.0 - self.epsilon)
        p_x_out = audit.shadow_probs.mean(axis=1)
        p_x = 0.5 * ((self.offline_a + 1.0) * p_x_out + (1.0 - self.offline_a))
        p_x = np.clip(p_x * float(self.score_scale), self.epsilon, 1.0 - self.epsilon)
        ratio_x = p_x_given_target / (p_x + self.epsilon)

        p_z_given_target = np.clip(aux.target_probs, self.epsilon, 1.0 - self.epsilon)
        p_z_out = aux.shadow_probs.mean(axis=1)
        p_z = 0.5 * ((self.offline_a + 1.0) * p_z_out + (1.0 - self.offline_a))
        p_z = np.clip(p_z * float(self.score_scale), self.epsilon, 1.0 - self.epsilon)
        ratio_z = p_z_given_target / (p_z + self.epsilon)

        likelihoods = ratio_x[:, None] / ratio_z[None, :]
        return (likelihoods >= float(self.gamma)).mean(axis=1)

    def _score_online_vectorized(self: Self) -> np.ndarray:
        if self.audit_bundle is None or self.aux_bundle is None:
            raise ValueError("Audit/auxiliary bundles must be prepared before scoring")

        audit_bundle = self.audit_bundle
        aux_bundle = self.aux_bundle

        p_theta_x = np.clip(audit_bundle.target_probs, self.epsilon, 1.0 - self.epsilon)
        prior_x = _compute_prior_from_bundle(
            audit_bundle,
            use_out_only=False,
            eps=self.epsilon,
            score_scale=self.score_scale,
        )
        ratio_x = p_theta_x / (prior_x + self.epsilon)

        p_theta_z = np.clip(aux_bundle.target_probs, self.epsilon, 1.0 - self.epsilon)
        shadow_probs_aux = aux_bundle.shadow_probs  # (num_aux, num_models)

        out_model_indices = audit_bundle.extra.get("out_model_indices")
        if out_model_indices is None:
            raise ValueError("Online bundle missing out_model_indices mask")

        mask = out_model_indices.astype(bool)
        counts = mask.sum(axis=0, dtype=np.int64)
        counts_safe = np.clip(counts, 1, None).astype(np.float64).reshape(1, -1)
        weights = np.divide(
            mask.astype(np.float64),
            counts_safe,
            out=np.zeros_like(mask, dtype=np.float64),
            where=counts_safe > 0,
        )
        marginal = shadow_probs_aux @ weights  # (num_aux, num_audit)

        need_fallback = counts == 0
        if np.any(need_fallback):
            marginal[:, need_fallback] = aux_bundle.global_mean[:, None]

        marginal = np.clip(marginal * float(self.score_scale), self.epsilon, 1.0 - self.epsilon)
        ratio_z = (p_theta_z[:, None] / (marginal + self.epsilon)).T

        likelihoods = ratio_x[:, None] / ratio_z
        return (likelihoods >= float(self.gamma)).mean(axis=1)

    def _score_online_looped(self: Self) -> np.ndarray:
        audit_bundle = self.audit_bundle
        aux_bundle = self.aux_bundle
        if audit_bundle is None or aux_bundle is None:
            raise ValueError("Audit/auxiliary bundles must be prepared before scoring")

        out_model_indices = audit_bundle.extra.get("out_model_indices")
        if out_model_indices is None:
            raise ValueError("Online bundle missing out_model_indices mask")

        p_x_given_target = np.clip(audit_bundle.target_probs, self.epsilon, 1.0 - self.epsilon)
        prior_x = np.clip(audit_bundle.shadow_probs.mean(axis=1) * float(self.score_scale), self.epsilon, 1.0 - self.epsilon)
        ratio_x = p_x_given_target / (prior_x + self.epsilon)

        p_theta_z = np.clip(aux_bundle.target_probs, self.epsilon, 1.0 - self.epsilon)
        shadow_probs_aux = aux_bundle.shadow_probs  # (num_aux, num_models)

        num_audit = audit_bundle.num_points
        num_aux = aux_bundle.num_points
        ratio_z = np.zeros((num_audit, num_aux), dtype=np.float64)

        for i in range(num_audit):
            mask = out_model_indices[:, i]
            if not np.any(mask):
                mask = ~audit_bundle.shadow_inmask[i]
            if not np.any(mask):
                marginal = shadow_probs_aux.mean(axis=1)
            else:
                marginal = shadow_probs_aux[:, mask].mean(axis=1)
            marginal = np.clip(marginal * float(self.score_scale), self.epsilon, 1.0 - self.epsilon)
            ratio_z[i] = p_theta_z / (marginal + self.epsilon)

        likelihoods = ratio_x[:, None] / ratio_z
        return (likelihoods >= float(self.gamma)).mean(axis=1)

    # ------------------------------------------------------------------
    # Attack execution
    # ------------------------------------------------------------------

    def run_attack(self: Self) -> MIAResult:
        if self.audit_bundle is None or self.aux_bundle is None:
            self.prepare_attack()

        if self.online:
            if self.vectorized:
                scores = self._score_online_vectorized()
            else:
                scores = self._score_online_looped()
            membership = self.audit_bundle.target_inmask
        else:
            if self.vectorized:
                scores = self._score_offline_vectorized()
            else:
                scores = self._score_offline_looped()
            membership = self.audit_bundle.target_inmask

        if membership is None:
            raise ValueError("Audit bundle must include membership labels")

        membership = membership.astype(bool)
        self.in_member_signals = scores[membership].reshape(-1, 1)
        self.out_member_signals = scores[~membership].reshape(-1, 1)

        true_labels = np.concatenate(
            (
                np.ones(len(self.in_member_signals)),
                np.zeros(len(self.out_member_signals)),
            )
        )
        signal_values = np.concatenate((self.in_member_signals, self.out_member_signals))

        self.load_for_optuna = True

        return MIAResult.from_full_scores(
            true_membership=true_labels,
            signal_values=signal_values,
            result_name="RMIA",
            metadata=self.configs.model_dump(),
        )

    def reset_attack(self: Self, config: BaseModel) -> None:
        for key, value in config.model_dump().items():
            setattr(self, key, value)
        self.prepare_attack()


# Backwards compatibility alias
AttackRMIAModular = AttackRMIA















# ---------------------------------------------------------------------------
# Lightweight tests (for quick validation)
# ---------------------------------------------------------------------------


def _build_dummy_bundle() -> tuple[RMIAEvaluationBundle, RMIAEvaluationBundle]:
    """Create small deterministic bundles for testing."""
    # Two audit points, two shadow models, binary classification
    audit_target_probs = _sigmoid(np.array([0.4, -0.2]))
    audit_shadow_logits = np.array([[0.2, -0.1], [0.5, 0.0]])  # (points, models)
    audit_shadow_probs = _sigmoid(audit_shadow_logits)
    audit_inmask = np.zeros_like(audit_shadow_probs, dtype=bool)
    audit_membership = np.array([True, False])
    audit_bundle = RMIAEvaluationBundle(
        target_probs=audit_target_probs,
        shadow_probs=audit_shadow_probs,
        shadow_inmask=audit_inmask,
        target_inmask=audit_membership,
    )

    aux_target_probs = _sigmoid(np.array([0.1, 0.3, -0.4]))
    aux_shadow_logits = np.array([
        [0.0, 0.2],
        [0.4, 0.3],
        [-0.2, -0.3],
    ])
    aux_shadow_probs = _sigmoid(aux_shadow_logits)
    aux_inmask = np.zeros_like(aux_shadow_probs, dtype=bool)
    aux_bundle = RMIAEvaluationBundle(
        target_probs=aux_target_probs,
        shadow_probs=aux_shadow_probs,
        shadow_inmask=aux_inmask,
    )
    return audit_bundle, aux_bundle


def test_vectorized_rmia_score_reference() -> None:
    audit_bundle, aux_bundle = _build_dummy_bundle()
    scores = vectorized_rmia_score(
        audit_bundle,
        reference_bundle=aux_bundle,
        use_all_z=True,
        gamma=1.1,
    )
    assert scores.shape == (2,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_attack_rmia_modular_with_precomputed() -> None:
    class DummyHandler:
        population = type("Population", (), {"data": np.zeros((6, 2)), "targets": np.array([0, 1, 0, 1, 0, 1])})()
        population_size = 6
        train_indices = np.array([0, 1])
        test_indices = np.array([2, 3])
        def get_labels(self, indices: Iterable[int]) -> np.ndarray:
            return self.population.targets[np.array(indices, dtype=int)]

    class DummyTargetModel:
        def state_dict(self) -> dict:
            return {}

    dummy_handler = DummyHandler()
    shared_model = DummyTargetModel()
    dummy_handler.target_model = shared_model
    AbstractMIA.population = dummy_handler.population
    AbstractMIA.population_size = dummy_handler.population_size
    AbstractMIA.handler = dummy_handler
    AbstractMIA.target_model = shared_model
    AbstractMIA.audit_dataset = {
        "data": np.array([0, 1, 2, 3]),
        "in_members": np.array([0, 1]),
        "out_members": np.array([2, 3]),
    }
    AbstractMIA._initialized = True

    audit_bundle, aux_bundle = _build_dummy_bundle()
    for flag in (True, False):
        configs = {"online": False, "gamma": 1.1, "num_shadow_models": 2, "vectorized": flag}
        attack = AttackRMIA(dummy_handler, configs)
        attack.set_precomputed_bundles(audit_bundle=audit_bundle, aux_bundle=aux_bundle)
        result = attack.run_attack()
        assert result.result_name == "RMIA"
        assert attack.in_member_signals.shape[0] == np.sum(audit_bundle.target_inmask)
        assert attack.out_member_signals.shape[0] == np.sum(~audit_bundle.target_inmask)


def test_attack_rmia_modular_offline_flow() -> None:
    class DummyHandler:
        population = type("Population", (), {"data": np.zeros((6, 2)), "targets": np.array([0, 1, 0, 1, 0, 1])})()
        population_size = 6
        train_indices = np.array([0, 1])
        test_indices = np.array([2, 3])
        def get_labels(self, indices: Iterable[int]) -> np.ndarray:
            return self.population.targets[np.array(indices, dtype=int)]

    class DummyTargetModel:
        def state_dict(self) -> dict:
            return {}

    dummy_handler = DummyHandler()
    shared_model = DummyTargetModel()
    dummy_handler.target_model = shared_model
    AbstractMIA.population = dummy_handler.population
    AbstractMIA.population_size = dummy_handler.population_size
    AbstractMIA.handler = dummy_handler
    AbstractMIA.target_model = shared_model
    AbstractMIA.audit_dataset = {
        "data": np.array([0, 1, 2, 3]),
        "in_members": np.array([0, 1]),
        "out_members": np.array([2, 3]),
    }
    AbstractMIA._initialized = True

    audit_bundle, aux_bundle = _build_dummy_bundle()

    class StubAttack(AttackRMIA):
        def _prepare_shadow_models(self):  # noqa: D401
            self.shadow_models = ["shadow0", "shadow1"]
            self.shadow_model_indices = np.array([0, 1])

        def _collect_offline_aux_bundle(self) -> RMIAEvaluationBundle:  # noqa: D401
            return aux_bundle.copy()

        def _collect_offline_audit_bundle(self) -> RMIAEvaluationBundle:  # noqa: D401
            return audit_bundle.copy()

    configs = {"online": False, "gamma": 1.1, "num_shadow_models": 2}
    attack = StubAttack(dummy_handler, configs)
    attack.prepare_attack()
    result = attack.run_attack()
    assert result.result_name == "RMIA"
    assert np.isfinite(attack.in_member_signals).all()
    assert np.isfinite(attack.out_member_signals).all()


if __name__ == "__main__":  # pragma: no cover - manual smoke tests
    test_vectorized_rmia_score_reference()
    test_attack_rmia_modular_with_precomputed()
    test_attack_rmia_modular_offline_flow()
    print("RMIA modular smoke tests passed")
