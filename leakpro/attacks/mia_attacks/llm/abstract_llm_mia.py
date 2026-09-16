#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Shared base for membership-inference attacks on fine-tuned causal language models.

The attacks in this family (EZ-MIA, WBC, LOSS, Reference-Loss, Min-K%, ...) differ only in the
NumPy reduction they apply to per-token evidence. Everything else is here:

* the attack-side dataloader (built directly from ``handler.get_dataset`` with
  :class:`~leakpro.signals.token_evidence.CausalLMCollate`, because the handler's stored
  training-time dataloader parameters cannot be trusted to yield ``(input_ids, mask)`` batches);
* loading of frozen *reference* models — the pretrained base checkpoint, the target itself, or
  a re-initialised copy of the target — declared inside the attack's own config so that they
  land in the attack hash and the result metadata;
* :func:`rank_top`, which turns "this sample must be classified as a member" edge cases into
  finite scores that :meth:`MIAResult.from_full_scores` accepts.

None of these attacks train anything, so they opt out of the shadow/distillation handlers.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.signals.token_evidence import CausalLMCollate, CausalLMModel, EvidenceRequest, TokenEvidence
from leakpro.utils.device import get_device
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger
from leakpro.utils.save_load import fingerprint_model, hash_attack
from leakpro.utils.seed import seed_everything


class ReferenceModelConfig(BaseModel):
    """Where a frozen reference language model comes from.

    ``pretrained`` is the paper-default for both EZ-MIA and WBC: the base checkpoint the target was
    fine-tuned from. ``self`` (target as its own reference, delta ≡ 0) and ``random_init`` are
    EZ-MIA's sanity-check ablations (paper appendix E.2) and cost no download.
    """

    source: Literal["pretrained", "self", "random_init"] = Field(default="pretrained", description="Reference kind")
    pretrained_name_or_path: Optional[str] = Field(default=None, description="HF hub id or local dir (source=pretrained)")
    dtype: Literal["float32", "bfloat16", "float16"] = Field(default="bfloat16", description="Load dtype (source=pretrained)")

    model_config = ConfigDict(extra="forbid")

    def key(self: Self) -> str:
        """Canonical JSON used to memoise loaded models."""
        return json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))


class LLMAttackConfig(BaseModel):
    """Fields every LLM attack config carries. Attack-specific configs subclass this."""

    references: List[ReferenceModelConfig] = Field(default_factory=list, description="Frozen reference models")
    batch_size: int = Field(default=8, ge=1, description="Sequences per forward pass")
    pad_token_id: int = Field(default=0, ge=0, description="Id used to right-pad; masked out of every reduction")
    need_moments: bool = Field(default=False, description="Extract per-position vocab mean/std of log p (Min-K%++)")
    max_samples: Optional[int] = Field(
        default=None, ge=1,
        description="Cap on audited rows (paper configs' `test_samples`), stratified to keep the "
                    "member/non-member ratio and seeded by `audit.random_seed`. None audits everything.",
    )
    n_bootstrap_samples: Optional[int] = Field(
        default=None, ge=1,
        description="If set, resample the audit rows with replacement this many times and report "
                    "mean/CI for AUC and the fixed-FPR TPRs alongside the point estimate.",
    )

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class TokenEvidenceSet:
    """Evidence for the target and each configured reference over the same rows.

    Args:
    ----
        target: Evidence under the audited (fine-tuned) model.
        references: Evidence under each reference, in config order. Empty for reference-free attacks.
        indices: Population indices the rows correspond to.

    """

    target: TokenEvidence
    references: Tuple[TokenEvidence, ...]
    indices: np.ndarray

    def ref(self: Self, i: int = 0) -> TokenEvidence:
        """Evidence for the ``i``-th reference model."""
        return self.references[i]


def rank_top(scores: np.ndarray, force_top: np.ndarray, tiebreak: np.ndarray) -> np.ndarray:
    """Return finite scores where forced rows sit strictly above every ordinary row.

    Several attacks have configurations the paper says must be classified as members regardless of
    the numeric score (EZ-MIA: ``N == 0`` or no error positions). Those rows, and any row whose
    score is ``+inf``, are placed above the highest ordinary score, ordered among themselves by
    ``tiebreak`` (larger = higher). A ``-inf`` score is a legitimately *weakest* signal (e.g.
    ``log(P/N)`` with ``P == 0``) and is placed just below the lowest ordinary score, so monotone
    transforms of the same statistic keep the same ranking. ``nan`` outside the forced rows is a
    caller bug and raises. All returned values are finite, which
    :meth:`~leakpro.reporting.mia_result.MIAResult.from_full_scores` requires: its descending-sort
    monotonicity assert fails on ``nan`` and on two or more ``inf`` values.

    Args:
    ----
        scores: ``(N,)`` raw scores, higher = more likely member. May contain ±inf; nan only where forced.
        force_top: ``(N,)`` bool, rows that must rank as members.
        tiebreak: ``(N,)`` values ordering the forced rows among themselves.

    Returns:
    -------
        ``(N,)`` float64 finite scores with the same ordering as ``scores`` on ordinary rows.

    """
    scores = np.asarray(scores, dtype=np.float64)
    forced = np.asarray(force_top, dtype=bool) | np.isposinf(scores)
    bottom = np.isneginf(scores) & ~forced
    if np.isnan(scores[~forced]).any():
        raise ValueError("rank_top: nan score in a row that is not forced to the top")
    out = scores.copy()
    ordinary = out[~forced & ~bottom]
    if bottom.any():
        out[bottom] = (ordinary.min() if ordinary.size else 0.0) - 1.0
    if not forced.any():
        return out
    base = out[~forced].max() if (~forced).any() else 0.0
    tb = np.asarray(tiebreak, dtype=np.float64)[forced]
    tb = np.where(np.isfinite(tb), tb, 0.0)
    order = np.argsort(tb, kind="stable")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.float64) / len(order)
    out[forced] = base + 1.0 + ranks
    return out


def gpt2_style_init_(module: nn.Module, std: float = 0.02) -> None:
    """Re-initialise every parameter of ``module`` in place at GPT-2's scale (random-init reference).

    Weights with two or more dimensions (``Linear``, HF ``Conv1D``, ``Embedding``) get ``N(0, std)``;
    one-dimensional weights (``LayerNorm``) get ones; biases get zeros. This deliberately does *not*
    go through each module's ``reset_parameters``: HF ``Conv1D`` has none, and ``nn.Embedding``'s is
    ``N(0, 1)`` — fifty times GPT-2's scale — so the paper's random-reference ablation would not be
    representative.
    """
    with torch.no_grad():
        for name, param in module.named_parameters():
            leaf = name.rsplit(".", 1)[-1]
            if param.dim() >= 2:
                nn.init.normal_(param, mean=0.0, std=std)
            elif leaf == "bias":
                nn.init.zeros_(param)
            else:
                nn.init.ones_(param)


def load_pretrained_causal_lm(name_or_path: str, dtype: str = "float32") -> nn.Module:
    """Load a HuggingFace ``AutoModelForCausalLM``.

    Uncached on purpose: the example's target wrapper fine-tunes what it loads, so it must get a fresh
    module. The reference path memoises per run via :func:`_run_memo`. ``transformers`` is imported
    lazily so leakpro core does not depend on it; both argument spellings (``dtype`` in >= 5,
    ``torch_dtype`` before) are handled here and nowhere else.
    """
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    torch_dtype = getattr(torch, dtype)
    try:
        return AutoModelForCausalLM.from_pretrained(name_or_path, dtype=torch_dtype)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=torch_dtype)


def _run_memo(handler: AbstractInputHandler) -> dict:
    """Per-handler memo shared by every LLM attack built on that handler.

    Holds extracted evidence (key: ``(model_key, need_moments, indices_hash)``) and loaded reference
    modules (key: ``("module", cfg.key())``). Attacks in one ``attack_list`` share a handler, so the
    second attack reuses the first's forward passes and models; the memo dies with the handler, i.e.
    with the run, so nothing is shared across audits in one process.

    ``batch_size`` is deliberately not part of the evidence key: per-sequence evidence is independent
    of how sequences were batched (padding is masked out; pinned by the token_evidence tests).
    """
    memo = getattr(handler, "_llm_run_memo", None)
    if memo is None:
        memo = {}
        handler._llm_run_memo = memo
    return memo


def load_reference(
    cfg: ReferenceModelConfig,
    handler: AbstractInputHandler,
    device: Optional[torch.device] = None,
) -> CausalLMModel:
    """Materialise a frozen reference model as a :class:`CausalLMModel`.

    Args:
    ----
        cfg: Which reference to build.
        handler: The MIA handler; supplies the target module and replica constructor for ``self`` /
            ``random_init`` and hosts the per-run memo for ``pretrained`` modules.
        device: Device to place the model on. Defaults to :func:`leakpro.utils.device.get_device`.

    Returns:
    -------
        The wrapped reference, in eval mode with gradients disabled.

    """
    device = device if device is not None else get_device()
    if cfg.source == "self":
        module = handler.target_model
    elif cfg.source == "random_init":
        seed_everything(handler.configs.audit.random_seed)
        module, _, _ = handler.get_target_replica()  # same construction path (and GroupNorm fix-up) as shadow models
        gpt2_style_init_(module)
    else:
        if not cfg.pretrained_name_or_path:
            raise ValueError("reference_model.source='pretrained' requires pretrained_name_or_path")
        memo = _run_memo(handler)
        module = memo.get(("module", cfg.key()))
        if module is None:
            module = load_pretrained_causal_lm(cfg.pretrained_name_or_path, cfg.dtype)
            memo[("module", cfg.key())] = module
    module.requires_grad_(False)
    logger.info(f"Loaded reference model (source={cfg.source}) onto {device}")
    return CausalLMModel(module, device=device)


def bootstrap_auc_and_tpr(
    true_membership: np.ndarray,
    signal_values: np.ndarray,
    n_bootstrap_samples: int,
    fpr_thresholds: Optional[List[float]] = None,
    seed: Optional[int] = None,
) -> dict:
    """Bootstrap-resample rows with replacement and report mean/95% CI for AUC and TPR at fixed FPRs.

    Matches the shape of the paper reference configs' ``n_bootstrap_samples`` field, which this repo's
    :class:`~leakpro.reporting.mia_result.MIAResult` does not otherwise compute (it reports a single
    point estimate). Returned as a plain ``dict`` rather than folded into ``MIAResult`` so it does not
    change that class's schema/serialization — attach it manually if wanted, e.g.
    ``result.bootstrap = bootstrap_auc_and_tpr(...)``.

    Args:
    ----
        true_membership: ``(N,)`` 1/0 ground truth, same order as ``signal_values``.
        signal_values: ``(N,)`` attack scores, higher = more likely member.
        n_bootstrap_samples: Number of resamples (the papers typically use 10-100).
        fpr_thresholds: FPRs to report TPR at; defaults to the same five
            :meth:`~leakpro.reporting.mia_result.MIAResult._compute_metrics` uses.
        seed: Seeds the resampling for reproducibility.

    Returns:
    -------
        ``{"n_bootstrap_samples", "auc": {mean, ci_low, ci_high, n}, "tpr_at_fpr": {"<fpr>": {...}, ...}}``.
        A resample that draws only one class contributes to no metric (no ROC curve exists); ``n`` in
        each entry is how many of the ``n_bootstrap_samples`` draws actually contributed.

    """
    fpr_thresholds = fpr_thresholds if fpr_thresholds is not None else [0.0, 0.0001, 0.001, 0.01, 0.1]
    true_membership = np.asarray(true_membership)
    signal_values = np.asarray(signal_values)
    n = len(true_membership)
    rng = np.random.RandomState(seed)

    aucs: List[float] = []
    tprs: dict = {t: [] for t in fpr_thresholds}
    for _ in range(n_bootstrap_samples):
        idx = rng.choice(n, size=n, replace=True)
        y, s = true_membership[idx], signal_values[idx]
        if len(np.unique(y)) < 2:
            continue
        aucs.append(roc_auc_score(y, s))
        fpr, tpr, _ = roc_curve(y, s)
        for t in fpr_thresholds:
            tprs[t].append(float(np.interp(t, fpr, tpr)))

    def _mean_ci(values: List[float]) -> dict:
        arr = np.asarray(values)
        if arr.size == 0:
            return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
        return {
            "mean": float(arr.mean()),
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
            "n": int(arr.size),
        }

    return {
        "n_bootstrap_samples": n_bootstrap_samples,
        "auc": _mean_ci(aucs),
        "tpr_at_fpr": {f"{t:g}": _mean_ci(v) for t, v in tprs.items()},
    }


class AbstractLLMMIA(AbstractMIA):
    """Base class for LLM membership-inference attacks.

    Subclasses set ``AttackConfig`` (an :class:`LLMAttackConfig` subclass), implement
    ``description()``, and typically implement ``prepare_attack`` as one call to :meth:`evidence`
    and ``run_attack`` as a reduction over the result.
    """

    requires_shadow_models = False
    requires_distillation_models = False

    def __init__(self: Self, handler: AbstractInputHandler, configs: Optional[dict]) -> None:
        """Validate the config, initialise the shared MIA state and flatten config fields onto ``self``."""
        self.configs = self.AttackConfig(**(configs or {}))
        super().__init__(handler)
        for key, value in self.configs.model_dump().items():
            setattr(self, key, value)

    @staticmethod
    def _wrap_target_model(handler: AbstractInputHandler) -> CausalLMModel:
        """Wrap the target as a causal LM instead of a classifier."""
        return CausalLMModel(handler.target_model)

    def _hash_attack(self: Self) -> None:
        """Attack id from the config and a cheap target fingerprint.

        The default hashes every weight, which for a multi-billion-parameter target reads tens of GB
        per attack construction — the cost opting out of the shadow handlers was meant to avoid.
        """
        self.attack_id = hash_attack(self.configs.model_dump(), self.handler.target_model, model_hasher=fingerprint_model)

    @property
    def membership_labels(self: Self) -> np.ndarray:
        """1 for audit rows that are training members, 0 otherwise, in ``audit_dataset["data"]`` order.

        Reflects the *full* audit set regardless of ``max_samples``; use :meth:`_audit_indices_and_labels`
        for the (possibly subsampled) rows an attack should actually score.
        """
        n_in = len(self.audit_dataset["in_members"])
        n_out = len(self.audit_dataset["out_members"])
        return np.concatenate([np.ones(n_in), np.zeros(n_out)])

    def _audit_indices_and_labels(self: Self) -> Tuple[np.ndarray, np.ndarray]:
        """Population indices to audit and their true membership labels, aligned 1:1.

        With ``max_samples`` unset (the default) this is just ``(audit_dataset["data"], membership_labels)``
        — everything. When set (the papers' ``test_samples``), it is a stratified subsample: the
        member/non-member ratio of the full audit set is preserved, and the draw is seeded by
        ``handler.configs.audit.random_seed`` so it is reproducible across runs and across attacks in one
        ``attack_list`` (they draw independently but from the same seed and the same full set).
        """
        indices = np.asarray(self.audit_dataset["data"])
        labels = self.membership_labels
        max_samples = self.configs.max_samples
        if max_samples is None or max_samples >= len(indices):
            return indices, labels

        rng = np.random.RandomState(self.handler.configs.audit.random_seed)
        n_in = len(self.audit_dataset["in_members"])
        n_out = len(indices) - n_in
        keep_in = int(round(max_samples * n_in / len(indices)))
        keep_in = max(1, min(keep_in, n_in)) if n_in else 0
        keep_out = max(0, min(max_samples - keep_in, n_out))

        in_pos = rng.choice(n_in, size=keep_in, replace=False) if keep_in else np.array([], dtype=np.int64)
        out_pos = (rng.choice(np.arange(n_in, len(indices)), size=keep_out, replace=False)
                  if keep_out else np.array([], dtype=np.int64))
        keep_pos = np.concatenate([in_pos, out_pos]).astype(np.int64)
        return indices[keep_pos], labels[keep_pos]

    def _attach_bootstrap_if_configured(self: Self, result: MIAResult, labels: np.ndarray, scores: np.ndarray) -> MIAResult:
        """Attach ``result.bootstrap`` (see :func:`bootstrap_auc_and_tpr`) when ``n_bootstrap_samples`` is set.

        A plain-attribute addition, not a new ``MIAResult`` field: it does not round-trip through
        ``to_json``/``from_json``. No-op (returns ``result`` unchanged) when not configured.
        """
        if self.configs.n_bootstrap_samples:
            result.bootstrap = bootstrap_auc_and_tpr(
                labels, scores, self.configs.n_bootstrap_samples,
                seed=self.handler.configs.audit.random_seed,
            )
        return result

    def _require_references(self: Self, n: int) -> None:
        """Raise a clear error if the config declares fewer than ``n`` reference models."""
        if len(self.configs.references) < n:
            raise ValueError(
                f"{self.__class__.__name__} needs {n} reference model(s) but the attack config declares "
                f"{len(self.configs.references)}. Add a `references:` list (e.g. source: pretrained, "
                "pretrained_name_or_path: <base checkpoint the target was fine-tuned from>)."
            )

    def _make_loader(self: Self, indices: np.ndarray) -> DataLoader:
        """Dataloader over the population rows ``indices`` yielding ``(input_ids, attention_mask)``."""
        dataset = self.handler.get_dataset(np.asarray(indices))
        return DataLoader(dataset, batch_size=self.configs.batch_size, shuffle=False,
                          collate_fn=CausalLMCollate(pad_token_id=self.configs.pad_token_id))

    def _score_model(
        self: Self,
        model: CausalLMModel,
        model_key: str,
        loader: DataLoader,
        indices_key: str,
        request: EvidenceRequest,
        label: str,
    ) -> TokenEvidence:
        """Run ``model`` over ``loader`` unless identical evidence is already memoised; offload afterwards."""
        key = (model_key, request.need_moments, indices_key)
        memo = _run_memo(self.handler)
        if key in memo:
            logger.info(f"Reusing memoised evidence for {label}")
            return memo[key]
        logger.info(f"Scoring {label}")
        evidence = model.evidence_from_loader(loader, request)
        model.offload()
        memo[key] = evidence
        return evidence

    def evidence(self: Self, indices: np.ndarray, request: Optional[EvidenceRequest] = None) -> TokenEvidenceSet:
        """Run the target and every configured reference over ``indices``.

        Models are run one after another and each — the target included — is offloaded to CPU after
        its pass, so at most one model is resident on the device at a time. Results are memoised per
        handler (see :func:`_run_memo`), so several LLM attacks in one audit share the forward passes.
        A ``source: self`` reference *is* the target and reuses its evidence without a second pass.

        Args:
        ----
            indices: Population indices to score.
            request: Extraction options; defaults to the config's ``need_moments``.

        Returns:
        -------
            Evidence for the target and each reference, rows aligned with ``indices``.

        """
        request = request if request is not None else EvidenceRequest(need_moments=self.configs.need_moments)
        indices = np.asarray(indices)
        indices_key = hashlib.sha256(indices.astype(np.int64).tobytes()).hexdigest()
        loader = self._make_loader(indices)
        device = self.target_model.device
        n = len(indices)

        target = self._score_model(self.target_model, "target", loader, indices_key, request,
                                   label=f"{n} sequences with the target model")

        references: List[TokenEvidence] = []
        for i, cfg in enumerate(self.configs.references):
            if cfg.source == "self":
                logger.info(f"Reference model {i} is the target itself; reusing its evidence")
                references.append(target)
                continue
            key = (cfg.key(), request.need_moments, indices_key)
            memo = _run_memo(self.handler)
            if key in memo:
                references.append(memo[key])
                logger.info(f"Reusing memoised evidence for reference model {i} (source={cfg.source})")
                continue
            ref = load_reference(cfg, self.handler, device)
            references.append(self._score_model(ref, cfg.key(), loader, indices_key, request,
                                                label=f"{n} sequences with reference model {i} (source={cfg.source})"))
        return TokenEvidenceSet(target=target, references=tuple(references), indices=indices)
