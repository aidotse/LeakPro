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

import functools
import json
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.utils.data import DataLoader

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.signals.token_evidence import CausalLMCollate, CausalLMModel, EvidenceRequest, TokenEvidence
from leakpro.utils.device import get_device
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger
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
    the numeric score (EZ-MIA: ``N == 0`` or no error positions). Those rows — and any row whose
    score is not finite — are placed above the highest ordinary score, ordered among themselves by
    ``tiebreak`` (larger = higher). All returned values are finite, which
    :meth:`~leakpro.reporting.mia_result.MIAResult.from_full_scores` requires: its descending-sort
    monotonicity assert fails on ``nan`` and on two or more ``inf`` values.

    Args:
    ----
        scores: ``(N,)`` raw scores, higher = more likely member. May contain nan/inf.
        force_top: ``(N,)`` bool, rows that must rank as members.
        tiebreak: ``(N,)`` values ordering the forced rows among themselves.

    Returns:
    -------
        ``(N,)`` float64 finite scores with the same ordering as ``scores`` on ordinary rows.

    """
    scores = np.asarray(scores, dtype=np.float64)
    forced = np.asarray(force_top, dtype=bool) | ~np.isfinite(scores)
    out = scores.copy()
    if not forced.any():
        return out
    ordinary = out[~forced]
    base = ordinary.max() if ordinary.size else 0.0
    tb = np.asarray(tiebreak, dtype=np.float64)[forced]
    tb = np.where(np.isfinite(tb), tb, 0.0)
    order = np.argsort(tb, kind="stable")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.float64) / len(order)
    out[forced] = base + 1.0 + ranks
    return out


def _reinitialise(module: nn.Module) -> None:
    """Re-run ``reset_parameters`` on every submodule that defines it (random-init reference)."""
    for child in module.modules():
        reset = getattr(child, "reset_parameters", None)
        if callable(reset):
            reset()


@functools.lru_cache(maxsize=4)
def _load_pretrained(name_or_path: str, dtype: str) -> nn.Module:
    """Load a HuggingFace causal LM. Imported lazily: leakpro core does not depend on transformers."""
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    torch_dtype = getattr(torch, dtype)
    try:
        return AutoModelForCausalLM.from_pretrained(name_or_path, dtype=torch_dtype)
    except TypeError:  # transformers < 5 spells the argument torch_dtype
        return AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=torch_dtype)


def load_reference(
    cfg: ReferenceModelConfig,
    handler: AbstractInputHandler,
    device: Optional[torch.device] = None,
) -> CausalLMModel:
    """Materialise a frozen reference model as a :class:`CausalLMModel`.

    Args:
    ----
        cfg: Which reference to build.
        handler: The MIA handler; supplies the target module and blueprint for ``self`` / ``random_init``.
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
        module = handler.target_model_blueprint(**handler.target_model_metadata.init_params)
        _reinitialise(module)
    else:
        if not cfg.pretrained_name_or_path:
            raise ValueError("reference_model.source='pretrained' requires pretrained_name_or_path")
        module = _load_pretrained(cfg.pretrained_name_or_path, cfg.dtype)
    module.requires_grad_(False)
    logger.info(f"Loaded reference model (source={cfg.source}) onto {device}")
    return CausalLMModel(module, device=device)


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
        self._references: List[CausalLMModel] = []

    @staticmethod
    def _wrap_target_model(handler: AbstractInputHandler) -> CausalLMModel:
        """Wrap the target as a causal LM instead of a classifier."""
        return CausalLMModel(handler.target_model)

    @property
    def membership_labels(self: Self) -> np.ndarray:
        """1 for audit rows that are training members, 0 otherwise, in ``audit_dataset["data"]`` order."""
        n_in = len(self.audit_dataset["in_members"])
        n_out = len(self.audit_dataset["out_members"])
        return np.concatenate([np.ones(n_in), np.zeros(n_out)])

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

    def evidence(self: Self, indices: np.ndarray, request: Optional[EvidenceRequest] = None) -> TokenEvidenceSet:
        """Run the target and every configured reference over ``indices``.

        Models are run one after another; each reference is offloaded to CPU after its pass so at most
        one large model is resident on the device at a time.

        Args:
        ----
            indices: Population indices to score.
            request: Extraction options; defaults to the config's ``need_moments``.

        Returns:
        -------
            Evidence for the target and each reference, rows aligned with ``indices``.

        """
        request = request if request is not None else EvidenceRequest(need_moments=self.configs.need_moments)
        loader = self._make_loader(indices)
        device = self.target_model.device

        logger.info(f"Scoring {len(indices)} sequences with the target model")
        target = self.target_model.evidence_from_loader(loader, request)

        references: List[TokenEvidence] = []
        for i, cfg in enumerate(self.configs.references):
            ref = load_reference(cfg, self.handler, device)
            logger.info(f"Scoring {len(indices)} sequences with reference model {i} (source={cfg.source})")
            references.append(ref.evidence_from_loader(loader, request))
            ref.offload()
        return TokenEvidenceSet(target=target, references=tuple(references), indices=np.asarray(indices))
