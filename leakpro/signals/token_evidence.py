#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Per-token evidence for autoregressive (causal) language models.

Membership-inference attacks against fine-tuned LLMs (EZ-MIA, WBC, LOSS, Reference-Loss,
Min-K%, Min-K%++, ...) are all reductions over the same per-token quantities: the
log-probability the model assigns to each *realised* token given its prefix, the model's
top-1 prediction at that position, and a mask of valid positions. This module extracts
exactly those quantities and nothing else — in particular the dense ``(batch, seq, vocab)``
logits are reduced on-device and never returned or cached, because at realistic sizes that
tensor is hundreds of gigabytes.

Alignment convention (stated once, relied on everywhere): every array in
:class:`TokenEvidence` is *already shifted*. ``logprob[i, j]``, ``argmax[i, j]`` and
``mask[i, j]`` all describe the prediction of ``token_ids[i, j]``, which is original position
``j + 1`` of sequence ``i``. Hence ``T = max_length - 1``.

This deliberately does **not** reuse :class:`leakpro.signals.signal_extractor.PytorchModel`:
that wrapper calls the module with a single positional tensor (no attention mask), assumes a
bare ``(batch, num_classes)`` tensor comes back, registers output-retaining forward hooks on
every child module, and moves the model between devices on every batch. None of that is
usable for a HuggingFace causal LM. :class:`CausalLMModel` implements the same
:class:`~leakpro.signals.signal_extractor.Model` ABC directly instead.

Nothing here imports ``transformers``: model outputs are duck-typed as
``out.logits if hasattr(out, "logits") else out``.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.utils.data import DataLoader

from leakpro.signals.signal_extractor import Model
from leakpro.utils.device import get_device, mark_step
from leakpro.utils.import_helper import Self


@dataclass(frozen=True)
class EvidenceRequest:
    """What to extract beyond the mandatory per-token log-prob / argmax / mask.

    Args:
    ----
        need_moments: Also return the mean and standard deviation of the log-probability
            distribution over the whole vocabulary at every position (``logp_mean`` and
            ``logp_std``). Min-K%++ needs these; they are free inside the forward pass and
            unrecoverable afterwards, so they are extracted on request rather than by default.

    """

    need_moments: bool = False


@dataclass(frozen=True)
class TokenEvidence:
    """Per-token evidence for one model over ``N`` sequences, right-padded to a common ``T``.

    All arrays are already shifted (see module docstring). Padded positions carry
    ``mask == False``, ``logprob == 0``, ``argmax == -1`` and ``token_ids == -1``.

    Args:
    ----
        logprob: ``(N, T)`` float32. ``log p_M(x_{j+1} | x_{<=j})`` of the realised token.
        argmax: ``(N, T)`` int64. The model's top-1 prediction at each position.
        token_ids: ``(N, T)`` int64. The realised tokens the predictions are scored against.
        mask: ``(N, T)`` bool. True where the position is a real (non-padding) prediction.
        lengths: ``(N,)`` int64. Number of scored positions per sequence, ``== mask.sum(1)``.
        logp_mean: ``(N, T)`` float32 or None. Vocabulary-mean of log-prob per position.
        logp_std: ``(N, T)`` float32 or None. Vocabulary-std of log-prob per position.

    """

    logprob: np.ndarray
    argmax: np.ndarray
    token_ids: np.ndarray
    mask: np.ndarray
    lengths: np.ndarray
    logp_mean: Optional[np.ndarray] = None
    logp_std: Optional[np.ndarray] = None

    def __post_init__(self: Self) -> None:
        """Validate shapes and dtypes so downstream reductions can rely on them."""
        shape = self.logprob.shape
        if self.logprob.ndim != 2:
            raise ValueError(f"logprob must be (N, T), got shape {shape}")
        for name in ("argmax", "token_ids", "mask", "logp_mean", "logp_std"):
            arr = getattr(self, name)
            if arr is not None and arr.shape != shape:
                raise ValueError(f"{name} has shape {arr.shape}, expected {shape}")
        if self.lengths.shape != (shape[0],):
            raise ValueError(f"lengths has shape {self.lengths.shape}, expected {(shape[0],)}")
        if self.mask.dtype != np.bool_:
            raise TypeError(f"mask must be bool, got {self.mask.dtype}")
        if not np.array_equal(self.mask.sum(axis=1), self.lengths):
            raise ValueError("lengths must equal mask.sum(axis=1)")

    @property
    def num_sequences(self: Self) -> int:
        """Number of sequences ``N``."""
        return self.logprob.shape[0]

    @property
    def has_moments(self: Self) -> bool:
        """Whether vocabulary moments were extracted."""
        return self.logp_mean is not None

    @staticmethod
    def concatenate(parts: Sequence["TokenEvidence"]) -> "TokenEvidence":
        """Stack evidence from several batches, right-padding every array to the longest ``T``.

        Args:
        ----
            parts: Evidence objects for disjoint batches, in row order.

        Returns:
        -------
            One TokenEvidence covering all rows.

        """
        if not parts:
            raise ValueError("Cannot concatenate an empty list of TokenEvidence")
        if len({p.has_moments for p in parts}) != 1:
            raise ValueError("All parts must agree on whether moments were extracted")
        t_max = max(p.logprob.shape[1] for p in parts)

        def _pad(arr: np.ndarray, fill: float) -> np.ndarray:
            width = t_max - arr.shape[1]
            return arr if width == 0 else np.pad(arr, ((0, 0), (0, width)), constant_values=fill)

        return TokenEvidence(
            logprob=np.concatenate([_pad(p.logprob, 0.0) for p in parts]),
            argmax=np.concatenate([_pad(p.argmax, -1) for p in parts]),
            token_ids=np.concatenate([_pad(p.token_ids, -1) for p in parts]),
            mask=np.concatenate([_pad(p.mask, False) for p in parts]),
            lengths=np.concatenate([p.lengths for p in parts]),
            logp_mean=None if not parts[0].has_moments else np.concatenate([_pad(p.logp_mean, 0.0) for p in parts]),
            logp_std=None if not parts[0].has_moments else np.concatenate([_pad(p.logp_std, 0.0) for p in parts]),
        )


class CausalLMCollate:
    """Right-pad variable-length token-id sequences into ``(input_ids, attention_mask)``.

    Consumes the ``(data, targets)`` 2-tuples LeakPro's ``UserDataset.__getitem__`` yields and
    uses only ``data`` (for a causal LM the labels *are* the input ids). Sequences may be
    tensors, NumPy arrays or lists of ints.

    Args:
    ----
        pad_token_id: Id written into padded positions. Any valid id works — padded positions
            are masked out of every reduction — but GPT-2-family tokenizers define no pad token,
            so it must be given explicitly.

    """

    def __init__(self: Self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self: Self, batch: Sequence[Tuple[object, object]]) -> Tuple[Tensor, Tensor]:
        """Collate a batch of ``(ids, ids)`` pairs.

        Returns
        -------
            ``input_ids`` of shape ``(B, L_max)`` int64 and ``attention_mask`` of the same shape
            with 1 at real tokens and 0 at padding.

        """
        seqs = [torch.as_tensor(np.asarray(item[0]), dtype=torch.long).reshape(-1) for item in batch]
        lengths = [int(s.numel()) for s in seqs]
        if min(lengths) < 2:
            raise ValueError("Every sequence needs at least two tokens to score one prediction")
        l_max = max(lengths)
        input_ids = torch.full((len(seqs), l_max), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(seqs), l_max), dtype=torch.long)
        for row, (seq, length) in enumerate(zip(seqs, lengths)):
            input_ids[row, :length] = seq
            attention_mask[row, :length] = 1
        return input_ids, attention_mask


class CausalLMModel(Model):
    """Query a causal language model for per-token evidence.

    Implements the :class:`~leakpro.signals.signal_extractor.Model` interface so it can travel
    wherever a ``Model`` is expected, but the dense-logit methods (``get_logits``, ``get_grad``,
    ``get_intermediate_outputs``) raise: materialising ``(B, T, vocab)`` for a language model is
    exactly what this class exists to avoid. Use :meth:`token_evidence`.

    Args:
    ----
        model_obj: A module whose forward accepts ``input_ids`` and ``attention_mask`` keyword
            arguments and returns either an object with a ``.logits`` attribute (HuggingFace
            convention) or the ``(B, L, vocab)`` logits tensor itself.
        device: Device to run on. Defaults to :func:`leakpro.utils.device.get_device`.

    """

    def __init__(self: Self, model_obj: nn.Module, device: Optional[torch.device] = None) -> None:
        super().__init__(model_obj, loss_fn=None)
        self.device = device if device is not None else get_device()
        self.model_obj.to(self.device)
        self.model_obj.eval()

    @torch.no_grad()
    def token_evidence(
        self: Self,
        input_ids: Tensor,
        attention_mask: Tensor,
        request: Optional[EvidenceRequest] = None,
    ) -> TokenEvidence:
        """Run one batch and reduce its logits to per-token evidence.

        Args:
        ----
            input_ids: ``(B, L)`` int64 token ids, right-padded.
            attention_mask: ``(B, L)`` with 1 at real tokens, 0 at padding.
            request: What to extract; defaults to log-prob / argmax / mask only.

        Returns:
        -------
            TokenEvidence with ``T = L - 1`` (already shifted).

        """
        request = request or EvidenceRequest()
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        out = self.model_obj(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits if hasattr(out, "logits") else out
        # Position j predicts token j+1: drop the last prediction and the first token.
        # Cast to fp32 *before* log_softmax — bf16 loses precision exactly where the low-probability
        # tail lives, which is where these attacks look.
        logits = logits[:, :-1, :].float()
        targets = input_ids[:, 1:]
        mask = attention_mask[:, 1:].bool()

        logp = F.log_softmax(logits, dim=-1)
        logprob = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        argmax = logits.argmax(dim=-1)

        logp_mean = logp.mean(dim=-1) if request.need_moments else None
        logp_std = logp.std(dim=-1) if request.need_moments else None
        del out, logits, logp  # (B, T, vocab) never leaves the device
        mark_step(self.device)

        # Zero-fill padded positions so the arrays are deterministic regardless of pad id.
        logprob = torch.where(mask, logprob, torch.zeros_like(logprob))
        argmax = torch.where(mask, argmax, torch.full_like(argmax, -1))
        targets = torch.where(mask, targets, torch.full_like(targets, -1))

        def _np(t: Optional[Tensor]) -> Optional[np.ndarray]:
            return None if t is None else t.cpu().numpy()

        mask_np = mask.cpu().numpy()
        return TokenEvidence(
            logprob=logprob.cpu().numpy().astype(np.float32, copy=False),
            argmax=argmax.cpu().numpy().astype(np.int64, copy=False),
            token_ids=targets.cpu().numpy().astype(np.int64, copy=False),
            mask=mask_np,
            lengths=mask_np.sum(axis=1).astype(np.int64),
            logp_mean=None if logp_mean is None else torch.where(mask, logp_mean, torch.zeros_like(logp_mean)).cpu().numpy(),
            logp_std=None if logp_std is None else torch.where(mask, logp_std, torch.zeros_like(logp_std)).cpu().numpy(),
        )

    def evidence_from_loader(
        self: Self,
        loader: Iterable[Tuple[Tensor, Tensor]],
        request: Optional[EvidenceRequest] = None,
    ) -> TokenEvidence:
        """Run every ``(input_ids, attention_mask)`` batch of a loader and concatenate the results.

        The loader must not shuffle: row order is the caller's link back to population indices.

        Args:
        ----
            loader: Yields ``(input_ids, attention_mask)`` batches, e.g. a DataLoader with
                :class:`CausalLMCollate`.
            request: Forwarded to :meth:`token_evidence`.

        Returns:
        -------
            TokenEvidence over all rows, padded to the longest batch.

        """
        if isinstance(loader, DataLoader) and getattr(loader.sampler, "shuffle", False):
            raise ValueError("DataLoader must not shuffle: row order must match the indices it was built from")
        parts: List[TokenEvidence] = [self.token_evidence(ids, mask, request) for ids, mask in loader]
        return TokenEvidence.concatenate(parts)

    def get_loss(self: Self, batch_samples: Tensor, batch_labels: Tensor, per_point: bool = True) -> np.ndarray:
        """Mean next-token negative log-likelihood over the valid positions of each sequence.

        For a causal LM the labels *are* the inputs, so ``batch_labels`` is interpreted as the
        attention mask (1 = real token) rather than as a separate target tensor.

        Args:
        ----
            batch_samples: ``(B, L)`` int64 input ids.
            batch_labels: ``(B, L)`` attention mask.
            per_point: Return a ``(B,)`` array if True, else the scalar mean over sequences.

        Returns:
        -------
            Per-sequence or batch-mean NLL.

        """
        ev = self.token_evidence(torch.as_tensor(batch_samples), torch.as_tensor(batch_labels))
        nll = -(ev.logprob * ev.mask).sum(axis=1) / np.maximum(ev.lengths, 1)
        return nll if per_point else np.asarray(nll.mean())

    def get_logits(self: Self, batch_samples: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Not supported: dense vocabulary logits are intentionally never materialised."""
        raise NotImplementedError("CausalLMModel does not expose dense logits; use token_evidence().")

    def get_grad(self: Self, batch_samples: np.ndarray, batch_labels: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Not supported for language models in this wrapper."""
        raise NotImplementedError("CausalLMModel does not compute gradients; use token_evidence().")

    def get_intermediate_outputs(  # noqa: ARG002
        self: Self,
        layers: List[int],
        batch_samples: np.ndarray,
        forward_pass: bool = True,
    ) -> List[np.ndarray]:
        """Not supported: this wrapper registers no forward hooks by design."""
        raise NotImplementedError("CausalLMModel does not capture intermediate outputs; use token_evidence().")
