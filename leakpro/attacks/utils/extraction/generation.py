#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Candidate generation for memorization extraction.

Two backends produce the same sequences for the same settings:

* ``hf`` calls the model's HuggingFace ``generate()`` (found on the module itself or on its
  ``.model`` attribute, as in ``HFCausalLMWrapper``). It is the fast path for real models (KV
  cache).
* ``torch`` is a plain sampling loop over ``model(input_ids=..., attention_mask=...)``. It needs
  nothing but a causal LM returning logits, so tests and non-HuggingFace models can use it.

Both apply the logits processors in HuggingFace's order -- repetition penalty, minimum new tokens,
temperature decay, temperature, top-k, top-p -- and draw from the global torch RNG, so with the same
seed they sample the same tokens.

``top_k`` is passed explicitly: ``transformers`` 4.x applied ``top_k=50`` by default when sampling,
while 5.x has no default, so leaving it out would change the samples between versions.
"""

import math
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from leakpro.attacks.utils.extraction.records import Candidate
from leakpro.utils.import_helper import Self


class TokenizerLike(Protocol):
    """The part of a HuggingFace tokenizer the extraction stages use."""

    bos_token_id: Optional[int]
    eos_token_id: Optional[int]
    pad_token_id: Optional[int]

    def encode(self: Self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Token ids of ``text``."""
        ...

    def decode(self: Self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        """Text of ``token_ids``."""
        ...


def pad_id_of(tok: TokenizerLike) -> int:
    """Padding id: the tokenizer's pad token, else its EOS token."""
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad is None:
        raise ValueError("The tokenizer defines neither a pad nor an EOS token; cannot pad batches.")
    return int(pad)


class TempDecayLogitsProcessor:
    """Divide logits by a temperature that decays linearly from ``t_start`` to ``t_end``.

    The temperature is ``t_start`` for the first generated token and reaches ``t_end`` after
    ``decay_steps`` generated tokens; prompt positions are left untouched.

    Args:
    ----
        t_start: Temperature for the first generated token.
        t_end: Temperature once the decay is over.
        decay_steps: Number of generated tokens over which the temperature decays.
        prompt_len: Padded prompt length, used to count generated tokens.

    """

    def __init__(self: Self, t_start: float, t_end: float, decay_steps: int, prompt_len: int) -> None:
        self.t_start = float(t_start)
        self.t_end = float(t_end)
        self.decay_steps = int(decay_steps)
        self.prompt_len = int(prompt_len)

    def temperature(self: Self, step: int) -> float:
        """Temperature applied when ``step`` tokens have been generated so far."""
        if step >= self.decay_steps:
            return self.t_end
        return max(self.t_start - (self.t_start - self.t_end) * (step / self.decay_steps), 0.01)

    def __call__(self: Self, input_ids: Tensor, scores: Tensor) -> Tensor:
        """Scale ``scores`` by the current temperature."""
        step = input_ids.shape[1] - self.prompt_len
        if step < 0:
            return scores
        return scores / self.temperature(step)


@dataclass(frozen=True)
class SamplingSettings:
    """Decoding settings for one generation call.

    Args:
    ----
        max_new_tokens: Upper bound on generated tokens.
        min_new_tokens: EOS is suppressed until this many tokens have been generated.
        do_sample: Sample (True) or decode greedily (False).
        temperature: Fixed temperature (sampling only).
        top_k: Keep the ``top_k`` most likely tokens (sampling only; None or 0 disables).
        top_p: Nucleus threshold (sampling only; 1.0 disables).
        repetition_penalty: HuggingFace-style penalty on tokens already in the sequence (1.0 disables).
        temp_decay: ``(t_start, t_end, decay_steps)`` for :class:`TempDecayLogitsProcessor`, or None.

    """

    max_new_tokens: int
    min_new_tokens: int = 0
    do_sample: bool = True
    temperature: float = 1.0
    top_k: Optional[int] = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    temp_decay: Optional[Tuple[float, float, int]] = None


def left_pad(prompts: Sequence[Sequence[int]], pad_id: int) -> Tuple[Tensor, Tensor]:
    """Left-pad token-id prompts into ``(input_ids, attention_mask)``."""
    if not prompts or any(len(p) == 0 for p in prompts):
        raise ValueError("Every prompt needs at least one token")
    width = max(len(p) for p in prompts)
    input_ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(prompts), width), dtype=torch.long)
    for row, prompt in enumerate(prompts):
        input_ids[row, width - len(prompt):] = torch.as_tensor(list(prompt), dtype=torch.long)
        attention_mask[row, width - len(prompt):] = 1
    return input_ids, attention_mask


def find_hf_generate(model: nn.Module) -> Optional[Callable]:
    """Return a HuggingFace ``generate`` for ``model`` (or its ``.model``), or None."""
    for obj in (model, getattr(model, "model", None)):
        if obj is not None and callable(getattr(obj, "generate", None)):
            return obj.generate
    return None


def _apply_repetition_penalty(input_ids: Tensor, scores: Tensor, penalty: float) -> Tensor:
    score = torch.gather(scores, 1, input_ids)
    score = torch.where(score < 0, score * penalty, score / penalty)
    return scores.scatter(1, input_ids, score)


def _apply_top_k(scores: Tensor, top_k: int) -> Tensor:
    top_k = min(top_k, scores.size(-1))
    kth = torch.topk(scores, top_k)[0][..., -1, None]
    return scores.masked_fill(scores < kth, -float("inf"))


def _apply_top_p(scores: Tensor, top_p: float) -> Tensor:
    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
    cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_remove = cumulative <= (1 - top_p)
    sorted_remove[..., -1:] = False
    remove = sorted_remove.scatter(1, sorted_indices, sorted_remove)
    return scores.masked_fill(remove, -float("inf"))


def _process_scores(
    input_ids: Tensor,
    scores: Tensor,
    step: int,
    settings: SamplingSettings,
    decay: Optional[TempDecayLogitsProcessor],
    eos_id: Optional[int],
) -> Tensor:
    """Apply the logits processors in HuggingFace's order to next-token ``scores``."""
    if settings.repetition_penalty != 1.0:
        scores = _apply_repetition_penalty(input_ids, scores, settings.repetition_penalty)
    if eos_id is not None and step < settings.min_new_tokens:
        scores[:, eos_id] = -float("inf")
    if decay is not None:
        scores = decay(input_ids, scores)
    if settings.do_sample:
        if settings.temperature != 1.0:
            scores = scores / settings.temperature
        if settings.top_k:
            scores = _apply_top_k(scores, settings.top_k)
        if settings.top_p < 1.0:
            scores = _apply_top_p(scores, settings.top_p)
    return scores


@torch.no_grad()
def _generate_torch(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    settings: SamplingSettings,
    pad_id: int,
    eos_id: Optional[int],
) -> Tensor:
    """Plain sampling loop without a KV cache. Returns only the generated tokens ``(B, <=max_new)``."""
    prompt_len = input_ids.shape[1]
    decay = TempDecayLogitsProcessor(*settings.temp_decay, prompt_len=prompt_len) if settings.temp_decay else None
    unfinished = torch.ones(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    for step in range(settings.max_new_tokens):
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits if hasattr(out, "logits") else out
        scores = _process_scores(input_ids, logits[:, -1, :].float(), step, settings, decay, eos_id)
        if settings.do_sample:
            next_tokens = torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(scores, dim=-1)
        next_tokens = torch.where(unfinished, next_tokens, torch.full_like(next_tokens, pad_id))
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_tokens)[:, None]], dim=-1)
        if eos_id is not None:
            unfinished &= next_tokens != eos_id
            if not unfinished.any():
                break
    return input_ids[:, prompt_len:]


@torch.no_grad()
def _generate_hf(
    generate: Callable,
    input_ids: Tensor,
    attention_mask: Tensor,
    settings: SamplingSettings,
    pad_id: int,
    eos_id: Optional[int],
) -> Tensor:
    """HuggingFace ``generate()``. Returns only the generated tokens."""
    from transformers import LogitsProcessorList  # noqa: PLC0415

    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": settings.max_new_tokens,
        "do_sample": settings.do_sample,
        "pad_token_id": pad_id,
    }
    if eos_id is not None:
        kwargs["eos_token_id"] = eos_id
    if settings.min_new_tokens:
        kwargs["min_new_tokens"] = settings.min_new_tokens
    if settings.repetition_penalty != 1.0:
        kwargs["repetition_penalty"] = settings.repetition_penalty
    if settings.do_sample:
        kwargs.update(temperature=settings.temperature, top_k=settings.top_k or 0, top_p=settings.top_p)
    if settings.temp_decay:
        kwargs["logits_processor"] = LogitsProcessorList(
            [TempDecayLogitsProcessor(*settings.temp_decay, prompt_len=input_ids.shape[1])]
        )
    out = generate(**kwargs)
    return out[:, input_ids.shape[1]:]


def generate_tokens(
    model: nn.Module,
    prompts: Sequence[Sequence[int]],
    settings: SamplingSettings,
    pad_id: int,
    eos_id: Optional[int],
    seed: int,
    device: torch.device,
    backend: Literal["auto", "hf", "torch"] = "auto",
) -> List[List[int]]:
    """Generate continuations for a batch of token-id prompts.

    Args:
    ----
        model: Causal LM (a HuggingFace model, a wrapper exposing one as ``.model``, or any module
            returning logits from ``model(input_ids=..., attention_mask=...)``).
        prompts: Token ids per row; rows may differ in length (left-padded here).
        settings: Decoding settings.
        pad_id: Id used for left padding and after EOS.
        eos_id: End-of-sequence id, or None.
        seed: Seed for the global torch RNG before generating.
        device: Device to run on.
        backend: ``hf``, ``torch``, or ``auto`` (``hf`` when a ``generate()`` is available).

    Returns:
    -------
        The generated token ids per row, padded positions removed, cut after the first EOS.

    """
    generate = find_hf_generate(model)
    if backend == "hf" and generate is None:
        raise ValueError("backend='hf' requested but the model has no HuggingFace generate()")
    use_hf = generate is not None and backend in ("auto", "hf")

    input_ids, attention_mask = left_pad(prompts, pad_id)
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
    torch.manual_seed(seed)
    if use_hf:
        new = _generate_hf(generate, input_ids, attention_mask, settings, pad_id, eos_id)
    else:
        new = _generate_torch(model, input_ids, attention_mask, settings, pad_id, eos_id)

    rows: List[List[int]] = []
    for ids in new.cpu().tolist():
        if eos_id is not None and eos_id in ids:
            ids = ids[: ids.index(eos_id) + 1]
        rows.append(ids)
    return rows


def batch_seed(seed: int, strategy_index: int, batch_index: int) -> int:
    """Seed for one generation batch, independent of batch size changes elsewhere and of resumes."""
    return int(np.random.SeedSequence([seed, strategy_index, batch_index]).generate_state(1)[0])


class StrategyGenerator:
    """Generate the candidates of one prompting strategy, batch by batch.

    Strategies:

    * ``baseline``: BOS-only prompt, sampling at ``temperature``.
    * ``temp_decay``: BOS-only prompt, sampling with a temperature decaying from ``t_start`` to ``t_end``.
    * ``conditional``: prompt = a prefix from the prefix pool, greedy decoding to ``max_tokens``
      tokens *in total* (prefix + continuation). A batch generates
      ``max_tokens - shortest prefix`` tokens and each row keeps ``max_tokens - its prefix length``.

    Args:
    ----
        name: Strategy name stored on every candidate.
        strategy_index: Position of the strategy in the config (part of every batch seed).
        cfg: The strategy config (see ``GenStrategyConfig``).
        tok: Tokenizer.
        prefixes: Prefix texts (conditional only).
        phase: Phase name stored on every candidate.

    """

    def __init__(  # noqa: D417
        self: Self,
        name: str,
        strategy_index: int,
        cfg: object,
        tok: TokenizerLike,
        prefixes: Optional[Sequence[str]] = None,
        phase: str = "initial",
    ) -> None:
        self.name = name
        self.strategy_index = strategy_index
        self.cfg = cfg
        self.tok = tok
        self.phase = phase
        if cfg.type == "conditional":
            if not prefixes:
                raise ValueError("The conditional strategy needs a prefix pool (target.data_path)")
            self.prefixes = list(prefixes)[: cfg.n]
            self.total = len(self.prefixes)
        else:
            if tok.bos_token_id is None:
                raise ValueError(f"Strategy {cfg.type!r} prompts with the BOS token, but the tokenizer has none")
            self.prefixes = None
            self.total = int(cfg.n)

    def num_batches(self: Self) -> int:
        """Number of generation batches."""
        return math.ceil(self.total / self.cfg.batch_size)

    def _rows(self: Self, batch_index: int) -> range:
        start = batch_index * self.cfg.batch_size
        return range(start, min(start + self.cfg.batch_size, self.total))

    def generate_batch(
        self: Self,
        batch_index: int,
        model: nn.Module,
        seed: int,
        device: torch.device,
        backend: Literal["auto", "hf", "torch"] = "auto",
    ) -> List[Candidate]:
        """Generate one batch and wrap every row as a :class:`Candidate`."""
        cfg, tok = self.cfg, self.tok
        rows = self._rows(batch_index)
        pad_id, eos_id = pad_id_of(tok), tok.eos_token_id
        seed_b = batch_seed(seed, self.strategy_index, batch_index)

        if cfg.type == "conditional":
            prompt_texts = [self.prefixes[i] for i in rows]
            prompts = [tok.encode(text, add_special_tokens=False) for text in prompt_texts]
            gen_len = max(cfg.max_tokens - min(len(p) for p in prompts), 1)
            settings = SamplingSettings(max_new_tokens=gen_len, min_new_tokens=gen_len, do_sample=False)
            new = generate_tokens(model, prompts, settings, pad_id, eos_id, seed_b, device, backend)
            new = [ids[: max(cfg.max_tokens - len(p), 0)] for ids, p in zip(new, prompts)]
        else:
            prompt_texts = [""] * len(rows)
            prompts = [[tok.bos_token_id] for _ in rows]
            decay = (cfg.t_start, cfg.t_end, cfg.decay_steps) if cfg.type == "temp_decay" else None
            settings = SamplingSettings(
                max_new_tokens=cfg.max_tokens, do_sample=True, temperature=cfg.temperature,
                top_k=cfg.top_k, top_p=cfg.top_p, temp_decay=decay,
            )
            new = generate_tokens(model, prompts, settings, pad_id, eos_id, seed_b, device, backend)

        candidates = []
        for row, prompt, prompt_text, ids in zip(rows, prompts, prompt_texts, new):
            continuation = tok.decode(ids, skip_special_tokens=True)
            candidates.append(Candidate(
                id=f"{self.phase}-{self.name}-{row:06d}",
                phase=self.phase,
                strategy=self.name,
                prompt_ids=list(prompt),
                gen_ids=list(ids),
                prompt_text=prompt_text,
                continuation=continuation,
                full_text=f"{prompt_text} {continuation}".strip(),
                meta={"batch": batch_index},
            ))
        return candidates
