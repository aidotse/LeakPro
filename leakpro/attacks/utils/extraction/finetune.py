#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Optional Phases 3 and 4: fine-tune on what was extracted, then extract again.

Phase 3:

* :func:`build_finetune_corpus`: S0 ∪ S1-novel, drop texts under ``min_words`` words, deduplicate on
  the normalized text (the first original text is kept), shuffle with ``seed``.
* :func:`attach_lora` / :func:`train_lora`: LoRA continual pre-training on the raw texts (causal LM
  loss on every token, no prompt/response structure). The adapter is injected *in place*, so the
  same model object generates with the adapter, and :func:`adapter_disabled` gives back the base
  model (Phase 4 scores with the base model). :func:`detach_lora` removes the adapter at the end.

Phase 4:

* :class:`TokenWindowIndex`: every ``window``-token window of the fine-tuning corpus (normalized
  text, tokenized without special tokens). A Phase 4 generation sharing any window with it is
  dropped before filtering, since reproducing fine-tuning data is not evidence of memorization.

``peft`` is imported only when LoRA is used.
"""

import contextlib
import math
from typing import Any, Iterator, List, Optional, Sequence, Set

import numpy as np
import torch
from torch import nn

from leakpro.attacks.utils.extraction.generation import TokenizerLike, pad_id_of
from leakpro.attacks.utils.extraction.progress import Progress
from leakpro.attacks.utils.extraction.records import normalize
from leakpro.signals.token_evidence import CausalLMCollate
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


def build_finetune_corpus(texts: Sequence[str], min_words: int = 20, seed: int = 42) -> List[str]:
    """Union of the given texts: filtered, deduplicated and shuffled.

    The shuffle is ``numpy.random.default_rng(seed).permutation``, the same as ``datasets.shuffle``.
    """
    seen: Set[str] = set()
    kept = []
    for text in texts:
        text = text.strip() if isinstance(text, str) else ""
        if len(text.split()) < min_words:
            continue
        key = normalize(text)
        if key in seen:
            continue
        seen.add(key)
        kept.append(text)
    order = np.random.default_rng(seed).permutation(len(kept))
    return [kept[i] for i in order]


def find_hf_model(model: nn.Module) -> nn.Module:
    """The HuggingFace ``PreTrainedModel`` to adapt: ``model`` itself or its ``.model`` attribute."""
    from transformers import PreTrainedModel  # noqa: PLC0415

    if isinstance(model, PreTrainedModel):
        return model
    inner = getattr(model, "model", None)
    if isinstance(inner, PreTrainedModel):
        return inner
    raise TypeError("Fine-tuning needs a HuggingFace PreTrainedModel (the target itself or its `.model` attribute)")


def attach_lora(model: nn.Module, r: int, alpha: int, dropout: float, target_modules: Any) -> Any:  # noqa: ANN401
    """Inject LoRA adapters into the target's HuggingFace model in place; returns the ``PeftModel``."""
    try:
        from peft import LoraConfig, get_peft_model  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("Fine-tuning (Phase 3) needs `pip install peft`") from e
    config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target_modules, bias="none",
                        task_type="CAUSAL_LM")
    peft_model = get_peft_model(find_hf_model(model), config)
    # Adapter weights are fp32 master weights, also on a 16-bit base model:
    # peft runs the LoRA path in the adapter's dtype and casts the result back to the base model's dtype.
    for p in peft_model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    return peft_model


def param_dtypes(model: nn.Module, trainable: Optional[bool] = None) -> str:
    """The parameter dtypes of ``model`` as a sorted, comma-separated string (for logging).

    ``trainable`` restricts it to parameters that do (True) or do not (False) require gradients.
    """
    dtypes = {str(p.dtype).removeprefix("torch.") for p in model.parameters()
              if trainable is None or p.requires_grad == trainable}
    return ", ".join(sorted(dtypes)) or "none"


def detach_lora(peft_model: Any) -> None:  # noqa: ANN401
    """Remove the adapters again, restoring the original modules in place."""
    peft_model.unload()


@contextlib.contextmanager
def adapter_disabled(peft_model: Optional[Any]) -> Iterator[None]:  # noqa: ANN401
    """Run the block with the base model (no-op when ``peft_model`` is None)."""
    if peft_model is None:
        yield
    else:
        with peft_model.disable_adapter():
            yield


def lora_state(peft_model: Any) -> dict:  # noqa: ANN401
    """The adapter weights only."""
    from peft import get_peft_model_state_dict  # noqa: PLC0415

    return {k: v.detach().cpu() for k, v in get_peft_model_state_dict(peft_model).items()}


def load_lora_state(peft_model: Any, state: dict) -> None:  # noqa: ANN401
    """Load adapter weights saved with :func:`lora_state`."""
    from peft import set_peft_model_state_dict  # noqa: PLC0415

    set_peft_model_state_dict(peft_model, state)


def train_lora(
    peft_model: Any,  # noqa: ANN401
    texts: Sequence[str],
    tok: TokenizerLike,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 1,
    grad_accum: int = 8,
    lr: float = 1e-4,
    warmup_steps: int = 0,
    max_length: int = 256,
    seed: int = 42,
    progress_label: Optional[str] = None,
) -> List[float]:
    """Continual pre-training of the adapter on ``texts``; returns the loss of every optimizer step.

    Each text is encoded with the tokenizer's special tokens plus EOS, truncated to ``max_length``,
    right-padded within a batch, and trained with the causal LM loss on every non-pad token. AdamW at
    a constant learning rate after an optional linear warm-up; gradients are accumulated over
    ``grad_accum`` batches.

    On an fp16 base model the loss is scaled dynamically (as in fp16 mixed-precision training):
    steps with inf/NaN gradients are skipped and the scale is halved. Raises if the
    trained adapter weights are not finite.
    """
    collate = CausalLMCollate(pad_id_of(tok))
    encoded = _encode_with_eos(texts, tok, max_length)
    params = [p for p in peft_model.parameters() if p.requires_grad]
    scaler = _LossScaler(enabled=any(p.dtype == torch.float16 for p in peft_model.parameters()))
    optimizer = torch.optim.AdamW(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup_steps) if warmup_steps else 1.0)
    generator = torch.Generator().manual_seed(seed)
    progress = None
    if progress_label is not None:
        progress = Progress(progress_label, epochs * math.ceil(len(encoded) / batch_size), "batch")
    losses: List[float] = []
    peft_model.train()
    micro = 0
    running = 0.0
    for epoch in range(epochs):
        order = torch.randperm(len(encoded), generator=generator).tolist()
        for start in range(0, len(order), batch_size):
            batch = [encoded[i] for i in order[start:start + batch_size]]
            input_ids, attention_mask = collate([(ids, ids) for ids in batch])
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            labels = input_ids.masked_fill(attention_mask == 0, -100)
            loss = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            (loss * (scaler.scale / grad_accum)).backward()
            running += loss.item() / grad_accum
            micro += 1
            if progress is not None:
                progress.update()
            if micro % grad_accum == 0:
                if scaler.step(optimizer, params):
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(running)
                running = 0.0
        logger.info(f"LoRA epoch {epoch + 1}/{epochs}: last loss {losses[-1] if losses else float('nan'):.4f}")
    if micro % grad_accum:
        scaler.step(optimizer, params)
        optimizer.zero_grad(set_to_none=True)
        losses.append(running * grad_accum / (micro % grad_accum))
    peft_model.eval()
    if scaler.skipped:
        logger.info(f"LoRA: skipped {scaler.skipped} optimizer steps with inf/NaN gradients (fp16 loss scaling)")
    if not all(torch.isfinite(p).all() for p in params):
        raise RuntimeError("LoRA training produced inf/NaN adapter weights")
    return losses



def _encode_with_eos(texts: Sequence[str], tok: TokenizerLike, max_length: int) -> List[List[int]]:
    """Token ids of each text with the tokenizer's special tokens plus EOS, truncated to ``max_length``."""
    eos_id = tok.eos_token_id
    encoded = []
    for text in texts:
        ids = tok.encode(text, add_special_tokens=True)
        if eos_id is not None and (not ids or ids[-1] != eos_id):
            ids = [*ids, eos_id]
        if len(ids[:max_length]) >= 2:  # a single token gives no prediction to train on
            encoded.append(ids[:max_length])
    return encoded


class _LossScaler:
    """Device-independent dynamic loss scaling for training against an fp16 base model.

    The same policy as ``torch.amp.GradScaler`` (start at 2**16, halve on overflow, double after
    2000 clean steps), written out because GradScaler does not cover every backend (e.g. Gaudi).
    """

    def __init__(self, enabled: bool, init_scale: float = 2.0**16, growth_interval: int = 2000) -> None:
        self.scale = init_scale if enabled else 1.0
        self.enabled = enabled
        self.growth_interval = growth_interval
        self.clean_steps = 0
        self.skipped = 0

    def step(self, optimizer: torch.optim.Optimizer, params: Sequence[torch.Tensor]) -> bool:
        """Unscale the gradients and step; returns False (step skipped) when a gradient is inf/NaN."""
        if not self.enabled:
            optimizer.step()
            return True
        grads = [p.grad for p in params if p.grad is not None]
        if not all(torch.isfinite(g).all() for g in grads):
            self.scale /= 2
            self.clean_steps = 0
            self.skipped += 1
            return False
        for g in grads:
            g.div_(self.scale)
        optimizer.step()
        self.clean_steps += 1
        if self.clean_steps % self.growth_interval == 0:
            self.scale *= 2
        return True


class TokenWindowIndex:
    """Every ``window``-token window of a set of texts (normalized, no special tokens).

    Args:
    ----
        texts: The fine-tuning corpus.
        tok: Tokenizer of the audited model.
        window: Window length in tokens.

    """

    def __init__(self: Self, texts: Sequence[str], tok: TokenizerLike, window: int = 30) -> None:
        self.tok = tok
        self.window = window
        # A 64-bit hash per window instead of the tuple of token ids: the same test at a fraction of the
        # memory (Python hashes tuples of ints deterministically; a collision is ~1e-9 likely here).
        self.windows: Set[int] = set()
        for text in texts:
            ids = tok.encode(normalize(text), add_special_tokens=False)
            for i in range(len(ids) - window + 1):
                self.windows.add(hash(tuple(ids[i:i + window])))

    def overlaps(self: Self, text: str) -> bool:
        """Whether ``text`` shares any window with the indexed texts."""
        ids = self.tok.encode(normalize(text), add_special_tokens=False)
        return any(hash(tuple(ids[i:i + self.window])) in self.windows for i in range(len(ids) - self.window + 1))
