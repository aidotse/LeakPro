"""Tiny models and a character tokenizer for extraction tests (CPU only, no downloads).

This file is also the ``target.module_path`` of the end-to-end tests, so every class here can be
constructed without arguments.
"""

from types import SimpleNamespace
from typing import List, Sequence

import torch
from torch import Tensor, nn

from leakpro.input_handler.abstract_input_handler import AbstractInputHandler

PAD, BOS, EOS = 0, 1, 2
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 .,"
VOCAB = len(CHARS) + 3

# 60 words. The parrot model below reproduces it whatever it is prompted with.
SECRET = " ".join(f"w{i}x" for i in range(60))


class CharTokenizer:
    """One token per character; ids 0-2 are pad, BOS and EOS."""

    pad_token_id = PAD
    bos_token_id = BOS
    eos_token_id = EOS
    name_or_path = "char-tokenizer"

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Characters outside the vocabulary are dropped."""
        ids = [CHARS.index(ch) + 3 for ch in text.lower() if ch in CHARS]
        return [BOS, *ids] if add_special_tokens else ids

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        """Special ids decode to nothing when skipped, else to markers."""
        out = []
        for i in token_ids:
            if i >= 3:
                out.append(CHARS[i - 3])
            elif not skip_special_tokens:
                out.append(["<pad>", "<s>", "</s>"][i])
        return "".join(out)


SECRET_IDS = CharTokenizer().encode(SECRET, add_special_tokens=False)


class ParrotLM(nn.Module):
    """Predicts the next character of ``SECRET`` with near-certainty, then EOS.

    The prediction at a position depends only on how many real (non-pad, non-BOS) tokens precede
    it, so left padding, BOS prompts and prefix prompts all continue ``SECRET`` correctly as long as
    a prefix prompt is itself a prefix of ``SECRET``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))  # gives the model a state_dict to fingerprint

    def forward(self, input_ids: Tensor, attention_mask: Tensor = None) -> SimpleNamespace:
        """Return ``(B, L, VOCAB)`` logits."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        real = (attention_mask.bool() & (input_ids != BOS)).long()
        count = real.cumsum(dim=-1).clamp(max=len(SECRET_IDS))
        table = torch.tensor([*SECRET_IDS, EOS], device=input_ids.device)
        target = table[count]
        logits = torch.full((*input_ids.shape, VOCAB), -1e4, device=input_ids.device) + self.bias
        logits.scatter_(-1, target.unsqueeze(-1), 1e4)
        return SimpleNamespace(logits=logits)


class TinyLM(nn.Module):
    """Small random causal LM (embedding, causal prefix sum, linear head)."""

    def __init__(self, vocab: int = VOCAB, dim: int = 8) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, input_ids: Tensor, attention_mask: Tensor = None) -> SimpleNamespace:
        """Logits from the masked prefix sum of embeddings."""
        emb = self.embed(input_ids)
        if attention_mask is not None:
            emb = emb * attention_mask.unsqueeze(-1).to(emb.dtype)
        return SimpleNamespace(logits=self.head(torch.cumsum(emb, dim=1)))


class ExtractionInputHandler(AbstractInputHandler, role="model"):
    """User handler for the end-to-end tests: only supplies the tokenizer."""

    def get_tokenizer(self) -> CharTokenizer:
        """The character tokenizer."""
        return CharTokenizer()

    def train(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003, D102
        raise NotImplementedError

    def eval(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003, D102
        raise NotImplementedError


class TinyGPT2LM(nn.Module):
    """A one-layer random GPT-2 over the character vocabulary, wrapped like ``HFCausalLMWrapper``.

    Used where a real HuggingFace model is needed (LoRA fine-tuning). Requires ``transformers``.
    """

    def __init__(self) -> None:
        super().__init__()
        from transformers import GPT2Config, GPT2LMHeadModel

        torch.manual_seed(0)
        config = GPT2Config(vocab_size=VOCAB, n_positions=512, n_embd=16, n_layer=1, n_head=2,
                            bos_token_id=BOS, eos_token_id=EOS, pad_token_id=PAD)
        self.model = GPT2LMHeadModel(config)

        # Fixed logit bias: frequent spaces (so texts have words) and no special tokens (so the
        # torch backend never stops early). Applied in forward(), i.e. with generation_backend: torch.
        bias = torch.zeros(VOCAB)
        bias[[PAD, BOS, EOS]] = -1e4
        bias[CHARS.index(" ") + 3] = 3.0
        self.register_buffer("logit_bias", bias)

    def forward(self, input_ids: Tensor, attention_mask: Tensor = None) -> SimpleNamespace:
        """Logits of the wrapped model plus the fixed bias."""
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return SimpleNamespace(logits=out.logits + self.logit_bias)
