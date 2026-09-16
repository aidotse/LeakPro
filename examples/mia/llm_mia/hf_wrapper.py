"""Thin nn.Module around a HuggingFace causal LM so LeakPro can reconstruct it from metadata.

LeakPro rebuilds the target as ``blueprint(**init_params)`` and then ``load_state_dict``s the saved
weights (``MIAHandler._load_trained_target_model``). ``init_params`` are recovered by
``get_model_init_params``, which reads instance attributes named after the ``__init__`` parameters —
so every constructor argument is stored under its own name.

For LoRA-fine-tuned targets, merge the adapters (``peft ... merge_and_unload()``) *before* saving,
so the saved state dict matches this plain wrapper.
"""

import torch
from torch import nn

from leakpro.attacks.mia_attacks.llm.abstract_llm_mia import load_pretrained_causal_lm


class HFCausalLMWrapper(nn.Module):
    """Load an ``AutoModelForCausalLM`` and expose the ``(input_ids, attention_mask)`` forward LeakPro expects."""

    def __init__(self, pretrained_name_or_path: str, dtype: str = "float32") -> None:
        super().__init__()
        self.pretrained_name_or_path = pretrained_name_or_path
        self.dtype = dtype
        # Uncached loader (this module gets fine-tuned); the transformers version shim lives in one place.
        self.model = load_pretrained_causal_lm(pretrained_name_or_path, dtype)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):  # noqa: ANN201
        """Return the HF output object; ``CausalLMModel`` reads ``.logits`` from it."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask)
