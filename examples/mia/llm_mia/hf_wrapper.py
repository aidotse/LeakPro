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


class HFCausalLMWrapper(nn.Module):
    """Load an ``AutoModelForCausalLM`` and expose the ``(input_ids, attention_mask)`` forward LeakPro expects."""

    def __init__(self, pretrained_name_or_path: str, dtype: str = "float32") -> None:
        super().__init__()
        self.pretrained_name_or_path = pretrained_name_or_path
        self.dtype = dtype
        from transformers import AutoModelForCausalLM

        torch_dtype = getattr(torch, dtype)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(pretrained_name_or_path, dtype=torch_dtype)
        except TypeError:  # transformers < 5 spells the argument torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(pretrained_name_or_path, torch_dtype=torch_dtype)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):  # noqa: ANN201
        """Return the HF output object; ``CausalLMModel`` reads ``.logits`` from it."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask)
