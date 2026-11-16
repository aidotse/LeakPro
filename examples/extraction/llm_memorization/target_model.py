"""Target models for the memorization extraction example.

LeakPro builds the target as ``model_class()`` (no arguments) when ``target.target_folder`` holds no
saved LeakPro model, so each class below loads its own checkpoint. The tokenizer is loaded from
``pretrained_name_or_path`` as well.
"""

import torch
from torch import nn

from leakpro.attacks.mia_attacks.llm.abstract_llm_mia import load_pretrained_causal_lm
from leakpro.utils.device import get_device
from leakpro.utils.logger import logger


def model_dtype() -> str:
    """Weight dtype for the device the attack runs on, used in every phase.

    bfloat16 on Gaudi (HPU) and on NVIDIA GPUs with native bfloat16 (Ampere and newer: A100, L4,
    RTX 30xx, H100, ...); float16 on older GPUs (T4, V100), where the LoRA training of Phase 3 then
    uses loss scaling; float32 on CPU.
    """
    device = get_device()
    if device.type == "hpu":
        return "bfloat16"
    if device.type == "cuda":
        try:
            native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:  # torch < 2.4 has no `including_emulation` (and reports native support only)
            native_bf16 = torch.cuda.is_bf16_supported()
        return "bfloat16" if native_bf16 else "float16"
    return "float32"


class HFCausalLM(nn.Module):
    """A HuggingFace causal LM exposed as ``.model`` (used for ``generate()`` and LoRA).

    To audit another model, subclass it with that model's ``pretrained_name_or_path`` and set
    ``target.model_class`` in audit.yaml to the subclass.
    """

    pretrained_name_or_path: str  # set by each subclass

    def __init__(self) -> None:
        super().__init__()
        dtype = model_dtype()
        logger.info(f"Loading {self.pretrained_name_or_path} in {dtype} for {get_device()}")
        self.model = load_pretrained_causal_lm(self.pretrained_name_or_path, dtype)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):  # noqa: ANN201
        """Return the model output (with ``.logits``)."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask)


class Llama31_8B(HFCausalLM):  # noqa: N801
    """Llama-3.1-8B. Gated: request access on Hugging Face, then `huggingface-cli login`."""

    pretrained_name_or_path = "meta-llama/Llama-3.1-8B"
