"""Phase 3-4 helpers: fine-tuning corpus, LoRA attach/train/detach, overlap index."""

import numpy as np
import pytest
import torch

from leakpro.attacks.utils.extraction.finetune import (
    TokenWindowIndex,
    adapter_disabled,
    attach_lora,
    build_finetune_corpus,
    detach_lora,
    load_lora_state,
    lora_state,
    param_dtypes,
    train_lora,
)
from leakpro.tests.extraction_attacks.fixtures import SECRET, CharTokenizer


def test_corpus_filters_dedupes_on_normalized_text_and_shuffles_like_datasets() -> None:
    long_a = " ".join(f"a{i}" for i in range(25))
    long_b = " ".join(f"b{i}" for i in range(25))
    long_c = " ".join(f"c{i}" for i in range(25))
    texts = [long_a, "too short", long_a.upper().replace(" ", "  \n"), f"  {long_b} ", long_c]
    corpus = build_finetune_corpus(texts, min_words=20, seed=42)
    kept = [long_a, long_b, long_c]
    assert corpus == [kept[i] for i in np.random.default_rng(42).permutation(3)]


def test_token_window_index() -> None:
    tok = CharTokenizer()
    index = TokenWindowIndex([SECRET[:50]], tok, window=10)
    assert index.overlaps("prefix " + SECRET[20:30].upper() + " suffix")
    assert not index.overlaps(SECRET[100:200])
    assert not index.overlaps("short")


def _tiny_model():  # noqa: ANN202
    pytest.importorskip("peft")
    from leakpro.tests.extraction_attacks.fixtures import TinyGPT2LM

    return TinyGPT2LM().eval()


def _logits(model: torch.nn.Module, ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=ids).logits


def test_lora_trains_toggles_and_detaches_in_place() -> None:
    model = _tiny_model()
    tok = CharTokenizer()
    ids = torch.tensor([tok.encode(SECRET[:40])])
    before = _logits(model, ids)

    peft_model = attach_lora(model, r=4, alpha=8, dropout=0.0, target_modules=["c_attn"])
    losses = train_lora(peft_model, [SECRET[:120]] * 4, tok, torch.device("cpu"), epochs=10, grad_accum=2, lr=1e-2,
                        max_length=128, seed=0)
    assert len(losses) == 20 and losses[-1] < losses[0]

    tuned = _logits(model, ids)  # the wrapper itself now runs with the adapter
    assert not torch.allclose(tuned, before)
    with adapter_disabled(peft_model):
        assert torch.allclose(_logits(model, ids), before, atol=1e-6)

    state = lora_state(peft_model)
    detach_lora(peft_model)
    assert torch.allclose(_logits(model, ids), before, atol=1e-6)

    again = attach_lora(model, r=4, alpha=8, dropout=0.0, target_modules=["c_attn"])
    load_lora_state(again, state)
    assert torch.allclose(_logits(model, ids), tuned, atol=1e-6)
    detach_lora(again)


def test_attach_lora_rejects_non_huggingface_models() -> None:
    pytest.importorskip("peft")
    from leakpro.tests.extraction_attacks.fixtures import ParrotLM

    with pytest.raises(TypeError, match="PreTrainedModel"):
        attach_lora(ParrotLM(), r=2, alpha=4, dropout=0.0, target_modules=["x"])


def test_fp16_base_model_trains_fp32_adapters_with_loss_scaling() -> None:
    model = _tiny_model().half()
    tok = CharTokenizer()
    peft_model = attach_lora(model, r=4, alpha=8, dropout=0.0, target_modules=["c_attn"])
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    assert trainable and all(p.dtype == torch.float32 for p in trainable)
    assert all(p.dtype == torch.float16 for p in peft_model.parameters() if not p.requires_grad)

    losses = train_lora(peft_model, [SECRET[:120]] * 4, tok, torch.device("cpu"), epochs=10, grad_accum=2, lr=1e-2,
                        max_length=128, seed=0)
    assert losses[-1] < losses[0]
    assert all(torch.isfinite(p).all() for p in trainable)
    ids = torch.tensor([tok.encode(SECRET[:40])])
    assert _logits(model, ids).dtype == torch.float16  # the LoRA path is cast back to the base dtype
    detach_lora(peft_model)


def test_loss_scaler_skips_overflowing_steps() -> None:
    from leakpro.attacks.utils.extraction.finetune import _LossScaler

    p = torch.nn.Parameter(torch.ones(2))
    opt = torch.optim.SGD([p], lr=1.0)
    scaler = _LossScaler(enabled=True, init_scale=4.0, growth_interval=1)
    p.grad = torch.tensor([float("inf"), 1.0])
    assert not scaler.step(opt, [p]) and scaler.scale == 2.0 and torch.equal(p.data, torch.ones(2))
    p.grad = torch.tensor([2.0, 2.0])  # = true gradient 1.0 at scale 2
    assert scaler.step(opt, [p]) and torch.equal(p.data, torch.zeros(2)) and scaler.scale == 4.0


def test_param_dtypes() -> None:
    model = _tiny_model().to(torch.bfloat16)
    assert param_dtypes(model) == "bfloat16"
    peft_model = attach_lora(model, r=2, alpha=4, dropout=0.0, target_modules=["c_attn"])
    assert param_dtypes(peft_model) == "bfloat16, float32"
    assert param_dtypes(peft_model, trainable=False) == "bfloat16"
    assert param_dtypes(peft_model, trainable=True) == "float32"
    detach_lora(peft_model)
