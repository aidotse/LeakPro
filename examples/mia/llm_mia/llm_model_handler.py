"""Model handler for LLM membership inference: fine-tune (full or LoRA) and evaluate a causal LM.

Usage:
    from llm_data_handler import LLMDataHandler
    from llm_model_handler import LLMModelHandler
    leakpro = LeakPro(LLMDataHandler, "audit.yaml", model_handler=LLMModelHandler)

``train`` / ``eval`` are only used by ``prepare_target.py`` to build the target; the LLM attacks
themselves train nothing. Both accept the ``(input_ids, input_ids)`` batches the data handler yields
(via default collate for fixed-length data, or ``CausalLMCollate`` for variable length, which yields
``(input_ids, attention_mask)``).
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.schemas import EvalOutput, TrainingOutput
from leakpro.utils.device import get_device, mark_step


def _unpack(batch: tuple, pad_token_id: int) -> tuple:
    """Return (input_ids, attention_mask) from either (ids, ids) or (ids, mask) batches."""
    first, second = batch
    if second.dtype == torch.bool or (second.max() <= 1 and second.min() >= 0 and not torch.equal(first, second)):
        return first, second.long()
    return first, (first != pad_token_id).long()


def _next_token_loss_and_acc(logits: torch.Tensor, input_ids: torch.Tensor, mask: torch.Tensor,
                             criterion: nn.Module) -> tuple:
    """Shifted next-token CE (padding ignored) and top-1 accuracy over valid positions."""
    logits = logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    valid = mask[:, 1:].bool()
    labels = torch.where(valid, targets, torch.full_like(targets, -100))
    loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    correct = ((logits.argmax(-1) == targets) & valid).sum()
    return loss, correct, valid.sum()


class LLMModelHandler(AbstractInputHandler, role="model"):
    """Fine-tuning and evaluation for a HuggingFace causal LM behind ``HFCausalLMWrapper``."""

    pad_token_id: int = 50256  # GPT-2 eos; prepare_target.py overrides this class attribute
    lora: dict = None          # {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": [...]} or None

    def train(
        self,
        dataloader: DataLoader,
        model: nn.Module = None,
        criterion: nn.Module = None,
        optimizer: optim.Optimizer = None,
        epochs: int = None,
    ) -> TrainingOutput:
        if epochs is None:
            raise ValueError("epochs not found in configs")
        device = get_device()

        if self.lora:
            from peft import LoraConfig, get_peft_model

            cfg = LoraConfig(r=self.lora["r"], lora_alpha=self.lora["alpha"], lora_dropout=self.lora["dropout"],
                             target_modules=self.lora["target_modules"], task_type="CAUSAL_LM")
            model.model = get_peft_model(model.model, cfg)
            # The optimizer was built over the base parameters; rebuild it over the trainable adapter ones.
            optimizer = type(optimizer)((p for p in model.parameters() if p.requires_grad), **optimizer.defaults)

        model.to(device)
        history = {"loss": [], "acc": []}
        for epoch in range(epochs):
            model.train()
            tot_loss, tot_correct, tot_tokens = 0.0, 0, 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
                ids, mask = _unpack(batch, self.pad_token_id)
                ids, mask = ids.to(device), mask.to(device)
                optimizer.zero_grad(set_to_none=True)
                out = model(input_ids=ids, attention_mask=mask)
                loss, correct, n_tok = _next_token_loss_and_acc(out.logits, ids, mask, criterion)
                loss.backward()
                optimizer.step()
                mark_step(device)
                tot_loss += loss.item() * n_tok.item()
                tot_correct += correct.item()
                tot_tokens += n_tok.item()
            history["loss"].append(tot_loss / max(tot_tokens, 1))
            history["acc"].append(tot_correct / max(tot_tokens, 1))
            print(f"epoch {epoch + 1}: loss {history['loss'][-1]:.4f}  next-token acc {history['acc'][-1]:.4f}")

        if self.lora:
            # Fold the adapters back in so the saved state dict matches a plain HFCausalLMWrapper.
            model.model = model.model.merge_and_unload()

        model.to("cpu")
        metrics = EvalOutput(accuracy=history["acc"][-1], loss=history["loss"][-1],
                             extra={"loss_history": history["loss"], "acc_history": history["acc"]})
        return TrainingOutput(model=model, metrics=metrics)

    def eval(self, dataloader: DataLoader, model: nn.Module, criterion: nn.Module) -> EvalOutput:
        device = get_device()
        model.to(device)
        model.eval()
        tot_loss, tot_correct, tot_tokens = 0.0, 0, 0
        with torch.no_grad():
            for batch in dataloader:
                ids, mask = _unpack(batch, self.pad_token_id)
                ids, mask = ids.to(device), mask.to(device)
                out = model(input_ids=ids, attention_mask=mask)
                loss, correct, n_tok = _next_token_loss_and_acc(out.logits, ids, mask, criterion)
                mark_step(device)
                tot_loss += loss.item() * n_tok.item()
                tot_correct += correct.item()
                tot_tokens += n_tok.item()
        model.to("cpu")
        return EvalOutput(accuracy=tot_correct / max(tot_tokens, 1), loss=tot_loss / max(tot_tokens, 1))
