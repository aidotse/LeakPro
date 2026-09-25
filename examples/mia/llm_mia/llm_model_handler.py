"""Model handler for LLM membership inference: fine-tune (full or LoRA) and evaluate a causal LM.

Usage:
    from llm_data_handler import LLMDataHandler
    from llm_model_handler import LLMModelHandler
    leakpro = LeakPro(LLMDataHandler, "audit.yaml", model_handler=LLMModelHandler)

``train`` / ``eval`` are only used by ``prepare_target.py`` to build the target; the LLM attacks
themselves train nothing. Both accept the ``(input_ids, input_ids)`` batches the data handler yields
(always built with ``CausalLMCollate`` in ``prepare_target.py``, so batches are ``(input_ids, attention_mask)``).
"""

import shutil
from pathlib import Path
from typing import Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.schemas import EvalOutput, TrainingOutput
from leakpro.utils.device import get_device, mark_step


def _unpack(batch: tuple) -> tuple:
    """Return (input_ids, attention_mask). Every loader in this example uses CausalLMCollate, which yields exactly that."""
    input_ids, attention_mask = batch
    return input_ids, attention_mask.long()


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


def _run_eval_pass(dataloader: DataLoader, model: nn.Module, criterion: nn.Module, device: torch.device) -> EvalOutput:
    """Shared by `eval()` and mid-training checks; leaves the model on `device` (caller's responsibility to move it)."""
    model.eval()
    tot_loss, tot_correct, tot_tokens = 0.0, 0, 0
    with torch.no_grad():
        for batch in dataloader:
            ids, mask = _unpack(batch)
            ids, mask = ids.to(device), mask.to(device)
            out = model(input_ids=ids, attention_mask=mask)
            loss, correct, n_tok = _next_token_loss_and_acc(out.logits, ids, mask, criterion)
            mark_step(device)
            tot_loss += loss.item() * n_tok.item()
            tot_correct += correct.item()
            tot_tokens += n_tok.item()
    return EvalOutput(accuracy=tot_correct / max(tot_tokens, 1), loss=tot_loss / max(tot_tokens, 1))


def _linear_warmup_then_decay(step: int, num_warmup_steps: int, num_training_steps: int) -> float:
    """LR multiplier: linear 0->1 over the warmup, then linear 1->0 over the rest -- HF Trainer's default schedule."""
    if step < num_warmup_steps:
        return step / max(1, num_warmup_steps)
    remaining = num_training_steps - num_warmup_steps
    return max(0.0, (num_training_steps - step) / max(1, remaining))


def _prune_old_checkpoints(checkpoint_dir: Path, save_total_limit: int) -> None:
    """Keep only the `save_total_limit` most recently written `checkpoint-*` directories."""
    checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
    for stale in checkpoints[:-save_total_limit] if save_total_limit > 0 else checkpoints:
        shutil.rmtree(stale)


class LLMModelHandler(AbstractInputHandler, role="model"):
    """Fine-tuning and evaluation for a HuggingFace causal LM behind ``HFCausalLMWrapper``."""

    lora: dict = None  # {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": [...]} or None

    def train(  # noqa: PLR0913
        self,
        dataloader: DataLoader,
        model: nn.Module = None,
        criterion: nn.Module = None,
        optimizer: optim.Optimizer = None,
        epochs: int = None,
        gradient_accumulation_steps: int = 1,
        warmup_steps: int = 0,
        eval_dataloader: Optional[DataLoader] = None,
        checkpoint_dir: Optional[str] = None,
        save_total_limit: int = 1,
        load_best_checkpoint_at_end: bool = False,
    ) -> TrainingOutput:
        """Fine-tune `model`.

        Args:
        ----
            dataloader: Training batches.
            model: The (unwrapped) HFCausalLMWrapper to fine-tune.
            criterion: Loss function (next-token cross-entropy).
            optimizer: Built over `model.parameters()` (or rebuilt over the LoRA adapter's, below).
            epochs: Number of passes over `dataloader`.
            gradient_accumulation_steps: Micro-batches accumulated before each optimizer step -- lets
                a big model use a small per-step `batch_size` while keeping a larger effective batch
                size (`batch_size * gradient_accumulation_steps`), matching a paper's own recipe.
            warmup_steps: Linear LR warmup over this many *optimizer* steps (not micro-batches), then
                linear decay to 0 over the remaining steps -- matches HF `Trainer`'s default schedule
                when its `warmup_steps` is set. 0 (default) keeps the LR constant, as before.
            eval_dataloader: If given, run a held-out eval pass after every epoch (`eval_strategy:
                epoch`) and print/record it in `history`. `None` (default) evaluates only once, at
                the very end via a separate `self.eval(...)` call -- unchanged from before.
            checkpoint_dir: If given, save the model after every epoch to
                `<checkpoint_dir>/checkpoint-<global_step>/model.pkl` (`save_strategy: epoch`),
                pruning down to `save_total_limit` afterwards.
            save_total_limit: How many of the newest checkpoints under `checkpoint_dir` to keep.
            load_best_checkpoint_at_end: If True, restore the epoch with the lowest `eval_dataloader`
                loss into `model` before returning, instead of leaving whatever the last epoch
                produced. Matches the EZ-MIA paper's own protocol (Appendix A.2): it explicitly
                selects the checkpoint with the lowest validation loss "to mitigate overfitting
                artifacts that could confound membership signals" -- an over-trained target is
                easier to attack for reasons that have nothing to do with the attack itself.
                Requires `eval_dataloader`. Tracked in memory (not by re-reading `checkpoint_dir`),
                so it works even when `checkpoint_dir` is None or `save_total_limit` would otherwise
                have pruned the best epoch's file away.

        """
        if epochs is None:
            raise ValueError("epochs not found in configs")
        if gradient_accumulation_steps < 1:
            raise ValueError(f"gradient_accumulation_steps must be >= 1, got {gradient_accumulation_steps}")
        if load_best_checkpoint_at_end and eval_dataloader is None:
            raise ValueError("load_best_checkpoint_at_end requires eval_dataloader (eval_strategy: epoch)")
        device = get_device()

        if self.lora:
            from peft import LoraConfig, get_peft_model

            cfg = LoraConfig(r=self.lora["r"], lora_alpha=self.lora["alpha"], lora_dropout=self.lora["dropout"],
                             target_modules=self.lora["target_modules"], task_type="CAUSAL_LM")
            model.model = get_peft_model(model.model, cfg)
            # The optimizer was built over the base parameters; rebuild it over the trainable adapter ones.
            # Only lr/weight_decay are ever set explicitly (see prepare_target.py) -- splatting the rest of
            # `.defaults` is fragile: some torch versions' AdamW.defaults includes keys (e.g.
            # decoupled_weight_decay) that its own __init__ does not accept as a kwarg.
            optimizer = type(optimizer)(
                (p for p in model.parameters() if p.requires_grad),
                lr=optimizer.defaults["lr"], weight_decay=optimizer.defaults["weight_decay"],
            )

        n_batches = len(dataloader)
        steps_per_epoch = -(-n_batches // gradient_accumulation_steps)  # ceil div
        total_optimizer_steps = steps_per_epoch * epochs
        scheduler = (
            optim.lr_scheduler.LambdaLR(
                optimizer, lambda s: _linear_warmup_then_decay(s, warmup_steps, total_optimizer_steps))
            if warmup_steps > 0 else None
        )

        model.to(device)
        history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
        global_step = 0
        best_val_loss, best_epoch, best_state_dict = float("inf"), None, None
        for epoch in range(epochs):
            model.train()
            tot_loss, tot_correct, tot_tokens = 0.0, 0, 0
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
                ids, mask = _unpack(batch)
                ids, mask = ids.to(device), mask.to(device)
                out = model(input_ids=ids, attention_mask=mask)
                loss, correct, n_tok = _next_token_loss_and_acc(out.logits, ids, mask, criterion)
                # Scaled so accumulated gradients approximate the mean over the larger effective batch
                # (gradient_accumulation_steps micro-batches of `batch_size` rows), not their sum.
                (loss / gradient_accumulation_steps).backward()
                is_last_batch = (step + 1) == n_batches
                if (step + 1) % gradient_accumulation_steps == 0 or is_last_batch:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()
                    global_step += 1
                mark_step(device)
                tot_loss += loss.item() * n_tok.item()
                tot_correct += correct.item()
                tot_tokens += n_tok.item()
            history["loss"].append(tot_loss / max(tot_tokens, 1))
            history["acc"].append(tot_correct / max(tot_tokens, 1))
            print(f"epoch {epoch + 1}: loss {history['loss'][-1]:.4f}  next-token acc {history['acc'][-1]:.4f}")

            if eval_dataloader is not None:
                eval_result = _run_eval_pass(eval_dataloader, model, criterion, device)
                model.train()  # _run_eval_pass leaves the model in eval() mode
                history["val_loss"].append(eval_result.loss)
                history["val_acc"].append(eval_result.accuracy)
                print(f"epoch {epoch + 1}: eval loss {eval_result.loss:.4f}  eval next-token acc {eval_result.accuracy:.4f}")
                if load_best_checkpoint_at_end and eval_result.loss < best_val_loss:
                    best_val_loss, best_epoch = eval_result.loss, epoch + 1
                    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if checkpoint_dir is not None:
                ckpt_dir = Path(checkpoint_dir)
                ckpt_path = ckpt_dir / f"checkpoint-{global_step}"
                ckpt_path.mkdir(parents=True, exist_ok=True)
                torch.save({k: v.cpu() for k, v in model.state_dict().items()}, ckpt_path / "model.pkl")
                _prune_old_checkpoints(ckpt_dir, save_total_limit)

        if load_best_checkpoint_at_end and best_state_dict is not None and best_epoch != epochs:
            print(f"restoring epoch {best_epoch} (eval loss {best_val_loss:.4f}), "
                  f"the best of {epochs} epochs, instead of the last epoch's weights")
            model.load_state_dict(best_state_dict)

        if self.lora:
            # Fold the adapters back in so the saved state dict matches a plain HFCausalLMWrapper.
            model.model = model.model.merge_and_unload()

        model.to("cpu")
        metrics = EvalOutput(
            accuracy=history["acc"][-1], loss=history["loss"][-1],
            extra={"loss_history": history["loss"], "acc_history": history["acc"],
                  "val_loss_history": history["val_loss"], "val_acc_history": history["val_acc"],
                  "best_epoch": best_epoch, "best_val_loss": best_val_loss if best_epoch is not None else None},
        )
        return TrainingOutput(model=model, metrics=metrics)

    def eval(self, dataloader: DataLoader, model: nn.Module, criterion: nn.Module) -> EvalOutput:
        device = get_device()
        model.to(device)
        result = _run_eval_pass(dataloader, model, criterion, device)
        model.to("cpu")
        return result
