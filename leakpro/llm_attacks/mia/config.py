#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import argparse
from pathlib import Path
from omegaconf import OmegaConf


AttackRefVariant = Literal["base", "distillation", "sft"]

@dataclass
class AttackConfig:
    dataset: str
    ref_variant: AttackRefVariant

    seed: int = 42
    save_artifacts_path: str | None = None

    domain_dataset: str | None = None

    train_total: int = 20000
    eval_total: int = 20000

    epochs: int = 3
    batch_size: int = 16
    lr: float = 5e-5
    sequence_length: int = 128

    model_name: str = "gpt2"
    finetune_method: Literal["auto", "full", "lora"] = "auto"

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None

    val_total: int = 500

    distil_max_prompts: int = 1000
    distil_completions: int = 1
    distil_max_new_tokens: int = 64
    distil_temperature: float = 0.9
    distil_top_p: float = 0.9
    distil_input_max_tokens: int = 128
    distil_train_epochs: int = 4
    distil_train_batch: int = 16
    distil_train_lr: float = 1e-4

    sft_train_epochs: int = 1
    sft_train_batch: int = 16
    sft_train_lr: float = 1e-4


def make_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for running experiments (single run)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="Path to YAML config; when provided, runs a single experiment using YAML values")
    ap.add_argument(
        "--dataset",
        choices=[
            "ag_news",
            "xsum",
            "wikitext",
            "tokyotech-llm/swallow-code",
        ],
        default="ag_news",
    )
    ap.add_argument("--ref-variant", choices=["base", "distillation", "sft"], default="base")
    ap.add_argument(
        "--target-model",
        choices=[
            "gpt2",
            "EleutherAI/gpt-j-6B",
            "meta-llama/Llama-2-7b-hf",
            "codellama/CodeLlama-7b-hf",
        ],
        default="gpt2",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-total", type=int, default=8000)
    ap.add_argument("--eval-total", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--sequence-length", type=int, default=128, help="Exact whitespace-token length for each training/eval example (truncates long samples, concatenates consecutive samples when short)")

    ap.add_argument("--val-total", type=int, default=500, help="Number of validation datapoints sampled from the original dataset (independent of train/eval subsets)")

    ap.add_argument("--finetune-method", choices=["auto", "full", "lora"], default="auto")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)

    ap.add_argument("--distil-max-prompts", type=int, default=1000)
    ap.add_argument("--distil-completions", type=int, default=1)
    ap.add_argument("--distil-max-new-tokens", type=int, default=64)
    ap.add_argument("--distil-temperature", type=float, default=0.9)
    ap.add_argument("--distil-top-p", type=float, default=0.9)
    ap.add_argument("--distil-input-max-tokens", type=int, default=128)
    ap.add_argument("--distil-train-epochs", type=int, default=4)
    ap.add_argument("--distil-train-batch", type=int, default=16)
    ap.add_argument("--distil-train-lr", type=float, default=1e-4)

    ap.add_argument("--sft-train-epochs", type=int, default=1)
    ap.add_argument("--sft-train-batch", type=int, default=16)
    ap.add_argument("--sft-train-lr", type=float, default=1e-4)

    ap.add_argument("--save-artifacts-path", type=str, default=None,
                    help="Path to save models and data for later use with verification_harness.py (disabled by default)")
    return ap


def load_attack_config_from_yaml(path: str) -> AttackConfig:
    """Load AttackConfig from a YAML file using OmegaConf.

    The YAML may contain any AttackConfig fields; missing fields use dataclass defaults.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config YAML not found: {path}")
    oc = OmegaConf.load(str(cfg_path))
    data = OmegaConf.to_container(oc, resolve=True) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping of AttackConfig fields")
    return AttackConfig(**data)
