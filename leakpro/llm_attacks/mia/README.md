# EZ-MIA: Error Zone Membership Inference Attack

A membership inference attack (MIA) framework for large language models (LLMs) that uses error-zone analysis to determine whether specific text samples were part of a model's training data.

This module is a standalone port of the original [ez-mia](https://github.com/) research tool into the LeakPro repository. It is **not** wired into LeakPro's `AbstractMIA`/`AttackFactoryMIA` plugin system: LeakPro's classic MIA attacks audit a pre-trained classifier via shadow models, while this attack trains its own target LLM and a reference model as part of the experiment, and scores membership from per-token log-probabilities rather than fixed-size class logits. It runs as its own CLI/library, reusing LeakPro's device detection (`leakpro.utils.device`) so it works the same way on Habana Gaudi (HPU), NVIDIA GPU (CUDA), or CPU.

## Overview

EZ-MIA exploits the observation that fine-tuned language models exhibit distinct behavior on tokens where they make prediction errors. By comparing a target model against a reference model, we can compute membership scores that reveal training data membership.

The attack computes a ratio of positive vs. negative log-probability differences specifically on "error-zone" tokens—positions where the target model's top-1 prediction doesn't match the ground truth.

## Installation

Install LeakPro with the `llm-mia` extra:

```bash
pip install -e ".[llm-mia]"
```

For Habana Gaudi (HPU) hosts, also install the `hpu` extra (see the repository root README / `python -m leakpro.utils.hardware_setup`).

### Key Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework (device selection delegated to `leakpro.utils.device`) |
| `transformers` | Hugging Face model loading and training |
| `peft` | Parameter-efficient fine-tuning (LoRA) |
| `datasets` | Dataset loading from Hugging Face Hub |
| `scikit-learn` | Evaluation metrics (AUC, ROC) |
| `omegaconf` | YAML configuration parsing |
| `tqdm` | Progress bars |
| `accelerate` | Efficient model loading |

## Quick Start

### Using Command Line Arguments

```bash
python -m leakpro.llm_attacks.mia --dataset ag_news --ref-variant base --target-model gpt2
```

### Using YAML Configuration

```bash
python -m leakpro.llm_attacks.mia --config leakpro/llm_attacks/mia/configs/exp_ag_news_gpt2_base.yaml
```

### Device selection

The attack automatically picks Habana HPU, then CUDA, then CPU via `leakpro.utils.device.get_device()`. To force a specific device (e.g. for debugging on a mixed HPU+CUDA machine):

```bash
LEAKPRO_DEVICE=cpu python -m leakpro.llm_attacks.mia --dataset ag_news
```

## Supported Datasets

| Dataset | Description |
|---------|-------------|
| `ag_news` | AG News classification dataset |
| `xsum` | Extreme summarization dataset |
| `wikitext` | WikiText language modeling dataset |
| `tokyotech-llm/swallow-code` | Code dataset (Python) |

## Supported Models

| Model | Identifier |
|-------|------------|
| GPT-2 | `gpt2` |
| GPT-J 6B | `EleutherAI/gpt-j-6B` |
| Llama 2 7B | `meta-llama/Llama-2-7b-hf` |
| CodeLlama 7B | `codellama/CodeLlama-7b-hf` |

## Reference Model Variants

EZ-MIA supports three strategies for building the reference model:

| Variant | Description |
|---------|-------------|
| `base` | Uses the pre-trained base model directly as reference |
| `distillation` | Distills knowledge from the target model using generated completions |
| `sft` | Supervised fine-tuning on domain-specific non-member data |

## Command Line Arguments

### Core Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | None | Path to YAML config file |
| `--dataset` | `ag_news` | Dataset to use |
| `--ref-variant` | `base` | Reference model strategy (`base`, `distillation`, `sft`) |
| `--target-model` | `gpt2` | Target model to attack |
| `--seed` | `42` | Random seed |

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--train-total` | `8000` | Number of training samples |
| `--eval-total` | `2000` | Number of evaluation samples |
| `--val-total` | `500` | Number of validation samples |
| `--epochs` | `3` | Training epochs |
| `--batch-size` | `8` | Batch size |
| `--lr` | `5e-5` | Learning rate |
| `--sequence-length` | `128` | Token sequence length |

### Fine-tuning Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--finetune-method` | `auto` | Fine-tuning method (`auto`, `full`, `lora`) |
| `--lora-r` | `8` | LoRA rank |
| `--lora-alpha` | `16` | LoRA alpha |
| `--lora-dropout` | `0.05` | LoRA dropout |

### Distillation Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--distil-max-prompts` | `1000` | Max prompts for distillation |
| `--distil-completions` | `1` | Completions per prompt |
| `--distil-max-new-tokens` | `64` | Max new tokens to generate |
| `--distil-temperature` | `0.9` | Sampling temperature |
| `--distil-top-p` | `0.9` | Top-p sampling |
| `--distil-train-epochs` | `4` | Distillation training epochs |
| `--distil-train-batch` | `16` | Distillation batch size |
| `--distil-train-lr` | `1e-4` | Distillation learning rate |

### Output Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--save-artifacts-path` | None | Path to save models and data artifacts |

## Configuration Files

YAML configuration files can be used to specify all experiment parameters. Example configs are provided in the `configs/` directory:

```
configs/
├── exp_ag_news_gpt2_base.yaml      # AG News with GPT-2, base reference
├── exp_ag_news_gpt2_distil.yaml    # AG News with GPT-2, distillation reference
├── exp_wikitext_gpt2_base.yaml     # WikiText with GPT-2, base reference
├── exp_xsum_gpt2_base.yaml         # XSum with GPT-2, base reference
└── ...
```

## Output Metrics

The attack outputs the following metrics:

| Metric | Description |
|--------|-------------|
| **AUC** | Area under the ROC curve (0.5 = random, 1.0 = perfect) |
| **TPR@1%FPR** | True positive rate at 1% false positive rate |
| **TPR@0.1%FPR** | True positive rate at 0.1% false positive rate |

Results are automatically appended to `results.csv` next to this module.

## Project Structure

```
leakpro/llm_attacks/mia/
├── __main__.py      # CLI entry point (python -m leakpro.llm_attacks.mia)
├── attack.py        # Main attack logic
├── config.py        # Configuration dataclass and argument parsing
├── datasets.py       # Dataset loading and preprocessing
├── ez_score.py       # Error-zone score computation
├── metrics.py        # Evaluation metrics (AUC, TPR@FPR)
├── models.py          # Model building, training, and scoring
└── configs/           # Example YAML experiment configs
```

## How It Works

1. **Data Sampling**: Split dataset into member (training) and non-member (evaluation) sets
2. **Target Model Training**: Fine-tune the target model on member data
3. **Reference Model Construction**: Build reference model using selected variant (base/distillation/sft)
4. **Score Computation**: For each text sample, compute the error-zone ratio:
   - Get log-probabilities from both target and reference models
   - Identify error-zone tokens (where target's top-1 ≠ ground truth)
   - Compute ratio of positive vs. negative log-prob differences on error tokens
5. **Membership Classification**: Higher scores indicate membership

## Roadmap

A memorization-based extraction attack is planned as a future addition alongside this MIA attack under `leakpro/llm_attacks/`.

## License

See the LeakPro repository's LICENSE file for details.
