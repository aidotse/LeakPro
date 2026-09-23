# LLM membership inference: EZ-MIA and WBC

Audits a fine-tuned causal language model with two training-free, reference-based membership
inference attacks — **EZ-MIA** (arXiv:2601.12104) and **WBC** (arXiv:2601.02751). Both compare the
fine-tuned target against the frozen base checkpoint it came from. No shadow models.

## Install

```bash
pip install -e ".[llm]"        # transformers, datasets, accelerate, peft
```

## Layout

```
llm_mia/
├── ...                        # shared code/scripts (prepare_target.py, run_audit.py, etc.)
├── ez-mia/                    # GPT-2 / WikiText-103 demo — EZ-MIA only, small and fast
└── wbc/                       # Pythia-2.8B / Khan Academy — WBC only, paper setup, HPU-tuned training
```
Each folder is self-contained (own config, data, output); run scripts *from inside* the folder,
e.g. `cd ez-mia && python ../run_audit.py`.

## EZ-MIA — GPT-2 / WikiText-103

**Easiest:** open and run `ez-mia/llm_mia_main.ipynb` top to bottom — prepares the target, plots
training curves, and runs the audit, all in one notebook.

**Or from a terminal:**
```bash
cd ez-mia
python ../prepare_target.py --config train_config_ez-mia.yaml
python ../run_audit.py
```
Expect around EZ-MIA AUC 0.98 (paper: 0.984) — a large shortfall means something's broken, not a
real finding.

## WBC — Pythia-2.8B / Khan Academy

A bigger fine-tune (real time and memory, not a quick demo) — three steps, in order. Device
(HPU/CUDA/CPU) auto-detects; set `LEAKPRO_DEVICE=hpu` (or `cuda`/`cpu`) only if you need to pin one.

**1. Get the data** (a third-party HF dataset, not shipped here):
```bash
cd examples/mia/llm_mia
python prepare_json_subset.py \
    --dataset-name HuggingFaceTB/cosmopedia --config khanacademy \
    --num-samples 20000 --min-length 512 --tokenizer-name EleutherAI/pythia-2.8b \
    --output-dir wbc/cosmopedia-khanacademy-subset
```

**2. Fine-tune the target:**
```bash
cd wbc
python ../prepare_target.py --config train_config.yaml
```

**3. Run the audit:**
```bash
python ../run_audit.py --audit wbc_audit.yaml
```
Results land in `./leakpro_output_wbc_pythia/`.

**Out of memory in step 2?** Lower `train.batch_size` in `train_config.yaml` and raise
`train.gradient_accumulation_steps` by the same factor (keeps the effective batch size the same,
trades memory for time).

## Going further

Every other option — `window_lengths`, `max_samples`, `n_bootstrap_samples`, `data.source: local`,
`warmup_steps`, `eval_strategy`/`save_strategy`, auditing an already-trained checkpoint via
`import_external_target.py`, comparing LoRA vs. full fine-tuning via `sweep_report.py` — is
documented inline as comments in the relevant `.yaml` file or script docstring. Start from
`train_config.yaml`/`wbc_audit.yaml` (or their `ez-mia/` equivalents) and
read the comments next to the setting you want to change.
