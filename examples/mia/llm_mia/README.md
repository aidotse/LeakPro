# LLM membership inference: EZ-MIA and WBC

Audits a fine-tuned causal language model with two training-free, reference-based membership
inference attacks:

- **EZ-MIA** — Ilić, Stanojević, Cvejoski, arXiv:2601.12104. Error-zone ratio `P/N` of log-prob shifts at mispredicted tokens.
- **WBC** — Chen et al., arXiv:2601.02751. Sign vote over sliding windows of geometrically spaced sizes.

Both need only the fine-tuned target and the frozen base checkpoint it was fine-tuned from. No shadow models.

## Install

```bash
pip install -e ".[llm]"        # transformers, datasets, accelerate, peft
```

`transformers >= 5` needs `torch >= 2.5`.

## 1. Prepare the target

```bash
python prepare_target.py --config train_config.yaml                       # → ./target_full
python prepare_target.py --config train_config.yaml --finetune-method lora # → ./target_lora
```

Defaults reproduce the EZ-MIA paper's WikiText-103 / GPT-2 / 128-token / 3-epoch setting: 10k member
and 10k non-member chunks. The script tokenises once (`data/wikitext_gpt2_128.pkl`), fine-tunes with
`LLMModelHandler.train`, and writes `target_model.pkl` + `model_metadata.pkl` through
`LeakPro.make_mia_metadata` — the same files every other LeakPro MIA example produces. The metadata's
`accuracy` is next-token top-1 accuracy.

For LoRA the adapters are merged before saving so the plain `HFCausalLMWrapper` can reload the weights.

## 2. Audit

```bash
python run_audit.py                          # target_full, from audit.yaml
python run_audit.py --target ./target_lora   # second target, separate output dir
```

`audit.yaml` declares the reference model **inside each attack's entry** (`references:`), so it is
part of the attack hash and of the saved result metadata.

Paper numbers to expect (WikiText / GPT-2 / full FT): EZ-MIA AUC 0.984, TPR@1%FPR 66.3%, TPR@0.1%FPR 14.0%.
A large shortfall is a bug in the pipeline, not a finding.

## 3. LoRA vs full fine-tuning

```bash
python sweep_report.py leakpro_output_full leakpro_output_lora --out sweep_report
```

Loads every saved result (`data_objects/*.json`) from each run, and writes one combined log-log ROC
plus a LaTeX table with AUC and TPR at fixed FPRs. The paper reports a 55× gap in TPR@1%FPR between
full fine-tuning and LoRA on the same model and data (XSum / GPT-2); this is the audit-relevant
result and the reason the sweep exists.

## Files

| File | Role |
|---|---|
| `hf_wrapper.py` | `HFCausalLMWrapper` — the `target.module_path` class LeakPro reconstructs |
| `llm_data_handler.py` | `role="data"` handler: tokenised `UserDataset` |
| `llm_model_handler.py` | `role="model"` handler: full / LoRA fine-tuning, next-token eval |
| `prepare_target.py` | tokenise → fine-tune → save target + metadata |
| `run_audit.py` | `LeakPro(...).run_audit()` |
| `sweep_report.py` | combined report over several runs |
