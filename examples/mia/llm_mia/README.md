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

## Layout

Two self-contained experiments, kept in their own folders so their configs/data/output never collide;
the code that drives both (`prepare_target.py`, `llm_model_handler.py`, etc.) is shared, one level up:

```
llm_mia/
├── hf_wrapper.py, llm_data_handler.py, llm_model_handler.py, json_dataset.py   # shared code
├── prepare_target.py, import_external_target.py, run_audit.py, sweep_report.py # shared scripts
├── ez-mia/    # GPT-2 / WikiText-103 demo (EZ-MIA only, small and fast)
│   ├── train_config_ez-mia.yaml, audit.yaml, llm_mia_main.ipynb
│   └── data/wikitext_gpt2_128.pkl   # cached population (produced on first run)
└── wbc/       # Pythia-2.8B / Khan Academy, the WBC paper's own reference config
    └── train_config_wbc_pythia.yaml, audit_wbc_pythia_paper.yaml
```

Run everything from *inside* the relevant subfolder, invoking the shared scripts with `../`:
```bash
cd ez-mia && python ../prepare_target.py --config train_config_ez-mia.yaml && python ../run_audit.py
# or
cd wbc && python ../prepare_target.py --config train_config_wbc_pythia.yaml && python ../run_audit.py --audit audit_wbc_pythia_paper.yaml
```
This works because Python adds a script's *own* directory to `sys.path` regardless of cwd (so
`from hf_wrapper import ...` etc. still resolve), while every relative path *inside* a config
(`data_path`, `target_folder`, `output_dir`) resolves against the cwd -- which is the subfolder you're
standing in, so each experiment's artifacts land inside its own folder. The one path that needed an
explicit fix for this: `target.module_path` in both `audit.yaml` files is `../hf_wrapper.py`, since
`module_path` is resolved by `leakpro`'s loader the same cwd-relative way, not relative to the config
file's own location.

**Quick start:** `ez-mia/llm_mia_main.ipynb` runs the whole GPT-2/WikiText demo (prepare target -> plot
curves -> audit) interactively in one notebook, the same pattern as `examples/mia/cifar/cifar_main.ipynb`.
It's fast enough to sit in live cells; the paper-scale WBC/Pythia-2.8B run is not, so that one stays
script-based (`prepare_target.py`/`import_external_target.py` + `run_audit.py`, from inside `wbc/`).

## 1. Prepare the target

```bash
cd ez-mia
python ../prepare_target.py --config train_config_ez-mia.yaml                       # → ./target_full
python ../prepare_target.py --config train_config_ez-mia.yaml --finetune-method lora # → ./target_lora
```

Defaults reproduce the EZ-MIA paper's WikiText-103 / GPT-2 / 128-token / 3-epoch setting: 10k member
and 10k non-member chunks. The dataset is referenced by its full hub id (`Salesforce/wikitext`);
recent `datasets` releases no longer resolve the bare `wikitext` name. The script tokenises once (`data/wikitext_gpt2_128.pkl`), fine-tunes with
`LLMModelHandler.train`, and writes `target_model.pkl` + `model_metadata.pkl` through
`LeakPro.make_mia_metadata` — the same files every other LeakPro MIA example produces. The metadata's
`accuracy` is next-token top-1 accuracy.

For LoRA the adapters are merged before saving so the plain `HFCausalLMWrapper` can reload the weights.

## 2. Audit

Still inside `ez-mia/`:
```bash
python ../run_audit.py                          # target_full, from audit.yaml
python ../run_audit.py --target ./target_lora   # second target, separate output dir
```

`audit.yaml` declares the reference model **inside each attack's entry** (`references:`), so it is
part of the attack hash and of the saved result metadata.

Paper numbers to expect (WikiText / GPT-2 / full FT): EZ-MIA AUC 0.984, TPR@1%FPR 66.3%, TPR@0.1%FPR 14.0%.
A large shortfall is a bug in the pipeline, not a finding.

## Reproducing a specific paper config

A few knobs exist specifically to match a paper's own reference config rather than this example's
defaults:

- **`window_lengths`** (WBC only) — an explicit list of window sizes, used verbatim instead of
  `w_min`/`w_max`/`n_windows`'s geometric formula. Needed for exact reproduction: the formula does
  *not* reproduce the WBC paper's own published grid (`geometric_windows(2, 40, 10)` gives
  `[2, 3, 4, 5, 8, 11, 15, 21, 29, 40]`, not the paper's `[2, 3, 4, 6, 9, 13, 18, 25, 32, 40]`).
- **`max_samples`** — caps how many audit rows are actually scored (the papers' `test_samples`),
  stratified to keep the member/non-member ratio, seeded by `audit.random_seed`. Independent of
  `n_members`/`n_nonmembers` in `train_config_ez-mia.yaml`, which fix the population size at data-prep time.
- **`n_bootstrap_samples`** — bootstrap-resamples the audit rows this many times and reports mean/95%
  CI for AUC and TPR-at-fixed-FPR, attached to the result as `result.bootstrap` (a plain attribute,
  not part of `MIAResult`'s serialized schema). `None` (default) reports only the point estimate.
- **Auditing an already-trained checkpoint** (not one `prepare_target.py` fine-tunes itself), e.g.
  from pre-split member/non-member JSON files: use `import_external_target.py` instead of
  `prepare_target.py` — see its docstring. Produces the same `target_model.pkl`/`model_metadata.pkl`/
  population pickle, so `run_audit.py`/`audit.yaml` need no changes. `audit_wbc_pythia_paper.yaml` is
  a worked example (Pythia-2.8B, 512-token sequences, the paper's window list, `batch_size: 1`,
  `n_bootstrap_samples: 100`) for the WBC paper's own reference config shape.

**Not covered by any config, needs code changes:** multi-GPU (this runs on exactly one device, set
via `LEAKPRO_DEVICE`).

**Reference model dtype** — checked against the actual [WBC reference codebase](https://github.com/Stry233/WBC):
its target-training path (`trainer/misc/models.py`) hardcodes `torch.bfloat16`, but its reference-model
loader (`utils.py:init_model`, used via `attacks/misc/utils.py:load_reference`) never forces a dtype at
all — the reference stays at whatever the checkpoint natively is (typically `float32`), *not* matched to
the target. So `dtype: bfloat16` on both target and reference is not itself wrong, but if you're trying to
reproduce the original codebase's actual numbers, leave the reference at `float32` rather than matching
the target's dtype — see `audit_wbc_pythia_paper.yaml`.

### Fine-tuning-phase fidelity (`train_config_ez-mia.yaml`/`train_config_wbc_pythia.yaml`'s `train:`/`data:` sections)

For matching a paper's own *target-training* recipe (not just the audit), rather than this
example's WikiText/GPT-2 defaults:

- **`data.source: local`** — fine-tune on your own pre-split member/non-member JSON files
  (`data.train_path`/`data.test_path`, plus `data.text_field` if not `"text"`) instead of tokenising
  an HF dataset. The split is whichever file a sequence came from -- `n_members`/`n_nonmembers`/
  `n_sequences` don't apply to this path. `train_config_wbc_pythia.yaml` is a worked example
  (Pythia-2.8B on a local cosmopedia/Khan Academy subset).
- **`train.warmup_steps`** — linear LR warmup over this many *optimizer* steps (post-accumulation),
  then linear decay to 0 over the rest — matches HF `Trainer`'s default schedule when its
  `warmup_steps` is set. `0` (default) keeps the LR constant, as before.
- **`train.eval_strategy: epoch`** — run a held-out eval pass after every epoch, not just once at the
  very end (`test.json`/the held-out split doubles as the paper's "val" set, as it does everywhere
  else in this repo). Anything else (including omitting the key) keeps the old end-only behavior.
- **`train.save_strategy: epoch`** + **`train.save_total_limit`** — checkpoint after every epoch to
  `log_dir/ckpts/checkpoint-<global_step>/model.pkl`, pruning down to the newest `save_total_limit`
  (default 1) afterwards.

**Still not covered, needs real infrastructure, not a config knob:** multi-GPU/DeepSpeed training
(this trains on exactly one device — a paper's "N cards" effective batch size only applies to that
one device here) and experiment tracking (no wandb/similar integration exists; progress prints to
stdout per epoch only).

## 3. LoRA vs full fine-tuning

Still inside `ez-mia/`:
```bash
python ../sweep_report.py leakpro_output_full leakpro_output_lora --out sweep_report
```

Loads every saved result (`data_objects/*.json`) from each run, and writes one combined log-log ROC
plus a LaTeX table with AUC and TPR at fixed FPRs. The paper reports a 55× gap in TPR@1%FPR between
full fine-tuning and LoRA on the same model and data (XSum / GPT-2); this is the audit-relevant
result and the reason the sweep exists.

## Files

Shared code (`llm_mia/`, one level above both experiment folders):

| File | Role |
|---|---|
| `hf_wrapper.py` | `HFCausalLMWrapper` — the `target.module_path` class LeakPro reconstructs |
| `llm_data_handler.py` | `role="data"` handler: tokenised `UserDataset` |
| `llm_model_handler.py` | `role="model"` handler: full / LoRA fine-tuning, next-token eval |
| `prepare_target.py` | tokenise (HF dataset or local JSON) → fine-tune → save target + metadata |
| `json_dataset.py` | shared local-JSON loading/tokenising, used by `prepare_target.py` and `import_external_target.py` |
| `import_external_target.py` | import an already-trained checkpoint + JSON train/test files, no fine-tuning |
| `run_audit.py` | `LeakPro(...).run_audit()` |
| `sweep_report.py` | combined report over several runs |

`ez-mia/` (GPT-2 / WikiText-103 demo):

| File | Role |
|---|---|
| `train_config_ez-mia.yaml` | fine-tuning recipe: GPT-2, WikiText-103, 128-token chunks |
| `audit.yaml` | EZ-MIA audit config -- this folder is EZ-MIA only; WBC lives in `../wbc/` |
| `llm_mia_main.ipynb` | interactive version of the same: prepare target → plot curves → audit, in one notebook |

`wbc/` (Pythia-2.8B / Khan Academy, the WBC paper's own reference config):

| File | Role |
|---|---|
| `train_config_wbc_pythia.yaml` | worked example: WBC paper's own Pythia-2.8B fine-tuning recipe |
| `audit_wbc_pythia_paper.yaml` | worked example: WBC paper's own reference config shape (window list, `batch_size: 1`, `n_bootstrap_samples: 100`) |
