# LLM training-data extraction (memorization)

This example audits a causal language model for **verbatim memorization**: whether an attacker who can
only query the model can make it reproduce sequences from its training data. Unlike membership
inference (`examples/mia/llm_mia`), which asks "was *this* text in the training set?", extraction has
to *produce* the text itself. LeakPro runs it as `attack_type: extraction`.

A generated sequence counts as extracted when one of its 30-word windows occurs verbatim (after
lowercasing and collapsing whitespace) in a reference corpus.

## The attack

| Phase | Access | What it does | Config key |
|---|---|---|---|
| 1. Initial extraction | black box | BOS-only sampling, temperature-decay sampling and greedy continuation of corpus prefixes; rank by zlib and lowercase membership scores; verify the top 5% per strategy → **S0** | `initial` |
| 2. Expansion | black box | re-prompt with 30-50 token anchors cut from S0, 150 samples per anchor; verify; keep sequences with windows S0 did not have → **S1-novel** | `expansion` |
| 3. Fine-tuning *(optional)* | weights | LoRA continual pre-training on S0 ∪ S1-novel | `finetune` |
| 4. Re-extraction *(optional)* | weights | Phase 1 again on the tuned model; drop generations whose generated text shares a 30-token window with the fine-tuning data, before filtering → **S_new** | `finetune` |

Phases 1-2 are the base attack: cheap, and possible against any model you can sample from. They
always run (Phase 2 needs an `attacker_verifier`; `expansion: null` limits the run to Phase 1).
Phases 3-4 are an optional, more extensive attack for auditors who hold the weights: they run only
when the config has a `finetune` entry (`finetune: {}` for the defaults).

**References.** The four-phase pipeline and its defaults (150 generations per anchor; temperature
3 → 1 with `top_p=0.9` and repetition penalty 1.1 in Phase 2; LoRA `r=16`, `alpha=32`, dropout 0.1,
5 epochs, learning rate 1e-4) are those of A.-C. Matacuta, *Anything You Say Can and Will Be Used
Against You: Iterative Memorization Extraction and Fine-Tuning in Large Language Models*, Master's
thesis, University of Gothenburg, 2026 ([hdl.handle.net/2077/93397](https://hdl.handle.net/2077/93397)).
Phase 1's scoring, prefix strategy and temperature decay follow Carlini et al.,
[*Extracting Training Data from Large Language Models*](https://www.usenix.org/conference/usenixsecurity21/presentation/carlini-extracting)
(USENIX Security 2021), and the fine-tuning phase is influenced by Nasr et al.,
*Scalable Extraction of Training Data from Aligned, Production Language Models* (2025).
This implementation differs from those works in these ways:

| | Here | Thesis | Carlini et al. |
|---|---|---|---|
| Sampling (`baseline`, `temp_decay`) | `top_k=50`, `top_p=0.95` | `top_p=0.95` | top-n 40, no nucleus filter |
| Prefixes (`conditional`) | greedy | greedy | sampled (top-n) |
| Fine-tuning | LoRA, unquantized model, 10-step warm-up | QLoRA (4-bit base model) | n/a |
| Phase 4 overlap check | generated text only | whole sequence | n/a |

The **attacker verifier** (`attacker_verifier`) is a public corpus the attacker believes resembles the
training data; its results drive the attack. If you, as the auditor, have the real training data, add
it as `audit_oracle`: it is checked for every phase and reported, but never used by the attack.

## Install

```bash
pip install -e ".[extraction]"     # torch, transformers, elasticsearch, peft
pip install zstandard               # only to download the Common Crawl part of the corpus
```

## Full setup: Llama-3.1-8B, 20 GB corpus

`extraction_main.ipynb` walks through every step below in its section 0 and checks each one before
the run; `run_extraction.py` is the same run from the command line (see
[Step by step in a terminal](#step-by-step-in-a-terminal)). Both read the configuration from
`audit.yaml` (full budgets; Phases 1-2, optionally 3-4); the options of `run_extraction.py`
(`--prefixes`, `--n`, `--finetune`, ...) override single settings without editing the file.

1. **Model access.** `meta-llama/Llama-3.1-8B` is gated:
   - request access at <https://huggingface.co/meta-llama/Llama-3.1-8B> (accept the Llama 3.1
     license; Meta approves the request, usually within hours);
   - create a **Read** token at <https://huggingface.co/settings/tokens>;
   - run `hf auth login` (older versions: `huggingface-cli login`) or set `HF_TOKEN`.

   The weights (about 16 GB) download on the first run; set `HF_HOME` to cache them elsewhere.
2. **Data.** RedPajama-Data-1T is public, no account needed. The 20 GB download takes hours; the
   prefix step needs the Llama tokenizer, i.e. step 1:
   ```bash
   python prepare_data.py download --target-gb 20
   python prepare_data.py prefixes --model meta-llama/Llama-3.1-8B --output data/prefixes_conditional.jsonl
   ```
3. **Elasticsearch 8.x.** Install it as a system service (the notebook shows the package install)
   and export `ES_HOST`, `ES_USER`, `ES_PASSWORD` and `ES_CA_CERT`. Keep the password in the
   environment or in a file outside the repository. Then build the index (hours for 20 GB):
   ```bash
   python prepare_data.py index --index redpajama_chunks
   ```
4. **Run** one phase at a time with `python run_extraction.py --phase 1`, then `--phase 2` (and
   the optional 3 and 4; see the next section), or open `extraction_main.ipynb`.
   `python run_extraction.py` without `--phase` runs all configured phases in one go. Phases 3-4
   (LoRA on the 8B model) need a GPU or Gaudi card with about 24 GB of memory or more.

The notebook runs each phase in its own cell, because each takes hours. It calls the attack's
per-phase methods, which you can also use from your own code:

```python
from extraction_model_handler import ExtractionModelHandler   # this folder's model handler

attack = LeakPro(ExtractionModelHandler, "audit.yaml").attack_scheduler.attacks[0]
attack.prepare_attack()        # or: from run_extraction import prepare_attack
attack.run_phase1()   # -> S0;        run_phase2() -> S1-novel
attack.run_phase3()   # LoRA;         run_phase4() -> S_new    (optional: need `finetune`)
result = attack.finish()              # the phases run so far
```

`windowing: {min_span: 10}` also checks generations shorter than 30 words, with one window over
the whole text (at least 10 words); the default skips them.

## Step by step in a terminal

Each step runs for hours. Start it inside `tmux`, so that it keeps running when you close the
terminal or the SSH connection drops.

**Step 0. Open a session.** Once per working session:

```bash
tmux new -s extraction             # or, if it exists already: tmux attach -t extraction
conda activate leakpro_hpu         # your environment: Habana PyTorch (Gaudi) or CUDA PyTorch (GPU)
cd LeakPro/examples/extraction/llm_memorization
source ~/es_env.sh                 # ES_HOST, ES_USER, ES_PASSWORD, ES_CA_CERT
```

`Ctrl-b d` detaches: you can log out and the job keeps running. `tmux attach -t extraction` brings
it back.

**Step 1. One-time setup.** Skip what you already have (the notebook's section 0 explains each
command; the Python environment is its step 1):

```bash
hf auth login                                         # after Meta approves your access request
python prepare_data.py download --target-gb 20        # hours; writes data/aux_corpus.jsonl
python prepare_data.py prefixes --model meta-llama/Llama-3.1-8B --output data/prefixes_conditional.jsonl
python prepare_data.py index --index redpajama_chunks # hours; needs the Elasticsearch settings
```

**Step 2. Check the inputs.** Takes seconds, loads no model and writes nothing:

```bash
python run_extraction.py --check
```

It ends with `Inputs OK. Configured phases: [1, 2]` (with `--finetune`: `[1, 2, 3, 4]`), or says
what is missing.

**Step 3. Phase 1, initial extraction → S0.** The longest step:

```bash
python run_extraction.py --phase 1 2>&1 | tee -a run.log
```

**Step 4. Phase 2, expansion → S1-novel.** Its length grows with the size of S0:

```bash
python run_extraction.py --phase 2 2>&1 | tee -a run.log
```

Steps 3 and 4 are the base attack. Steps 5 and 6 are **optional**: the more extensive attack with
the weights. `--finetune` includes them (or uncomment `finetune` in `audit.yaml`).

**Step 5 (optional). Phase 3, LoRA fine-tuning:**

```bash
python run_extraction.py --finetune --phase 3 2>&1 | tee -a run.log
```

**Step 6 (optional). Phase 4, re-extraction with the fine-tuned model → S_new.** About as long as
Phase 1:

```bash
python run_extraction.py --finetune --phase 4 2>&1 | tee -a run.log
```

Each command loads the model again (a few minutes), reads the finished phases back from the cache
in seconds, runs its own phase, saves the result so far and prints the next command.

**Follow the progress** from a second terminal, or a second tmux window (`Ctrl-b c`):

```bash
tail -f leakpro_output_extraction/leakpro.log
```

A progress line is logged at least once a minute, for example
`initial-generate-baseline: 140/625 batches (22%) · 31.2 s/batch · ~4h 12m left`.

**Stopped or crashed?** Run the same command again. Finished batches come from
`leakpro_output_extraction/artifacts`, so at most the batch in progress is repeated. To stop on
purpose, press `Ctrl-C` in the job's tmux window.

**Step 7. Results:**

```bash
ls leakpro_output_extraction/results/*/      # result.json (counts), extracted.jsonl (sequences)
```

The counts per phase are also printed at the end of every step (and kept in `run.log`).

## Output

Results go to `<output_dir>/results/<attack id>/`:

- `result.json`: counts per phase and strategy (generated, kept, selected, verified, rate, and per
  phase `anchors`, `novel`, `overlap_removed` or `finetune_corpus`);
- `extracted.jsonl`: one record per extracted sequence with its matched windows. Set
  `store_text: hash` or `none` if the report must not contain the extracted text.

Intermediate stages are cached in `<output_dir>/artifacts`. Running the same command again resumes an
interrupted run; changing a setting recomputes only the stages after it.

## Implementation notes

- Each generation batch has its own seed, derived from `random_seed`, so runs reproduce exactly and
  an interrupted run resumes with the same samples.
- Phase 2 anchors are cut from the generated text (original casing) at the first verified window.
- Phase 3 trains LoRA with a plain PyTorch loop (AdamW; the learning rate warms up linearly over
  `warmup_steps`, default 10 optimizer steps, then stays constant). The base model is
  not quantized, and the trainable LoRA weights are float32.
- **Hardware.** The same code runs on NVIDIA GPUs (CUDA) and Intel Gaudi (HPU), and on CPU for
  small tests. LeakPro picks the device (Gaudi, then CUDA, then CPU; set `LEAKPRO_DEVICE=cuda`,
  `hpu` or `cpu` to choose) and the model's dtype follows it, the same in every phase: bfloat16 on
  Gaudi and on GPUs with native bfloat16 (Ampere and newer, e.g. A100, L4, H100); float16 on older
  GPUs (T4, V100), where Phase 3 adds loss scaling; float32 on CPU. The log states the device and
  dtype at the start of the run. Samples are reproducible on the same hardware; GPU and Gaudi draw
  different ones from the same seeds, so finish a run on the machine it started on.
  The unit tests run on the CPU; `LEAKPRO_TEST_DEVICE=cuda python -m pytest leakpro/tests/extraction_attacks`
  (or `hpu`) runs them, all four phases with tiny models, on the accelerator in about a minute.
- The Elasticsearch index covers every document completely: the last chunk is aligned to the end of
  the document and documents shorter than one chunk are indexed whole.
