# PET optimization on MIMIC-III length-of-stay

Traces the **utility-vs-attack-success frontier** for DP-SGD on the LOS binary
classification task: train the target many times under different privacy
settings, attack each one identically, and see what privacy costs in AUC.

Two targets, because they leak very differently:

| Script | Target | Utility | Notes |
|---|---|---|---|
| `run_campaign.py` | logistic regression (`target_models.LR`) | AUC | fast; the baseline campaign |
| `run_campaign_grud.py` | GRU-D over the raw time series | AUC | leakier and far slower — see below |

## Prerequisites

Both scripts read pickles that are **not in the repository**:

- `run_campaign.py` → `examples/mia/LOS/data/LR_data/{dataset,indices}.pkl`
- `run_campaign_grud.py` → `examples/mia/LOS/data/GRUD_data/{dataset,indices}.pkl`

Generate them first with the LOS example notebooks — `mimic_dataset_prep.ipynb`,
then `mimic_LR_main.ipynb` or `mimic_GRUD_main.ipynb` respectively. Without
this you get a bare `FileNotFoundError`. See `../ReadMe.md` for the MIMIC-III
access and preparation steps.

## Running it

```bash
python run_campaign.py --smoke            # pipeline check: 2 configs, 1 reference
python run_campaign.py --n-configs 50     # real sweep

python run_campaign_grud.py --smoke
python run_campaign_grud.py --n-configs 20
```

**Cost.** One configuration = 1 target + `--n-refs` reference models trained
from scratch. Always start with `--smoke`.

**GRU-D specifics.** It contains a custom `FilterLinear` and a manual 104-step
recurrence, so Opacus runs on its functorch per-sample-gradient path — correct
but much slower than LR. Keep `--n-configs` and `--epochs` modest. It also has
no `--device` flag on purpose: GRU-D pins `X_mean`, its identity matrix and the
`FilterLinear` filter to the auto-detected device at construction, and `.to()`
does not move those, so a flag would accept a value it could not honor.

## Output

Everything lands in `--out`:

| File | What it is |
|---|---|
| `evaluations.jsonl` | one JSON line per evaluated configuration; also the resume state |
| `campaign.json` | run identity (seed, proxy FPR, knob space) — resuming with different settings is refused |
| `frontier.png` | every configuration, with the Pareto front highlighted |

Interrupted runs resume by rerunning the same command. A configuration that
raises is recorded with an `error` field and retried next time rather than
killing the sweep.

## Reading a record

- `utility` — AUC on the held-out split.
- `attack_tpr` — attack success at the proxy operating point (1% false-alarm
  rate). **Lower is safer.**
- `attack_realized_fpr` / `attack_resolution_warning` — the false-alarm rate
  actually achievable in the data. When the warning is set, the TPR is
  interpolated rather than observed: unresolved, not evidence of privacy.
- `attack_tpr_ci95` — binomial sampling error over audit points only; it
  excludes target-training and reference-draw randomness.
- `attack_skipped: "utility_gate"` — the model did not beat chance AUC, so it
  was not attacked. A model that did not learn cannot be meaningfully audited.
- `epsilon` — the formal DP budget, recorded next to the empirical number so the
  two can be compared. `null` means the non-private anchor (ε = ∞).
