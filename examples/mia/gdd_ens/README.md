<!--
Copyright 2023-2026 Lindholmen Science Park AB
SPDX-License-Identifier: Apache-2.0
-->
# GDD_ENS membership inference example

Audits a tumor-type classifier trained on the **GDD_ENS** genomic feature table for
membership leakage, running **LiRA**, **RMIA**, and a **population** baseline.

GDD_ENS ([Darmofal et al., *Cancer Discovery* 2024](https://doi.org/10.1158/2159-8290.CD-23-0996);
[code](https://github.com/mmdarmofal/GDD_ENS)) predicts 38 tumor types from genomic features
derived from the MSK-IMPACT panel. The deployed model is a **10-MLP hyperparameter ensemble**
trained for 200 epochs with per-model Bayesian HPO and rare-class upsampling.

> **This example audits a proxy, not the published model.** It attacks a **single MLP** with a
> fixed 20-epoch recipe and **no** upsampling (see below). That makes shadow-model training
> feasible, but it means any leakage number here characterizes *this* GDD-like single MLP — it
> is a **methodological demonstration, not a privacy assessment of the deployed GDD-ENS**. A
> single, lightly-trained, uncalibrated MLP without ensembling likely *overstates* leakage
> relative to the real ensemble. See `knowledge/papers/gdd-ens-2024/notes.md` for the full
> divergence table.

## What this example does

1. Builds a population from the GDD **full feature table** (`msk_solid_heme_ft.csv`),
   reproducing GDD's preprocessing up to — but not including — its train/test split and
   oversampling:
   - keeps only `Classification_Category == 'train'` samples (drops low-purity / `other`),
   - keeps one sample per patient (`PATIENT_ID` = first 9 chars of `SAMPLE_ID`) so a patient
     never appears in both members and non-members,
   - label-encodes the `Cancer_Type` target (38 classes),
   - drops identifier/metadata columns and features that are constant across the population,
   - drops one sample if the count is odd (from the most frequent class) so the population is
     even — LeakPro's balanced shadow-model assignment requires an even shadow pool,
   - does **no** oversampling — LeakPro carves the in/out membership split itself.
2. Trains a fresh single-MLP target (`GddMLP`) with a fixed recipe and records its metadata.
3. Runs the audit with the split-handler API: `GddDataHandler` (data) + `GddModelHandler`
   (model). Shadow models replay the recorded recipe.

We deliberately do **not** consume GDD's `ft_train_labelled.csv` output: that file is GDD's
80% training split *after* `RandomOverSampler` balancing (duplicate rows for rare classes),
which is unsuitable for a clean membership split.

## Prerequisites

This example does **not** ship the data. GENIE is controlled-access (AACR credentials) and the
full feature table is ~860 MB.

1. Obtain GENIE access and produce the GDD full feature table `msk_solid_heme_ft.csv` — either
   unzip `data/msk_solid_heme_ft.zip` from the [GDD_ENS](https://github.com/mmdarmofal/GDD_ENS)
   repository, or regenerate it with that repo's `generate_feature_table.py`.
2. Place it in this example's `data/` folder (not tracked by git, so create it first):

   ```bash
   mkdir -p data
   cp /path/to/GDD_ENS/msk_solid_heme_ft.csv  data/
   ```

The example reads only this CSV — no GDD code runs — so it is unaffected by GDD's own Python
version (the example targets **python3.12** / the LeakPro environment).

## How to run

From this directory (paths in the configs are resolved relative to the working directory):

```bash
cd examples/mia/gdd_ens
jupyter notebook gdd_main.ipynb
```

Run the cells top to bottom. The first run builds and caches `data/gdd_ens.pkl` (~680 MB for
the ~36,976 x 4,599 float32 population; reading the 860 MB feature table peaks at several GB
of RAM — a one-time cost), trains the target, then runs the audit and writes a PDF report
under `leakpro_output/`.

## Configuration

| File | Purpose |
|------|---------|
| `train_config.yaml` | Target training recipe (epochs, batch size, optimizer, lr, weight decay) and the population split fractions (`f_train`, `f_test`). |
| `audit.yaml` | Attack list (LiRA / RMIA / population), shadow-model counts, and the target model/data paths. |

To control runtime, lower `num_shadow_models` in `audit.yaml` (8 per attack by default) or
`train.epochs` in `train_config.yaml` (20 by default). The audit trains
`num_shadow_models` models per shadow-based attack, so CPU-only runs are slow — use a GPU.

## Files

| File | Role |
|------|------|
| `gdd_data_handler.py` | `GddDataHandler(role="data")` — the `UserDataset` (raw features, no scaling). |
| `gdd_model_handler.py` | `GddModelHandler(role="model")` — `train` / `eval`. |
| `utils/gdd_model.py` | `GddMLP` target architecture (referenced from `audit.yaml`). |
| `utils/gdd_data.py` | One-time CSV → tensors preparation. |
| `gdd_main.ipynb` | End-to-end: prepare data, train target, run audit. |

## Notes

- **Reading the results.** On a representative run the shadow-based attacks beat the population
  baseline (RMIA ≈ 0.68, LiRA ≈ 0.64 ROC AUC), while the population attack sits at or just below
  0.5 — i.e. it finds **no** membership signal, which is the expected behaviour for a baseline
  that uses no shadow models. Treat the population AUC as a sanity floor, not a leakage estimate.
- **Class imbalance.** The 38 classes are very skewed (NSCLC has thousands of samples, the
  rarest a few hundred). Accuracy is a weak summary; consider macro vs. micro metrics when
  interpreting the target model. The target train/test split is currently a plain random split
  (the paper used `StratifiedShuffleSplit`); with this imbalance, rare classes may get very few
  or zero training samples, so per-class membership signal is weak and any average AUC is
  dominated by the head classes.
- **Features are raw**, matching how GDD fed the feature table to its MLP. No standardization.
  Note the feature set mixes binary alteration flags with a few unscaled numerical features
  (the 96 SBS substitution counts, mutation/CNA burden, MSI score), so the high-variance
  numerical features dominate gradient scale.
