# Tabular MINV — Handoff Notes

## What is included in this PR

- `TabularExtension` — one-hot encoding/decoding for tabular data (`leakpro/input_handler/modality_extensions/tabular_extension.py`)
- `TabularMetrics` — evaluation metrics for tabular synthetic data (`leakpro/input_handler/modality_extensions/tabular_metrics.py`)
- `plgmi.py` — full version with **all optimization methods kept intentionally** (image + tabular branches, `optimize_z_grad`, `optimize_z_grad_per_sample`, etc.) for testing different approaches
- MIMIC example: `train_target.py`, `run_leakpro.py`, `utils/` (CTGAN_extended, tabular_wrapper, etc.)

## Before you can run the MIMIC example

### 1. Register TabularExtension in leakpro.py

In `leakpro/leakpro.py`, the `modality_extensions` dict currently has `"tabular": None`.
Change it to:

```python
from leakpro.input_handler.modality_extensions.tabular_extension import TabularExtension

modality_extensions = {
    "tabular": TabularExtension,
    "image": ImageExtension,
    ...
}
```

### 2. Prepare the MIMIC data

Run `utils/mimic_data.py` to preprocess raw MIMIC-IV data. It expects:
- Raw MIMIC-IV tables at `data/physionet.org/files/mimiciv/3.1/`
- Outputs `data/df.pkl`

Then split into `data2/private_df.pkl` and `data2/public_df.pkl` (see `utils/preprocess_mimic_data.py`).

### 3. Train the target model

```bash
cd examples/minv/mimic
python train_target.py
```

This trains a pytorch_tabular GANdalF model and saves it to `./target/`.

### 4. Run the audit

```bash
python run_leakpro.py
```

---

## What still needs to be cleaned up before merging to main

### plgmi.py
- **Make `cudf` import conditional** — currently wrapped in `try/except` but any code path that uses `cudf` will still fail if not installed. Guard all `cudf` usage inside `if cudf is not None:` checks or replace with pandas equivalents.
- **Remove disk side effects** in `optimize_z_grad_per_sample` — it currently saves plots and CSVs to disk during optimization. Remove before merging.
- **Remove unused optimization variants** once the best approach is confirmed (`optimize_z_grad2`, `optimize_z_grad_original`).

### mimic_plgmi_handler.py
- `continuous_col_names` is hardcoded (lines ~133-135). Move to `train_config.yaml` or `audit.yaml` so it's configurable per dataset.
- `num_classes=705` TODO comment on line ~154 — fix this.

### conversion.py (optional cleanup)
The `review_tabular_example` branch renames functions in `leakpro/utils/conversion.py`
to use underscore prefix (`_get_model_init_params`, `_loss_to_config`, etc.) and updates
`leakpro/leakpro.py` accordingly. Apply this rename when doing the final cleanup PR.

### Run full test suite before opening the merge PR
```bash
python -m pytest leakpro/tests/
```
