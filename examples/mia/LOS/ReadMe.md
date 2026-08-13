<!--
Copyright 2023-2026 Lindholmen Science Park AB
SPDX-License-Identifier: Apache-2.0
-->
# Lenght-of-Stay Usecase
In this use case, we focus on attacking length-of-stay classifier models. As part of the example, we train a Logistic Regression model, a Gated Recurrent Unit with Decay (GRU-D), and an XGBoost model.
<br>
To run the use case follow these stpes:<br>
1. Prepare the data following instructions in ```mimic_prepration/ReadMe.md ``` 
2. Run `mimic_dataset_prep.ipynb` to prepare the dataset (requires LeakPro to be installed). Note that the preparation is configured via `train_config.yaml` — set `training_method` to `LR`, `XGB`, or `GRUD` depending on the target model. `LR` and `XGB` both produce the flattened `LR_data/` layout; `GRUD` produces the 3D `GRUD_data/` layout.


Once the dataset is ready, you can proceed to run any of the use case notebooks.

## Target models

| Notebook | Target | Audit config | Data |
|---|---|---|---|
| `mimic_LR_main.ipynb` | Logistic Regression | `audit.yaml` (LR block) | `LR_data/` |
| `mimic_GRUD_main.ipynb` | GRU-D | `audit.yaml` (GRUD block) | `GRUD_data/` |
| `mimic_GRUD_DPSGD_main.ipynb` | GRU-D with DP-SGD | `audit.yaml` (GRUD DP-SGD block) | `GRUD_data/` |
| `mimic_XGB_main.ipynb` | XGBoost | `audit_xgb.yaml` | `LR_data/` |

### XGBoost target

The XGBoost notebook needs an extra dependency, which is not part of LeakPro's `mia` extra:

```bash
pip install xgboost
```

LeakPro's MIA machinery assumes every target is a `torch.nn.Module`. `xgb_target_model.py` therefore wraps the booster in a **compatibility shim** (`XGBLOS`) that satisfies that interface — the serialised booster lives in a `uint8` state-dict buffer. This is a workaround, not first-class support for non-PyTorch models.

Because the shim has no gradients, only logit- and loss-based attacks work against it: `lira`, `rmia`, `population`, `base`, `multi_signal_lira`, `yoqo`, `ramia`. The attacks `hsj`, `loss_traj`, `seqmia` and `dts` will construct from a config and then fail or return meaningless scores. `audit_xgb.yaml` documents this at the top.



