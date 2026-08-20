<!--
Copyright 2023-2026 Lindholmen Science Park AB
SPDX-License-Identifier: Apache-2.0
-->
# Lenght-of-Stay Usecase
In this use case, we focus on attacking length-of-stay classifier models. As part of the example, we train a Logistic Regression model and a Gated Recurrent Unit with Decay (GRU-D).
<br>
To run the use case follow these stpes:<br>
1. Prepare the data following instructions in ```mimic_prepration/ReadMe.md ``` 
2. Run `mimic_dataset_prep.ipynb` to prepare the dataset (requires LeakPro to be installed). Note that the preparation is configured via `train_config.yaml` — set `training_method` to either `LR` or `GRUD` depending on the target model.


Once the dataset is ready, you can proceed to run any of the use case notebooks.




## DP-SGD privacy-utility optimization

The LOS DP-SGD optimization examples are pending migration to the integrated
RMIA + Bayesian-optimization pipeline; see
[`dpsgd_optimization/README.md`](dpsgd_optimization/README.md). The reference
implementation lives in
[`examples/mia/cifar/dpsgd_optimization`](../cifar/dpsgd_optimization/README.md).
