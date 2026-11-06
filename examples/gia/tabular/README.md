# Tabular data

Gradient inversion of tabular records (continuous plus one-hot categorical features). Run from the repository root; `--quick` is an offline CPU smoke run on a synthetic table (seconds). Results go to `examples/gia/outputs/<script>/`.

**Metrics.** "TabLeak accuracy" (`tableak_acc`, and `reconstruction_accuracy` in percent) is the share of values recovered after optimally matching reconstructed rows to true rows: a categorical value must match exactly, a continuous value must land within 0.319 standard deviations of the truth (the tolerance of Vero et al., *TabLeak*, ICML 2023). `prior_accuracy` scores a reconstruction that guesses every column's mean or mode, `random_accuracy` one drawn uniformly; both come from the training split passed as `reference_rows`, never from the attacked rows.

**Data.** Adult is fetched from OpenML (`fetch_openml("adult", version=2)`; UCI Adult, CC BY 4.0) and California Housing through `sklearn.datasets.fetch_california_housing` (StatLib; Pace and Barry, 1997). Both are cached by scikit-learn in `~/scikit_learn_data`.

## tabular_attack
Recover a client's rows from one gradient. Compares the softmax-relaxation attack with known labels and with iDLG label inference, and the raw one-hot attack (`tabular_raw_onehot`: raw one-hot slots, cosine only). The private rows are synthetic.

`python -m examples.gia.tabular.tabular_attack [--quick]`

<!-- results:tabular_attack -->
| Attack | reconstruction_accuracy | continuous_accuracy | categorical_accuracy | exact_row_recovery |
|---|---|---|---|---|
| softmax, known labels | 100.00 ± 0.00 (n=1) | 1.00 ± 0.00 (n=1) | 1.00 ± 0.00 (n=1) | 1.00 ± 0.00 (n=1) |
| softmax, iDLG labels | 100.00 ± 0.00 (n=1) | 1.00 ± 0.00 (n=1) | 1.00 ± 0.00 (n=1) | 1.00 ± 0.00 (n=1) |
| raw one-hot | 100.00 ± 0.00 (n=1) | 1.00 ± 0.00 (n=1) | 1.00 ± 0.00 (n=1) | 1.00 ± 0.00 (n=1) |

Command: `python -m examples.gia.tabular.tabular_attack --out examples/gia/outputs/tabular_attack --device cpu`; torch `2.5.1+cu121`, Quadro RTX 5000.
<!-- /results -->

All three variants recover every row exactly (reconstruction accuracy 100.00).

## california_housing
What does one client's gradient reveal about their housing blocks? Reports per-feature error in standard deviations and original units. California Housing is downloaded once through scikit-learn; `--quick` uses a synthetic table of the same shape.

`python -m examples.gia.tabular.california_housing [--quick]`

<!-- results:california_housing -->
| Attack | reconstruction_accuracy | continuous_accuracy | mae_sigma | rmse_sigma | prior_accuracy | random_accuracy |
|---|---|---|---|---|---|---|
| California Housing | 59.38 ± 0.00 (n=1) | 0.59 ± 0.00 (n=1) | 0.54 ± 0.00 (n=1) | 0.90 ± 0.00 (n=1) | 0.38 ± 0.00 (n=1) | 0.10 ± 0.00 (n=1) |

Command: `python -m examples.gia.tabular.california_housing --out examples/gia/outputs/california_housing --device cpu`; torch `2.5.1+cu121`, Quadro RTX 5000.
<!-- /results -->

`reconstruction_accuracy` is a percentage and `prior_accuracy` and `random_accuracy` are fractions, so on the same scale reconstruction accuracy is 59.38% against a prior accuracy of 38% and a random accuracy of 10%.

## leakage_factors
Which factors drive tabular leakage? Each script sweeps one factor over Adult and California Housing (downloaded on first use; see Data above), attacking with `tabular_raw_onehot` (raw one-hot slots, cosine-only matching) unless `--variants` adds the softmax relaxation. `--quick` uses a synthetic table. Run as `python -m examples.gia.tabular.leakage_factors.<script> [--quick]`. These scripts take `--seeds` and `--max-iter` in place of the shared `--seed`, `--num-images` and `--iter-scale`.

| Script | Question |
|---|---|
| `architecture` | Which architecture factors (width, depth, normalisation, dropout, activation) drive leakage? |
| `attack_iterations` | Does a larger attack budget help? |
| `batch_sizes` | How does leakage change with batch size, raw one-hot attack versus softmax relaxation? |
| `fedavg` | How do local epochs and client batch size affect FedAvg leakage? |
| `label_knowledge` | What changes when the attacker does not know the labels? |
| `training_exposure` | How does leakage change over the course of training? |

<!-- results:architecture -->
| Attack | tableak_acc | exact_row_recovery | prior_accuracy | random_accuracy | gain_over_prior |
|---|---|---|---|---|---|
| dataset=adult, variant=raw_onehot, d_hidden=32, n_layers=1, norm=layernorm, dropout=0.0, activation=relu | 0.42 ± 0.03 (n=2) | 0.00 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.13 ± 0.05 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=32, n_layers=1, norm=layernorm, dropout=0.0, activation=gelu | 0.39 ± 0.04 (n=2) | 0.00 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.16 ± 0.05 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=32, n_layers=1, norm=layernorm, dropout=0.1, activation=relu | 0.39 ± 0.04 (n=2) | 0.00 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.16 ± 0.06 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=32, n_layers=1, norm=layernorm, dropout=0.1, activation=gelu | 0.43 ± 0.01 (n=2) | 0.00 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.12 ± 0.02 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=32, n_layers=1, norm=batchnorm, dropout=0.0, activation=relu | 0.42 ± 0.06 (n=2) | 0.00 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.13 ± 0.07 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=32, n_layers=1, norm=batchnorm, dropout=0.0, activation=gelu | 0.50 ± 0.04 (n=2) | 0.00 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.04 ± 0.06 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=32, n_layers=1, norm=batchnorm, dropout=0.1, activation=relu | 0.40 ± 0.05 (n=2) | 0.00 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.15 ± 0.06 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=32, n_layers=1, norm=batchnorm, dropout=0.1, activation=gelu | 0.47 ± 0.07 (n=2) | 0.00 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.08 ± 0.09 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=32, n_layers=1, norm=none, dropout=0.0, activation=relu | 0.37 ± 0.01 (n=2) | 0.00 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.18 ± 0.02 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=32, n_layers=1, norm=none, dropout=0.0, activation=gelu | 0.36 ± 0.03 (n=2) | 0.00 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.19 ± 0.04 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=32, n_layers=1, norm=none, dropout=0.1, activation=relu | 0.38 ± 0.01 (n=2) | 0.00 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.17 ± 0.03 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=32, n_layers=1, norm=none, dropout=0.1, activation=gelu | 0.38 ± 0.01 (n=2) | 0.00 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.17 ± 0.03 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=32, n_layers=3, norm=layernorm, dropout=0.0, activation=relu | 0.45 ± 0.04 (n=2) | 0.03 ± 0.03 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.10 ± 0.06 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=32, n_layers=3, norm=layernorm, dropout=0.0, activation=gelu | 0.46 ± 0.05 (n=2) | 0.00 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.09 ± 0.07 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=32, n_layers=3, norm=layernorm, dropout=0.1, activation=relu | 0.46 ± 0.03 (n=2) | 0.00 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.09 ± 0.05 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=32, n_layers=3, norm=layernorm, dropout=0.1, activation=gelu | 0.45 ± 0.03 (n=2) | 0.03 ± 0.03 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.10 ± 0.04 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=32, n_layers=3, norm=batchnorm, dropout=0.0, activation=relu | 0.41 ± 0.08 (n=2) | 0.00 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.14 ± 0.09 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=32, n_layers=3, norm=batchnorm, dropout=0.0, activation=gelu | 0.45 ± 0.03 (n=2) | 0.06 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.10 ± 0.05 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=32, n_layers=3, norm=batchnorm, dropout=0.1, activation=relu | 0.40 ± 0.05 (n=2) | 0.00 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.15 ± 0.06 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=32, n_layers=3, norm=batchnorm, dropout=0.1, activation=gelu | 0.42 ± 0.05 (n=2) | 0.03 ± 0.03 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.13 ± 0.06 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=32, n_layers=3, norm=none, dropout=0.0, activation=relu | 0.38 ± 0.06 (n=2) | 0.00 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.17 ± 0.07 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=32, n_layers=3, norm=none, dropout=0.0, activation=gelu | 0.38 ± 0.02 (n=2) | 0.00 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.17 ± 0.04 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=32, n_layers=3, norm=none, dropout=0.1, activation=relu | 0.40 ± 0.04 (n=2) | 0.00 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.15 ± 0.05 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=32, n_layers=3, norm=none, dropout=0.1, activation=gelu | 0.36 ± 0.03 (n=2) | 0.00 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.19 ± 0.04 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=128, n_layers=1, norm=layernorm, dropout=0.0, activation=relu | 0.49 ± 0.06 (n=2) | 0.09 ± 0.03 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.06 ± 0.07 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=128, n_layers=1, norm=layernorm, dropout=0.0, activation=gelu | 0.56 ± 0.03 (n=2) | 0.22 ± 0.03 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | 0.01 ± 0.04 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=128, n_layers=1, norm=layernorm, dropout=0.1, activation=relu | 0.52 ± 0.04 (n=2) | 0.06 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.03 ± 0.06 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=128, n_layers=1, norm=layernorm, dropout=0.1, activation=gelu | 0.56 ± 0.02 (n=2) | 0.25 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | 0.02 ± 0.04 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=128, n_layers=1, norm=batchnorm, dropout=0.0, activation=relu | 0.43 ± 0.08 (n=2) | 0.06 ± 0.06 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.12 ± 0.09 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=128, n_layers=1, norm=batchnorm, dropout=0.0, activation=gelu | 0.45 ± 0.05 (n=2) | 0.03 ± 0.03 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.10 ± 0.06 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=128, n_layers=1, norm=batchnorm, dropout=0.1, activation=relu | 0.43 ± 0.08 (n=2) | 0.03 ± 0.03 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.12 ± 0.09 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=128, n_layers=1, norm=batchnorm, dropout=0.1, activation=gelu | 0.46 ± 0.06 (n=2) | 0.00 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.08 ± 0.08 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=128, n_layers=1, norm=none, dropout=0.0, activation=relu | 0.42 ± 0.05 (n=2) | 0.00 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.13 ± 0.06 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=128, n_layers=1, norm=none, dropout=0.0, activation=gelu | 0.40 ± 0.01 (n=2) | 0.00 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.15 ± 0.00 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=128, n_layers=1, norm=none, dropout=0.1, activation=relu | 0.42 ± 0.06 (n=2) | 0.00 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.13 ± 0.07 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=128, n_layers=1, norm=none, dropout=0.1, activation=gelu | 0.41 ± 0.00 (n=2) | 0.00 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.14 ± 0.01 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=128, n_layers=3, norm=layernorm, dropout=0.0, activation=relu | 0.51 ± 0.07 (n=2) | 0.09 ± 0.09 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.04 ± 0.08 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=128, n_layers=3, norm=layernorm, dropout=0.0, activation=gelu | 0.60 ± 0.00 (n=2) | 0.22 ± 0.09 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | 0.05 ± 0.02 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=128, n_layers=3, norm=layernorm, dropout=0.1, activation=relu | 0.52 ± 0.00 (n=2) | 0.09 ± 0.03 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.03 ± 0.01 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=128, n_layers=3, norm=layernorm, dropout=0.1, activation=gelu | 0.57 ± 0.08 (n=2) | 0.12 ± 0.12 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | 0.02 ± 0.10 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=128, n_layers=3, norm=batchnorm, dropout=0.0, activation=relu | 0.52 ± 0.07 (n=2) | 0.12 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.03 ± 0.08 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=128, n_layers=3, norm=batchnorm, dropout=0.0, activation=gelu | 0.59 ± 0.02 (n=2) | 0.38 ± 0.06 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | 0.04 ± 0.01 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=128, n_layers=3, norm=batchnorm, dropout=0.1, activation=relu | 0.52 ± 0.07 (n=2) | 0.19 ± 0.06 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.03 ± 0.09 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=128, n_layers=3, norm=batchnorm, dropout=0.1, activation=gelu | 0.67 ± 0.04 (n=2) | 0.38 ± 0.06 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | 0.12 ± 0.03 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=128, n_layers=3, norm=none, dropout=0.0, activation=relu | 0.47 ± 0.08 (n=2) | 0.00 ± 0.00 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.08 ± 0.10 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=128, n_layers=3, norm=none, dropout=0.0, activation=gelu | 0.42 ± 0.00 (n=2) | 0.03 ± 0.03 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.12 ± 0.02 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=128, n_layers=3, norm=none, dropout=0.1, activation=relu | 0.50 ± 0.09 (n=2) | 0.03 ± 0.03 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.05 ± 0.10 (n=2) |
| dataset=adult, variant=raw_onehot, d_hidden=128, n_layers=3, norm=none, dropout=0.1, activation=gelu | 0.44 ± 0.02 (n=2) | 0.03 ± 0.03 (n=2) | 0.55 ± 0.01 (n=2) | 0.22 ± 0.00 (n=2) | -0.11 ± 0.03 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=32, n_layers=1, norm=layernorm, dropout=0.0, activation=relu | 0.44 ± 0.03 (n=2) | 0.00 ± 0.00 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.06 ± 0.00 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=32, n_layers=1, norm=layernorm, dropout=0.0, activation=gelu | 0.49 ± 0.03 (n=2) | 0.09 ± 0.09 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.11 ± 0.06 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=32, n_layers=1, norm=layernorm, dropout=0.1, activation=relu | 0.38 ± 0.02 (n=2) | 0.03 ± 0.03 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.00 ± 0.05 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=32, n_layers=1, norm=layernorm, dropout=0.1, activation=gelu | 0.49 ± 0.03 (n=2) | 0.12 ± 0.12 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.11 ± 0.06 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=32, n_layers=1, norm=batchnorm, dropout=0.0, activation=relu | 0.34 ± 0.02 (n=2) | 0.00 ± 0.00 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | -0.04 ± 0.01 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=32, n_layers=1, norm=batchnorm, dropout=0.0, activation=gelu | 0.52 ± 0.09 (n=2) | 0.12 ± 0.12 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.14 ± 0.12 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=32, n_layers=1, norm=batchnorm, dropout=0.1, activation=relu | 0.36 ± 0.01 (n=2) | 0.00 ± 0.00 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | -0.01 ± 0.04 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=32, n_layers=1, norm=batchnorm, dropout=0.1, activation=gelu | 0.56 ± 0.15 (n=2) | 0.22 ± 0.22 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.18 ± 0.18 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=32, n_layers=1, norm=none, dropout=0.0, activation=relu | 0.43 ± 0.09 (n=2) | 0.00 ± 0.00 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.05 ± 0.06 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=32, n_layers=1, norm=none, dropout=0.0, activation=gelu | 0.46 ± 0.05 (n=2) | 0.00 ± 0.00 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.08 ± 0.08 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=32, n_layers=1, norm=none, dropout=0.1, activation=relu | 0.51 ± 0.02 (n=2) | 0.06 ± 0.06 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.14 ± 0.01 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=32, n_layers=1, norm=none, dropout=0.1, activation=gelu | 0.47 ± 0.07 (n=2) | 0.00 ± 0.00 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.10 ± 0.10 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=32, n_layers=3, norm=layernorm, dropout=0.0, activation=relu | 0.40 ± 0.03 (n=2) | 0.06 ± 0.00 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.03 ± 0.00 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=32, n_layers=3, norm=layernorm, dropout=0.0, activation=gelu | 0.58 ± 0.05 (n=2) | 0.28 ± 0.03 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.20 ± 0.09 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=32, n_layers=3, norm=layernorm, dropout=0.1, activation=relu | 0.45 ± 0.06 (n=2) | 0.06 ± 0.00 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.07 ± 0.03 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=32, n_layers=3, norm=layernorm, dropout=0.1, activation=gelu | 0.55 ± 0.05 (n=2) | 0.25 ± 0.06 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.17 ± 0.09 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=32, n_layers=3, norm=batchnorm, dropout=0.0, activation=relu | 0.46 ± 0.02 (n=2) | 0.00 ± 0.00 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.08 ± 0.01 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=32, n_layers=3, norm=batchnorm, dropout=0.0, activation=gelu | 0.64 ± 0.05 (n=2) | 0.31 ± 0.06 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.27 ± 0.02 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=32, n_layers=3, norm=batchnorm, dropout=0.1, activation=relu | 0.51 ± 0.02 (n=2) | 0.09 ± 0.03 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.14 ± 0.01 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=32, n_layers=3, norm=batchnorm, dropout=0.1, activation=gelu | 0.70 ± 0.02 (n=2) | 0.38 ± 0.06 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.32 ± 0.01 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=32, n_layers=3, norm=none, dropout=0.0, activation=relu | 0.44 ± 0.07 (n=2) | 0.00 ± 0.00 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.06 ± 0.04 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=32, n_layers=3, norm=none, dropout=0.0, activation=gelu | 0.52 ± 0.10 (n=2) | 0.09 ± 0.09 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.14 ± 0.13 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=32, n_layers=3, norm=none, dropout=0.1, activation=relu | 0.47 ± 0.03 (n=2) | 0.00 ± 0.00 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.10 ± 0.00 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=32, n_layers=3, norm=none, dropout=0.1, activation=gelu | 0.48 ± 0.07 (n=2) | 0.09 ± 0.09 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.11 ± 0.11 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=128, n_layers=1, norm=layernorm, dropout=0.0, activation=relu | 0.47 ± 0.04 (n=2) | 0.09 ± 0.03 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.10 ± 0.00 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=128, n_layers=1, norm=layernorm, dropout=0.0, activation=gelu | 0.48 ± 0.07 (n=2) | 0.09 ± 0.03 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.11 ± 0.10 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=128, n_layers=1, norm=layernorm, dropout=0.1, activation=relu | 0.47 ± 0.05 (n=2) | 0.12 ± 0.06 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.10 ± 0.08 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=128, n_layers=1, norm=layernorm, dropout=0.1, activation=gelu | 0.44 ± 0.01 (n=2) | 0.09 ± 0.03 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.07 ± 0.02 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=128, n_layers=1, norm=batchnorm, dropout=0.0, activation=relu | 0.39 ± 0.06 (n=2) | 0.00 ± 0.00 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.02 ± 0.03 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=128, n_layers=1, norm=batchnorm, dropout=0.0, activation=gelu | 0.53 ± 0.03 (n=2) | 0.03 ± 0.03 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.15 ± 0.06 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=128, n_layers=1, norm=batchnorm, dropout=0.1, activation=relu | 0.46 ± 0.00 (n=2) | 0.03 ± 0.03 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.09 ± 0.03 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=128, n_layers=1, norm=batchnorm, dropout=0.1, activation=gelu | 0.45 ± 0.04 (n=2) | 0.00 ± 0.00 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.08 ± 0.01 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=128, n_layers=1, norm=none, dropout=0.0, activation=relu | 0.49 ± 0.11 (n=2) | 0.12 ± 0.12 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.12 ± 0.14 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=128, n_layers=1, norm=none, dropout=0.0, activation=gelu | 0.51 ± 0.01 (n=2) | 0.00 ± 0.00 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.13 ± 0.02 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=128, n_layers=1, norm=none, dropout=0.1, activation=relu | 0.44 ± 0.08 (n=2) | 0.03 ± 0.03 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.06 ± 0.11 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=128, n_layers=1, norm=none, dropout=0.1, activation=gelu | 0.48 ± 0.04 (n=2) | 0.00 ± 0.00 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.10 ± 0.01 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=128, n_layers=3, norm=layernorm, dropout=0.0, activation=relu | 0.49 ± 0.02 (n=2) | 0.22 ± 0.03 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.12 ± 0.05 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=128, n_layers=3, norm=layernorm, dropout=0.0, activation=gelu | 0.62 ± 0.02 (n=2) | 0.31 ± 0.06 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.25 ± 0.05 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=128, n_layers=3, norm=layernorm, dropout=0.1, activation=relu | 0.51 ± 0.01 (n=2) | 0.25 ± 0.06 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.13 ± 0.02 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=128, n_layers=3, norm=layernorm, dropout=0.1, activation=gelu | 0.61 ± 0.02 (n=2) | 0.28 ± 0.03 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.23 ± 0.05 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=128, n_layers=3, norm=batchnorm, dropout=0.0, activation=relu | 0.65 ± 0.01 (n=2) | 0.28 ± 0.03 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.27 ± 0.04 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=128, n_layers=3, norm=batchnorm, dropout=0.0, activation=gelu | 0.64 ± 0.12 (n=2) | 0.09 ± 0.09 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.27 ± 0.09 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=128, n_layers=3, norm=batchnorm, dropout=0.1, activation=relu | 0.60 ± 0.05 (n=2) | 0.16 ± 0.09 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.23 ± 0.09 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=128, n_layers=3, norm=batchnorm, dropout=0.1, activation=gelu | 0.70 ± 0.05 (n=2) | 0.41 ± 0.09 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.32 ± 0.08 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=128, n_layers=3, norm=none, dropout=0.0, activation=relu | 0.60 ± 0.00 (n=2) | 0.12 ± 0.06 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.22 ± 0.03 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=128, n_layers=3, norm=none, dropout=0.0, activation=gelu | 0.53 ± 0.01 (n=2) | 0.09 ± 0.09 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.15 ± 0.02 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=128, n_layers=3, norm=none, dropout=0.1, activation=relu | 0.61 ± 0.04 (n=2) | 0.12 ± 0.00 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.24 ± 0.01 (n=2) |
| dataset=california, variant=raw_onehot, d_hidden=128, n_layers=3, norm=none, dropout=0.1, activation=gelu | 0.53 ± 0.05 (n=2) | 0.12 ± 0.00 (n=2) | 0.38 ± 0.03 (n=2) | 0.15 ± 0.01 (n=2) | 0.15 ± 0.02 (n=2) |

Command: `python -m examples.gia.tabular.leakage_factors.architecture --out examples/gia/outputs/architecture --device cpu --seeds 0 1 --max-iter 2000`; torch `2.5.1+cu121`, Quadro RTX 5000.
<!-- /results -->

Highest TabLeak accuracy: dataset=california, variant=raw_onehot, d_hidden=128, n_layers=3, norm=batchnorm, dropout=0.1, activation=gelu (0.70); lowest: dataset=california, variant=raw_onehot, d_hidden=32, n_layers=1, norm=batchnorm, dropout=0.0, activation=relu (0.34).

<!-- results:attack_iterations -->
| Attack | tableak_acc | exact_row_recovery | prior_accuracy | random_accuracy | gain_over_prior |
|---|---|---|---|---|---|
| dataset=adult, variant=raw_onehot, max_iter=1000, batch_size=8 | 0.47 ± 0.04 (n=3) | 0.04 ± 0.06 (n=3) | 0.58 ± 0.04 (n=3) | 0.19 ± 0.01 (n=3) | -0.11 ± 0.00 (n=3) |
| dataset=adult, variant=raw_onehot, max_iter=1000, batch_size=32 | 0.38 ± 0.02 (n=3) | 0.00 ± 0.00 (n=3) | 0.54 ± 0.01 (n=3) | 0.23 ± 0.00 (n=3) | -0.17 ± 0.01 (n=3) |
| dataset=adult, variant=raw_onehot, max_iter=3000, batch_size=8 | 0.58 ± 0.09 (n=3) | 0.17 ± 0.06 (n=3) | 0.58 ± 0.04 (n=3) | 0.19 ± 0.01 (n=3) | -0.00 ± 0.05 (n=3) |
| dataset=adult, variant=raw_onehot, max_iter=3000, batch_size=32 | 0.40 ± 0.02 (n=3) | 0.00 ± 0.00 (n=3) | 0.54 ± 0.01 (n=3) | 0.23 ± 0.00 (n=3) | -0.14 ± 0.02 (n=3) |
| dataset=california, variant=raw_onehot, max_iter=1000, batch_size=8 | 0.70 ± 0.03 (n=3) | 0.33 ± 0.06 (n=3) | 0.42 ± 0.03 (n=3) | 0.13 ± 0.00 (n=3) | 0.28 ± 0.00 (n=3) |
| dataset=california, variant=raw_onehot, max_iter=1000, batch_size=32 | 0.49 ± 0.03 (n=3) | 0.07 ± 0.04 (n=3) | 0.41 ± 0.03 (n=3) | 0.16 ± 0.00 (n=3) | 0.09 ± 0.01 (n=3) |
| dataset=california, variant=raw_onehot, max_iter=3000, batch_size=8 | 0.78 ± 0.03 (n=3) | 0.46 ± 0.16 (n=3) | 0.42 ± 0.03 (n=3) | 0.13 ± 0.00 (n=3) | 0.36 ± 0.06 (n=3) |
| dataset=california, variant=raw_onehot, max_iter=3000, batch_size=32 | 0.57 ± 0.06 (n=3) | 0.18 ± 0.04 (n=3) | 0.41 ± 0.03 (n=3) | 0.16 ± 0.00 (n=3) | 0.17 ± 0.04 (n=3) |

Command: `python -m examples.gia.tabular.leakage_factors.attack_iterations --out examples/gia/outputs/attack_iterations --device cpu --seeds 0 1 2 --iterations 1000 3000`; torch `2.5.1+cu121`, Quadro RTX 5000.
<!-- /results -->

Highest TabLeak accuracy: dataset=california, variant=raw_onehot, max_iter=3000, batch_size=8 (0.78); lowest: dataset=adult, variant=raw_onehot, max_iter=1000, batch_size=32 (0.38).

<!-- results:batch_sizes -->
| Attack | tableak_acc | exact_row_recovery | prior_accuracy | random_accuracy | gain_over_prior |
|---|---|---|---|---|---|
| dataset=adult, variant=raw_onehot, model=small, batch_size=1 | 1.00 ± 0.00 (n=3) | 1.00 ± 0.00 (n=3) | 0.60 ± 0.09 (n=3) | 0.13 ± 0.02 (n=3) | 0.40 ± 0.09 (n=3) |
| dataset=adult, variant=raw_onehot, model=small, batch_size=4 | 0.78 ± 0.04 (n=3) | 0.42 ± 0.12 (n=3) | 0.51 ± 0.02 (n=3) | 0.17 ± 0.01 (n=3) | 0.27 ± 0.04 (n=3) |
| dataset=adult, variant=raw_onehot, model=small, batch_size=16 | 0.45 ± 0.04 (n=3) | 0.02 ± 0.03 (n=3) | 0.53 ± 0.02 (n=3) | 0.21 ± 0.00 (n=3) | -0.09 ± 0.05 (n=3) |
| dataset=adult, variant=raw_onehot, model=small, batch_size=64 | 0.36 ± 0.02 (n=3) | 0.00 ± 0.00 (n=3) | 0.54 ± 0.02 (n=3) | 0.25 ± 0.00 (n=3) | -0.17 ± 0.04 (n=3) |
| dataset=adult, variant=softmax, model=small, batch_size=1 | 0.98 ± 0.03 (n=3) | 0.67 ± 0.47 (n=3) | 0.60 ± 0.09 (n=3) | 0.13 ± 0.02 (n=3) | 0.38 ± 0.07 (n=3) |
| dataset=adult, variant=softmax, model=small, batch_size=4 | 0.80 ± 0.07 (n=3) | 0.50 ± 0.00 (n=3) | 0.51 ± 0.02 (n=3) | 0.17 ± 0.01 (n=3) | 0.30 ± 0.05 (n=3) |
| dataset=adult, variant=softmax, model=small, batch_size=16 | 0.39 ± 0.02 (n=3) | 0.02 ± 0.03 (n=3) | 0.53 ± 0.02 (n=3) | 0.21 ± 0.00 (n=3) | -0.15 ± 0.00 (n=3) |
| dataset=adult, variant=softmax, model=small, batch_size=64 | 0.37 ± 0.04 (n=3) | 0.00 ± 0.00 (n=3) | 0.54 ± 0.02 (n=3) | 0.25 ± 0.00 (n=3) | -0.16 ± 0.03 (n=3) |
| dataset=california, variant=raw_onehot, model=small, batch_size=1 | 0.38 ± 0.44 (n=3) | 0.33 ± 0.47 (n=3) | 0.42 ± 0.06 (n=3) | 0.07 ± 0.02 (n=3) | -0.04 ± 0.48 (n=3) |
| dataset=california, variant=raw_onehot, model=small, batch_size=4 | 0.79 ± 0.04 (n=3) | 0.58 ± 0.24 (n=3) | 0.41 ± 0.03 (n=3) | 0.10 ± 0.01 (n=3) | 0.39 ± 0.06 (n=3) |
| dataset=california, variant=raw_onehot, model=small, batch_size=16 | 0.60 ± 0.03 (n=3) | 0.21 ± 0.03 (n=3) | 0.40 ± 0.04 (n=3) | 0.15 ± 0.01 (n=3) | 0.20 ± 0.05 (n=3) |
| dataset=california, variant=raw_onehot, model=small, batch_size=64 | 0.48 ± 0.02 (n=3) | 0.08 ± 0.03 (n=3) | 0.42 ± 0.02 (n=3) | 0.18 ± 0.00 (n=3) | 0.06 ± 0.02 (n=3) |
| dataset=california, variant=softmax, model=small, batch_size=1 | 0.38 ± 0.44 (n=3) | 0.33 ± 0.47 (n=3) | 0.42 ± 0.06 (n=3) | 0.07 ± 0.02 (n=3) | -0.04 ± 0.47 (n=3) |
| dataset=california, variant=softmax, model=small, batch_size=4 | 0.89 ± 0.08 (n=3) | 0.67 ± 0.31 (n=3) | 0.41 ± 0.03 (n=3) | 0.10 ± 0.01 (n=3) | 0.48 ± 0.06 (n=3) |
| dataset=california, variant=softmax, model=small, batch_size=16 | 0.52 ± 0.02 (n=3) | 0.17 ± 0.08 (n=3) | 0.40 ± 0.04 (n=3) | 0.15 ± 0.01 (n=3) | 0.12 ± 0.04 (n=3) |
| dataset=california, variant=softmax, model=small, batch_size=64 | 0.51 ± 0.05 (n=3) | 0.06 ± 0.04 (n=3) | 0.42 ± 0.02 (n=3) | 0.18 ± 0.00 (n=3) | 0.09 ± 0.03 (n=3) |

Command: `python -m examples.gia.tabular.leakage_factors.batch_sizes --out examples/gia/outputs/batch_sizes --device cpu --seeds 0 1 2 --max-iter 2000 --batch-sizes 1 4 16 64`; torch `2.5.1+cu121`, Quadro RTX 5000.
<!-- /results -->

Highest TabLeak accuracy: dataset=adult, variant=raw_onehot, model=small, batch_size=1 (1.00); lowest: dataset=adult, variant=raw_onehot, model=small, batch_size=64 (0.36).

<!-- results:leakage_fedavg -->
| Attack | tableak_acc | exact_row_recovery | prior_accuracy | random_accuracy | gain_over_prior |
|---|---|---|---|---|---|
| dataset=adult, variant=raw_onehot, local_epochs=1, batch_size=8 | 0.39 ± 0.05 (n=3) | 0.00 ± 0.00 (n=3) | 0.54 ± 0.01 (n=3) | 0.23 ± 0.00 (n=3) | -0.16 ± 0.05 (n=3) |
| dataset=adult, variant=raw_onehot, local_epochs=1, batch_size=16 | 0.40 ± 0.03 (n=3) | 0.00 ± 0.00 (n=3) | 0.54 ± 0.01 (n=3) | 0.23 ± 0.00 (n=3) | -0.15 ± 0.02 (n=3) |
| dataset=adult, variant=raw_onehot, local_epochs=1, batch_size=32 | 0.38 ± 0.02 (n=3) | 0.00 ± 0.00 (n=3) | 0.54 ± 0.01 (n=3) | 0.23 ± 0.00 (n=3) | -0.16 ± 0.02 (n=3) |
| dataset=adult, variant=raw_onehot, local_epochs=4, batch_size=8 | 0.38 ± 0.01 (n=3) | 0.00 ± 0.00 (n=3) | 0.54 ± 0.01 (n=3) | 0.23 ± 0.00 (n=3) | -0.16 ± 0.01 (n=3) |
| dataset=adult, variant=raw_onehot, local_epochs=4, batch_size=16 | 0.40 ± 0.03 (n=3) | 0.00 ± 0.00 (n=3) | 0.54 ± 0.01 (n=3) | 0.23 ± 0.00 (n=3) | -0.15 ± 0.02 (n=3) |
| dataset=adult, variant=raw_onehot, local_epochs=4, batch_size=32 | 0.38 ± 0.01 (n=3) | 0.00 ± 0.00 (n=3) | 0.54 ± 0.01 (n=3) | 0.23 ± 0.00 (n=3) | -0.16 ± 0.01 (n=3) |
| dataset=california, variant=raw_onehot, local_epochs=1, batch_size=8 | 0.53 ± 0.05 (n=3) | 0.08 ± 0.03 (n=3) | 0.41 ± 0.03 (n=3) | 0.16 ± 0.00 (n=3) | 0.12 ± 0.02 (n=3) |
| dataset=california, variant=raw_onehot, local_epochs=1, batch_size=16 | 0.54 ± 0.06 (n=3) | 0.14 ± 0.01 (n=3) | 0.41 ± 0.03 (n=3) | 0.16 ± 0.00 (n=3) | 0.13 ± 0.03 (n=3) |
| dataset=california, variant=raw_onehot, local_epochs=1, batch_size=32 | 0.56 ± 0.06 (n=3) | 0.16 ± 0.07 (n=3) | 0.41 ± 0.03 (n=3) | 0.16 ± 0.00 (n=3) | 0.15 ± 0.04 (n=3) |
| dataset=california, variant=raw_onehot, local_epochs=4, batch_size=8 | 0.58 ± 0.02 (n=3) | 0.15 ± 0.10 (n=3) | 0.41 ± 0.03 (n=3) | 0.16 ± 0.00 (n=3) | 0.17 ± 0.01 (n=3) |
| dataset=california, variant=raw_onehot, local_epochs=4, batch_size=16 | 0.57 ± 0.04 (n=3) | 0.17 ± 0.01 (n=3) | 0.41 ± 0.03 (n=3) | 0.16 ± 0.00 (n=3) | 0.17 ± 0.06 (n=3) |
| dataset=california, variant=raw_onehot, local_epochs=4, batch_size=32 | 0.56 ± 0.04 (n=3) | 0.09 ± 0.03 (n=3) | 0.41 ± 0.03 (n=3) | 0.16 ± 0.00 (n=3) | 0.15 ± 0.03 (n=3) |

Command: `python -m examples.gia.tabular.leakage_factors.fedavg --out examples/gia/outputs/leakage_fedavg --device cpu --seeds 0 1 2 --max-iter 2000`; torch `2.5.1+cu121`, Quadro RTX 5000.
<!-- /results -->

Highest TabLeak accuracy: dataset=california, variant=raw_onehot, local_epochs=4, batch_size=8 (0.58); lowest: dataset=adult, variant=raw_onehot, local_epochs=4, batch_size=32 (0.38).

<!-- results:label_knowledge -->
| Attack | tableak_acc | exact_row_recovery | prior_accuracy | random_accuracy | gain_over_prior |
|---|---|---|---|---|---|
| dataset=adult, variant=raw_onehot, label_mode=known, batch_size=8 | 0.54 ± 0.07 (n=3) | 0.08 ± 0.06 (n=3) | 0.58 ± 0.04 (n=3) | 0.19 ± 0.01 (n=3) | -0.04 ± 0.04 (n=3) |
| dataset=adult, variant=raw_onehot, label_mode=known, batch_size=32 | 0.38 ± 0.02 (n=3) | 0.00 ± 0.00 (n=3) | 0.54 ± 0.01 (n=3) | 0.23 ± 0.00 (n=3) | -0.16 ± 0.02 (n=3) |
| dataset=adult, variant=raw_onehot, label_mode=zeros, batch_size=8 | 0.28 ± 0.03 (n=3) | 0.00 ± 0.00 (n=3) | 0.58 ± 0.04 (n=3) | 0.19 ± 0.01 (n=3) | -0.30 ± 0.02 (n=3) |
| dataset=adult, variant=raw_onehot, label_mode=zeros, batch_size=32 | 0.27 ± 0.02 (n=3) | 0.00 ± 0.00 (n=3) | 0.54 ± 0.01 (n=3) | 0.23 ± 0.00 (n=3) | -0.27 ± 0.02 (n=3) |
| dataset=adult, variant=raw_onehot, label_mode=joint, batch_size=8 | 0.47 ± 0.02 (n=3) | 0.08 ± 0.06 (n=3) | 0.58 ± 0.04 (n=3) | 0.19 ± 0.01 (n=3) | -0.11 ± 0.04 (n=3) |
| dataset=adult, variant=raw_onehot, label_mode=joint, batch_size=32 | 0.42 ± 0.01 (n=3) | 0.00 ± 0.00 (n=3) | 0.54 ± 0.01 (n=3) | 0.23 ± 0.00 (n=3) | -0.13 ± 0.01 (n=3) |
| dataset=california, variant=raw_onehot, label_mode=known, batch_size=8 | 0.77 ± 0.03 (n=3) | 0.46 ± 0.16 (n=3) | 0.42 ± 0.03 (n=3) | 0.13 ± 0.00 (n=3) | 0.35 ± 0.05 (n=3) |
| dataset=california, variant=raw_onehot, label_mode=known, batch_size=32 | 0.56 ± 0.06 (n=3) | 0.16 ± 0.07 (n=3) | 0.41 ± 0.03 (n=3) | 0.16 ± 0.00 (n=3) | 0.15 ± 0.04 (n=3) |
| dataset=california, variant=raw_onehot, label_mode=zeros, batch_size=8 | 0.39 ± 0.21 (n=3) | 0.04 ± 0.06 (n=3) | 0.42 ± 0.03 (n=3) | 0.13 ± 0.00 (n=3) | -0.03 ± 0.18 (n=3) |
| dataset=california, variant=raw_onehot, label_mode=zeros, batch_size=32 | 0.34 ± 0.10 (n=3) | 0.02 ± 0.03 (n=3) | 0.41 ± 0.03 (n=3) | 0.16 ± 0.00 (n=3) | -0.07 ± 0.08 (n=3) |
| dataset=california, variant=raw_onehot, label_mode=joint, batch_size=8 | 0.80 ± 0.01 (n=3) | 0.71 ± 0.06 (n=3) | 0.42 ± 0.03 (n=3) | 0.13 ± 0.00 (n=3) | 0.39 ± 0.03 (n=3) |
| dataset=california, variant=raw_onehot, label_mode=joint, batch_size=32 | 0.60 ± 0.07 (n=3) | 0.25 ± 0.12 (n=3) | 0.41 ± 0.03 (n=3) | 0.16 ± 0.00 (n=3) | 0.19 ± 0.06 (n=3) |

Command: `python -m examples.gia.tabular.leakage_factors.label_knowledge --out examples/gia/outputs/label_knowledge --device cpu --seeds 0 1 2 --max-iter 2000`; torch `2.5.1+cu121`, Quadro RTX 5000.
<!-- /results -->

Highest TabLeak accuracy: dataset=california, variant=raw_onehot, label_mode=joint, batch_size=8 (0.80); lowest: dataset=adult, variant=raw_onehot, label_mode=zeros, batch_size=32 (0.27).

<!-- results:training_exposure -->
| Attack | tableak_acc | exact_row_recovery | prior_accuracy | random_accuracy | gain_over_prior |
|---|---|---|---|---|---|
| dataset=adult, variant=raw_onehot, epochs=0, batch_size=8 | 0.77 ± 0.01 (n=3) | 0.21 ± 0.06 (n=3) | 0.58 ± 0.04 (n=3) | 0.19 ± 0.01 (n=3) | 0.19 ± 0.05 (n=3) |
| dataset=adult, variant=raw_onehot, epochs=0, batch_size=32 | 0.42 ± 0.02 (n=3) | 0.00 ± 0.00 (n=3) | 0.54 ± 0.01 (n=3) | 0.23 ± 0.00 (n=3) | -0.12 ± 0.02 (n=3) |
| dataset=adult, variant=raw_onehot, epochs=1, batch_size=8 | 0.54 ± 0.07 (n=3) | 0.08 ± 0.06 (n=3) | 0.58 ± 0.04 (n=3) | 0.19 ± 0.01 (n=3) | -0.04 ± 0.04 (n=3) |
| dataset=adult, variant=raw_onehot, epochs=1, batch_size=32 | 0.38 ± 0.02 (n=3) | 0.00 ± 0.00 (n=3) | 0.54 ± 0.01 (n=3) | 0.23 ± 0.00 (n=3) | -0.16 ± 0.02 (n=3) |
| dataset=adult, variant=raw_onehot, epochs=5, batch_size=8 | 0.40 ± 0.06 (n=3) | 0.00 ± 0.00 (n=3) | 0.58 ± 0.04 (n=3) | 0.19 ± 0.01 (n=3) | -0.18 ± 0.04 (n=3) |
| dataset=adult, variant=raw_onehot, epochs=5, batch_size=32 | 0.41 ± 0.02 (n=3) | 0.00 ± 0.00 (n=3) | 0.54 ± 0.01 (n=3) | 0.23 ± 0.00 (n=3) | -0.14 ± 0.01 (n=3) |
| dataset=adult, variant=raw_onehot, epochs=10, batch_size=8 | 0.43 ± 0.03 (n=3) | 0.04 ± 0.06 (n=3) | 0.58 ± 0.04 (n=3) | 0.19 ± 0.01 (n=3) | -0.15 ± 0.02 (n=3) |
| dataset=adult, variant=raw_onehot, epochs=10, batch_size=32 | 0.41 ± 0.01 (n=3) | 0.01 ± 0.01 (n=3) | 0.54 ± 0.01 (n=3) | 0.23 ± 0.00 (n=3) | -0.13 ± 0.00 (n=3) |
| dataset=adult, variant=raw_onehot, epochs=25, batch_size=8 | 0.43 ± 0.03 (n=3) | 0.04 ± 0.06 (n=3) | 0.58 ± 0.04 (n=3) | 0.19 ± 0.01 (n=3) | -0.16 ± 0.05 (n=3) |
| dataset=adult, variant=raw_onehot, epochs=25, batch_size=32 | 0.40 ± 0.02 (n=3) | 0.01 ± 0.01 (n=3) | 0.54 ± 0.01 (n=3) | 0.23 ± 0.00 (n=3) | -0.15 ± 0.02 (n=3) |
| dataset=california, variant=raw_onehot, epochs=0, batch_size=8 | 0.72 ± 0.07 (n=3) | 0.33 ± 0.21 (n=3) | 0.42 ± 0.03 (n=3) | 0.13 ± 0.00 (n=3) | 0.31 ± 0.09 (n=3) |
| dataset=california, variant=raw_onehot, epochs=0, batch_size=32 | 0.58 ± 0.04 (n=3) | 0.16 ± 0.05 (n=3) | 0.41 ± 0.03 (n=3) | 0.16 ± 0.00 (n=3) | 0.18 ± 0.02 (n=3) |
| dataset=california, variant=raw_onehot, epochs=1, batch_size=8 | 0.77 ± 0.03 (n=3) | 0.46 ± 0.16 (n=3) | 0.42 ± 0.03 (n=3) | 0.13 ± 0.00 (n=3) | 0.35 ± 0.05 (n=3) |
| dataset=california, variant=raw_onehot, epochs=1, batch_size=32 | 0.56 ± 0.06 (n=3) | 0.16 ± 0.07 (n=3) | 0.41 ± 0.03 (n=3) | 0.16 ± 0.00 (n=3) | 0.15 ± 0.04 (n=3) |
| dataset=california, variant=raw_onehot, epochs=5, batch_size=8 | 0.71 ± 0.14 (n=3) | 0.42 ± 0.06 (n=3) | 0.42 ± 0.03 (n=3) | 0.13 ± 0.00 (n=3) | 0.29 ± 0.12 (n=3) |
| dataset=california, variant=raw_onehot, epochs=5, batch_size=32 | 0.45 ± 0.04 (n=3) | 0.14 ± 0.06 (n=3) | 0.41 ± 0.03 (n=3) | 0.16 ± 0.00 (n=3) | 0.04 ± 0.04 (n=3) |
| dataset=california, variant=raw_onehot, epochs=10, batch_size=8 | 0.62 ± 0.10 (n=3) | 0.38 ± 0.18 (n=3) | 0.42 ± 0.03 (n=3) | 0.13 ± 0.00 (n=3) | 0.21 ± 0.11 (n=3) |
| dataset=california, variant=raw_onehot, epochs=10, batch_size=32 | 0.48 ± 0.05 (n=3) | 0.08 ± 0.05 (n=3) | 0.41 ± 0.03 (n=3) | 0.16 ± 0.00 (n=3) | 0.07 ± 0.02 (n=3) |
| dataset=california, variant=raw_onehot, epochs=25, batch_size=8 | 0.56 ± 0.14 (n=3) | 0.33 ± 0.21 (n=3) | 0.42 ± 0.03 (n=3) | 0.13 ± 0.00 (n=3) | 0.14 ± 0.15 (n=3) |
| dataset=california, variant=raw_onehot, epochs=25, batch_size=32 | 0.51 ± 0.02 (n=3) | 0.15 ± 0.06 (n=3) | 0.41 ± 0.03 (n=3) | 0.16 ± 0.00 (n=3) | 0.10 ± 0.03 (n=3) |

Command: `python -m examples.gia.tabular.leakage_factors.training_exposure --out examples/gia/outputs/training_exposure --device cpu --seeds 0 1 2 --max-iter 2000`; torch `2.5.1+cu121`, Quadro RTX 5000.
<!-- /results -->

Highest TabLeak accuracy: dataset=adult, variant=raw_onehot, epochs=0, batch_size=8 (0.77); lowest: dataset=adult, variant=raw_onehot, epochs=1, batch_size=32 (0.38).
