# Getting started

Run everything from the repository root. `--quick` is an offline CPU smoke run on synthetic data (seconds); omit it for the full experiment.

| Script | What it shows | Quick | Full |
|---|---|---|---|
| `quickstart.py` | Reconstruct a client's images from one gradient with the Inverting Gradients preset, in about 20 lines of attack code. | `python -m examples.gia.getting_started.quickstart --quick` (~5 s) | drop `--quick` (CIFAR-10 downloaded once) |
| `compose_your_own_attack.py` | Build an attack from parts: cosine gradient matching, total variation and a custom loss term. | `python -m examples.gia.getting_started.compose_your_own_attack --quick` (~5 s) | drop `--quick` |
| `hpo_and_split_runs.py` | Tune an attack with Optuna using only attacker-measurable quantities, then run its stages one at a time. Requires `optuna`. | `python -m examples.gia.getting_started.hpo_and_split_runs --quick` (~20 s) | drop `--quick` |

Results are written to `examples/gia/outputs/<script>/` (override with `--out`).
