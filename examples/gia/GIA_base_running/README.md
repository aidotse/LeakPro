# GIA Running: Inferred BN Statistics from Momentum Updates

This example demonstrates **GIABaseRunning**, a variant of the **Huang et al., 2021** batch-normalization attack that uses a **different batch norm statistics loss**: instead of reading the client's BN statistics directly or estimating them from proxy data, it *infers* them from the running-statistics momentum updates the client would report.

## Goal

Show a middle-ground threat model between "attacker reads BN stats directly" (original Huang attack) and "attacker only has proxy data" (GIA Estimate): here the attacker knows the client's local training hyperparameters (e.g. BN momentum) and uses that to infer the BN statistics used during the client's training step. This corresponds to Model E (statistical-informed attacker, in the modular framework's terminology, see [gia_modular_basic](../gia_modular_basic/README.md)).

## Setup

- **Model:** A pre-activation ResNet (`PreActBlock`) defined in [model.py](model.py) — required for this attack to work properly.
- **Data:** 16 CIFAR-10 images as the client's private batch, from [cifar.py](cifar.py).
- **Attack:** `GIABaseRunning` (`leakpro.attacks.gia_attacks.gia_running`), tuned via Optuna (`OptunaConfig`) over a set of trial client loaders.

## How to Run

```bash
python main.py
```

## Credits

- Huang, Y., et al. (2021). [Evaluating Gradient Inversion Attacks and Defenses in Federated Learning.](https://arxiv.org/abs/2112.00059) NeurIPS.
