# GIA Estimate: BN Statistics from Proxy Data

This example demonstrates **GIABase** (a.k.a. "GIA Estimate"), a variant of the **Huang et al., 2021** batch-normalization attack that *estimates* the client's batch norm statistics from a proxy dataset instead of assuming direct access to them.

## Goal

The original Huang attack assumes the attacker can read the client's live batch norm statistics. GIA Estimate relaxes this: it only assumes access to a **proxy dataset** from a similar distribution, and estimates the needed BN statistics from that. This corresponds to a weaker, more realistic threat model (Model D — data-enhanced attacker, in the modular framework's terminology, see [gia_modular_basic](../gia_modular_basic/README.md)).

## Setup

- **Model:** A pre-activation ResNet (`PreActBlock`) defined in [model.py](model.py) — required for this attack to work properly.
- **Data:** 16 CIFAR-10 images as the client's private batch, plus a separate **proxy loader** of 16 different images standing in for the attacker's auxiliary data, both from [cifar.py](cifar.py).
- **Attack:** `GIABase` (`leakpro.attacks.gia_attacks.gia_estimate`), tuned via Optuna (`OptunaConfig`) over a set of trial (client, proxy) loader pairs.

## How to Run

```bash
python main.py
```

## Credits

- Huang, Y., et al. (2021). [Evaluating Gradient Inversion Attacks and Defenses in Federated Learning.](https://arxiv.org/abs/2112.00059) NeurIPS.
