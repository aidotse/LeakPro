# Huang Attack on CIFAR-10 (Batch of 16 Images)

This example demonstrates the **Huang et al., 2021** gradient inversion attack: **_"Evaluating Gradient Inversion Attacks and Defenses in Federated Learning."_**

## Goal

Show a stronger batch-reconstruction attack than plain gradient matching by additionally exploiting the model's **batch normalization statistics**, which leak extra information about the input batch beyond the gradient alone.

## Setup

- **Model:** A pre-activation ResNet (`PreActBlock`) defined in [model.py](model.py) — pre-activation batch norm is required for this attack to work well.
- **Data:** 16 CIFAR-10 images, loaded via [cifar.py](cifar.py).
- **Attack:** `Huang` (`leakpro.attacks.gia_attacks.huang`), tuned via Optuna (`OptunaConfig`) using a set of held-out data loaders (`trial_data`) as validation trials for the search.

## How to Run

```bash
python main.py
```

## Credits

- Huang, Y., et al. (2021). [Evaluating Gradient Inversion Attacks and Defenses in Federated Learning.](https://arxiv.org/abs/2112.00059) NeurIPS.
