# Huang Attack Hyperparameter Tuning with Optuna

This example shows how to use **Optuna** to automatically tune the hyperparameters of the **Huang et al., 2021** gradient inversion attack, rather than setting them by hand.

## Goal

Gradient inversion attacks are sensitive to hyperparameters (learning rate, regularization weights, etc.), and good settings vary by model/dataset. This example demonstrates LeakPro's built-in Optuna integration (`run_with_optuna`) to search for strong attack hyperparameters on CIFAR-10.

## Setup

- **Model:** A pre-activation ResNet (`PreActBlock`) defined in [model.py](model.py) — required for this attack to give decent results.
- **Data:** 16 CIFAR-10 images, loaded via [cifar.py](cifar.py).
- **Attack:** `Huang` (`leakpro.attacks.gia_attacks.huang`), run with `attack_object.run_with_optuna()`.

## How to Run

```bash
python main.py
```

## Credits

- Huang, Y., et al. (2021). [Evaluating Gradient Inversion Attacks and Defenses in Federated Learning.](https://arxiv.org/abs/2112.00059) NeurIPS.
