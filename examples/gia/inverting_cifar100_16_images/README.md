# Inverting Gradients on CIFAR-100 (Batch of 16 Images)

This example scales up the **Inverting Gradients** attack (Geiping et al., 2020) from a single image to a **batch of 16 images** on the harder **CIFAR-100** dataset.

## Goal

Show that gradient inversion remains effective on larger batches and more classes, provided the model has been given a reasonable starting point. Batch reconstruction is significantly harder than single-image reconstruction, so this example also pre-trains the target model before attacking it.

## Setup

- **Model:** A ResNet (`BasicBlock`, CIFAR-style) defined in [model.py](model.py).
- **Data:** 16 CIFAR-100 images, loaded via [cifar100.py](cifar100.py). Data augmentation (random cropping and horizontal flipping) is used when generating the client's training batch — without it, and without pre-training, the attack fails to reconstruct batches of this size.
- **Pre-training:** [pre_train.py](pre_train.py) trains the model for a number of epochs first (cached to `model_epochs_<N>.pth` so repeated runs skip re-training).
- **Attack:** `InvertingGradients` (`leakpro.attacks.gia_attacks.invertinggradients`), with `InvertingConfig` and Optuna-based hyperparameter search (`OptunaConfig`).

## How to Run

```bash
python main.py
```

The first run pre-trains and saves the model; subsequent runs load the cached weights and go straight to the attack.

## Credits

- Geiping, J., et al. (2020). [Inverting gradients - How easy is it to break privacy in federated learning?](https://arxiv.org/abs/2003.14053) NeurIPS.
