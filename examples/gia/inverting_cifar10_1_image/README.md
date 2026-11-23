# Inverting Gradients on CIFAR-10 (Single Image)

This example demonstrates the classic **gradient inversion attack** from **Geiping et al., 2020: _"Inverting Gradients - How Easy Is It to Break Privacy in Federated Learning?"_**, reconstructing a single CIFAR-10 image from its gradient alone.

## Goal

Show the baseline setting for gradient inversion: given the gradient a single client would send to a federated learning server after one training step on **one image**, reconstruct that image with no auxiliary information beyond a total-variation prior.

## Setup

- **Model:** A ResNet (`BasicBlock`, CIFAR-style) defined in [model.py](model.py).
- **Data:** 1 CIFAR-10 image, loaded via [cifar.py](cifar.py).
- **Attack:** `InvertingGradients` (`leakpro.attacks.gia_attacks.invertinggradients`), configured with `InvertingConfig`.

## How to Run

```bash
python main.py
```

This trains no model — it directly runs the attack via `run_gia_attack`, optimizing a random noise image to match the client's gradient, and reports the reconstruction quality.

## Credits

- Geiping, J., et al. (2020). [Inverting gradients - How easy is it to break privacy in federated learning?](https://arxiv.org/abs/2003.14053) NeurIPS.
