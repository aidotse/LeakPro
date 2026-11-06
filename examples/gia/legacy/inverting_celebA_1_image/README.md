# Inverting Gradients on CelebA (Single Image)

This example applies the **Inverting Gradients** attack (Geiping et al., 2020) to a **multi-label attribute classifier** trained on CelebA face images, instead of the usual single-label CIFAR classifier.

## Goal

Show that gradient inversion is not limited to standard single-label classification: here the target model is a ResNet-18 fine-tuned for CelebA's 40 binary face attributes (`BCEWithLogitsLoss`), and the attack still reconstructs the client's private face image from its gradient.

## Setup

- **Model:** `torchvision.models.resnet18`, with its final layer replaced to output 40 attribute logits.
- **Data:** 1 CelebA image, loaded via [celebA.py](celebA.py).
- **Attack:** `InvertingGradients` (`leakpro.attacks.gia_attacks.invertinggradients`), with a custom `InvertingConfig` tuned for this setting (e.g. `BCEWithLogitsLoss` as the criterion, `top10norms` and `median_pooling` enabled).

## How to Run

```bash
python main.py
```

Note: this requires the CelebA dataset to be available/downloadable via torchvision.

## Credits

- Geiping, J., et al. (2020). [Inverting gradients - How easy is it to break privacy in federated learning?](https://arxiv.org/abs/2003.14053) NeurIPS.
