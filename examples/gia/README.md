# Gradient Inversion Attack (GIA) Examples

Examples of gradient inversion attacks, which try to reconstruct a client's private training data from the gradient(s) it shares during federated learning.

**Start here:** [gia_modular_basic](gia_modular_basic/README.md) — the current, recommended modular framework for composing and running GIA attacks. The rest of the examples below predate it and each hard-code a single classic attack.

## Images (CIFAR / CelebA)

| Example | Attack | Notes |
|---|---|---|
| [inverting_cifar10_1_image](inverting_cifar10_1_image/README.md) | Inverting Gradients (Geiping et al.) | Baseline: single image |
| [inverting_cifar100_16_images](inverting_cifar100_16_images/README.md) | Inverting Gradients (Geiping et al.) | Batch of 16, with pre-training |
| [inverting_celebA_1_image](inverting_celebA_1_image/README.md) | Inverting Gradients (Geiping et al.) | Multi-label attribute classifier |
| [huang_cifar10_16_images](huang_cifar10_16_images/README.md) | Huang et al. | BN-statistics attack, batch of 16 |
| [huang_optuna](huang_optuna/README.md) | Huang et al. | Optuna hyperparameter tuning |
| [GIA_base](GIA_base/README.md) | GIA Estimate (Huang variant) | BN stats estimated from proxy data |
| [GIA_base_running](GIA_base_running/README.md) | GIA Running (Huang variant) | BN stats inferred from momentum updates |
| [GIA_base_convnext_imagenet](GIA_base_convnext_imagenet/README.md) | GIA Estimate | ConvNeXt on ImageNet |
| [coco_yolo_1_image](coco_yolo_1_image/README.md) | Inverting Gradients (Geiping et al.) | Modified YOLO on COCO |

## Text

| Example | Attack | Notes |
|---|---|---|
| [pii_inverting_masked_text](pii_inverting_masked_text/README.md) | Inverting Gradients (Geiping et al.), text variant | PII leakage from a Longformer NER model |

## Other

| Example | Purpose |
|---|---|
| [gia_modular_basic](gia_modular_basic/README.md) | Modular GIA framework overview and usage (recommended entry point) |
| [evaluating_gia_train/AMM_kit](evaluating_gia_train/AMM_kit/README.md) | Validates LeakPro's differentiable training loop matches plain PyTorch |
| [bn_research](bn_research/README.md) | Experimental, not maintained — slated for removal |
