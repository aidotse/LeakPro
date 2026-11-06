# Legacy GIA examples

These use the pre-modular classes in `leakpro/attacks/gia_attacks/*.py`; new work uses `examples/gia/getting_started`.
Run each example from its own folder (datasets are downloaded to `./data`). `check_imports.py` verifies that all LeakPro imports still resolve.

| folder | API (legacy classes, with file) | data | status |
|--------|---------------------------------|------|--------|
| `inverting_cifar10_1_image` | `InvertingGradients` (`invertinggradients.py`) | CIFAR-10 | runs |
| `inverting_cifar100_16_images` | `InvertingGradients` (`invertinggradients.py`) | CIFAR-100 | runs |
| `inverting_celebA_1_image` | `InvertingGradients` (`invertinggradients.py`) | CelebA | needs manual dataset |
| `huang_cifar10_16_images` | `Huang` (`huang.py`) | CIFAR-10 | runs |
| `huang_optuna` | `Huang` (`huang.py`) + Optuna | CIFAR-10 | runs |
| `GIA_base` | `GIABase` (`gia_estimate.py`) | CIFAR-10 | runs |
| `GIA_base_running` | `GIABaseRunning` (`gia_running.py`) | CIFAR-10 | runs |
| `GIA_base_convnext_imagenet` | `GIABase` (`gia_estimate.py`) | ImageNet subset | needs GPU |
| `evaluating_gia_train` | `fl_utils/gia_train.py` | CIFAR-10 | runs |
| `pii_inverting_masked_text` | `InvertingGradients` (`invertinggradients_text.py`) | ECHR json | needs manual dataset |
