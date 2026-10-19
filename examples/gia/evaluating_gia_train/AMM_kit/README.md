# LeakPro vs. PyTorch Training Equivalence Check

This is a validation/testing utility, not an attack demo. It checks that **LeakPro's meta-train function** used internally by GIA attacks (`leakpro.fl_utils.gia_train.train`, which needs to be differentiable through the training step for gradient inversion) produces the **same results as a standard PyTorch training loop** ([train_scaleout.py](train_scaleout.py)).

## Goal

GIA attacks rely on simulating a client's local training step in a way that stays differentiable (e.g. using functional/meta-learning style optimizers such as `MetaSGD`). This script ([pytorch_leakpro_train_comparison.py](pytorch_leakpro_train_comparison.py)) runs both the LeakPro meta-train function and an equivalent plain PyTorch training loop on the same model/data/seed, and compares the resulting weights/gradients (e.g. via L2 norm) to confirm they match. This guards against the meta-train reimplementation silently diverging from real PyTorch training semantics.

## Setup

- **Models:** `CNN` and `ResNet` ([model.py](model.py)).
- **Data:** CIFAR-10, loaded via [cifar.py](cifar.py).
- **Comparison notebook:** [test.ipynb](test.ipynb) provides an interactive version of the same check.

## How to Run

```bash
python pytorch_leakpro_train_comparison.py
```

or open [test.ipynb](test.ipynb) to run the comparison interactively.
