# Imagenette experiment

This setup works for most `torchvision.models` models that are designed for imagenet.

# CIFAR-10 architecture study

Experiments for the paper *Practical Feasibility of Gradient Inversion Attacks in Federated Learning*. These scripts are not cleaned up, so use them at your own risk.

`maxvit_cifar.py`, `swint_cifar.py`, `vit_b16_cifar.py` and `model.py` (ConvNeXt) contain `torchvision` architectures adapted for 32x32 CIFAR-10 inputs. To test a different architecture, change the model in `measuremodel.py`. `run.sh` runs `measuremodel.py` in parallel at different levels of noise added to the target image, which the attack uses as its starting point.
