"""Hyperparameter tuning with optuna on evaluating."""

from cifar import get_cifar10_loader

from leakpro.fl_utils.gia_train import train
from leakpro.run import huang_optuna
from model import ResNet, PreActBlock

if __name__ == "__main__":
    # Instantiate the base model and get the CIFAR-10 loader

    # Pre activation required for the attack to give decent results
    base_model = ResNet(PreActBlock, [2, 2, 2, 2], num_classes=10)
    cifar10_loader, mean, std = get_cifar10_loader(num_images=16, batch_size=16, num_workers=2)

    # Run Optuna optimization with Huang
    huang_optuna(base_model, cifar10_loader, train, mean, std)