"""Hyperparameter tuning with optuna on evaluating."""

from cifar import get_cifar10_loader

from leakpro.attacks.gia_attacks.huang import Huang
from model import ResNet, PreActBlock

if __name__ == "__main__":
    # Pre activation required for this attack to give decent results
    base_model = ResNet(PreActBlock, [2, 2, 2, 2], num_classes=10)
    client_loader, mean, std = get_cifar10_loader(num_images=16, batch_size=16, num_workers=2)

    # Run Optuna optimization with Huang
    attack_object = Huang(base_model, client_loader, mean, std)
    attack_object.run_with_optuna()
