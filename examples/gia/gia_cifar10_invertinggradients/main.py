"""Inverting on a single image."""
from dataclasses import asdict

from cifar import get_cifar10_loader
from model import ResNet
from torchvision.models.resnet import BasicBlock

from leakpro.attacks.gia_attacks.invertinggradients import InvertingConfig
from leakpro.fl_utils.gia_train import train
from leakpro.run import run_inverting

if __name__ == "__main__":
    model = ResNet(BasicBlock, [5, 5, 5], num_classes=10, base_width=16 * 10)
    client_dataloader, data_mean, data_std = get_cifar10_loader(num_images=1, batch_size=1, num_workers=2)

    # meta train function designed to work with GIA
    train_fn = train
    # baseline config
    configs = asdict(InvertingConfig())
    result = run_inverting(model, client_dataloader, train_fn, data_mean, data_std, configs)
