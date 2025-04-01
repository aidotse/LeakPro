"""Inverting on a single image."""

from cifar import get_cifar10_loader
from leakpro.fl_utils.data_utils import GiaImageDetectionExtension
from model import ResNet
from torchvision.models.resnet import BasicBlock

from leakpro.attacks.gia_attacks.invertinggradients import InvertingConfig, InvertingGradients
from leakpro.fl_utils.gia_train import train
from leakpro.run import run_gia_attack

if __name__ == "__main__":
    model = ResNet(BasicBlock, [5, 5, 5], num_classes=10, base_width=16 * 10)
    client_dataloader, data_mean, data_std = get_cifar10_loader(num_images=1, batch_size=1, num_workers=2)

    # meta train function designed to work with GIA
    train_fn = train
    configs = InvertingConfig(data_extension=GiaImageDetectionExtension())
    attack_object = InvertingGradients(model, client_dataloader, data_mean, data_std, configs=configs)

    result = run_gia_attack(attack_object)
