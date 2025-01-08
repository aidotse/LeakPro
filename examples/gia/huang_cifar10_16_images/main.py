"""Inverting on a single image."""

from cifar import get_cifar10_loader
import torchvision
import torch
import numpy as np
import random

from leakpro.attacks.gia_attacks.huang import HuangConfig
from leakpro.fl_utils.gia_train import train
from leakpro.fl_utils.model_utils import seed_everything
from leakpro.run import run_huang
from model import ResNet, PreActBlock

if __name__ == "__main__":
    # This attack needs pre activation batch normalization to function properly
    model = ResNet(PreActBlock, [2, 2, 2, 2], num_classes=10)

    seed_everything(1234)
    client_dataloader, data_mean, data_std = get_cifar10_loader(num_images=16, batch_size=16, num_workers=2)
    print(len(client_dataloader))

    # meta train function designed to work with GIA
    train_fn = train
    # baseline config
    configs = HuangConfig()
    result = run_huang(model, client_dataloader, train_fn, data_mean, data_std, configs)
