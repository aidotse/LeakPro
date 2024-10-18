"""Inverting on a batch of 16 images."""
import os
from dataclasses import asdict

import torch
from cifar100 import get_cifar100_loader
from model import ResNet
from pre_train import pre_train
from torchvision.models.resnet import BasicBlock

from leakpro.attacks.gia_attacks.invertinggradients import InvertingConfig
from leakpro.fl_utils.gia_train import train
from leakpro.run import run_inverting
from leakpro.utils.logger import logger

if __name__ == "__main__":
    model = ResNet(BasicBlock, [5, 5, 5], num_classes=10, base_width=16 * 10)
    # Note the transform.cropping and transform.horizontalflip
    # Without them and pre-training the the attacks wont work on larger batches.
    client_trainloader, pre_train_loader, data_mean, data_std = get_cifar100_loader(num_images=16, batch_size=1, num_workers=2)
    pre_train_epochs = 10
    model_path = "model_epochs_" + pre_train_epochs + ".pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        logger.info(f"Model loaded from {model_path}")
    else:
        logger.info("No saved model found. Training from scratch...")
        pre_train(model, pre_train_loader, epochs=10)
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model trained and saved to {model_path}")
    # meta train function designed to work with GIA
    train_fn = train
    # baseline config
    configs = asdict(InvertingConfig())
    result = run_inverting(model, client_trainloader, train_fn, data_mean, data_std, configs)
