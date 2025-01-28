"""Inverting on a single image."""

import os

import torch
from cifar import get_cifar10_dataset
from model import ResNet
from pre_train import pre_train
from torchvision.models.resnet import BasicBlock

from leakpro.fl_utils.gia_train import train
from leakpro.run import run_inverting_audit
from leakpro.utils.logger import logger

if __name__ == "__main__":
    model = ResNet(BasicBlock, [5, 5, 5], num_classes=10, base_width=16 * 10)
    pre_train_loader, client_dataset, data_mean, data_std = get_cifar10_dataset()

    # pre training the model is important since the attacks work better on later stages of model training.
    # check out the transforms in the dataset, pre-training with those transformations make the attacks a lot stronger.
    pre_train_epochs = 10
    model_path = "model_epochs_" + str(pre_train_epochs) + ".pth"
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
    # run audit with multiple client partitions, different epochs and total_variation scale.
    result = run_inverting_audit(model, client_dataset, train_fn, data_mean, data_std)
