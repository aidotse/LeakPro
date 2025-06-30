"""Inverting on a batch of 16 images."""
import os

import torch
from cifar100 import get_cifar100_loader
from leakpro.schemas import OptunaConfig
from model import ResNet
from pre_train import pre_train
from torchvision.models.resnet import BasicBlock

from leakpro.attacks.gia_attacks.invertinggradients import InvertingConfig, InvertingGradients
from leakpro.utils.logger import logger

if __name__ == "__main__":
    model = ResNet(BasicBlock, [5, 5, 5], num_classes=100, base_width=16 * 10)
    # Note the transform.cropping and transform.horizontalflip
    # Without them and pre-training the attacks wont work on larger batches.
    client_trainloader, pre_train_loader, data_mean, data_std = get_cifar100_loader(start_idx=None,
                                                                                    num_images=16,
                                                                                    client_batch_size=16,
                                                                                    pre_train_batch_size=64,
                                                                                    num_workers=2)
    trial_data = []
    for i in range(0,16*5,16):
        loader, _, _, _ = get_cifar100_loader(start_idx=i,
                                                num_images=16,
                                                client_batch_size=16,
                                                pre_train_batch_size=64,
                                                num_workers=2)
        trial_data.append(loader)
    pre_train_epochs = 10
    model_path = "model_epochs_" + str(pre_train_epochs) + ".pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        logger.info(f"Model loaded from {model_path}")
    else:
        logger.info("No saved model found. Training from scratch...")
        pre_train(model, pre_train_loader, epochs=pre_train_epochs)
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model trained and saved to {model_path}")
    # lower total variation scale for larger batch sizes.
    config = InvertingConfig()
    config.at_iterations = 12000
    attack_object = InvertingGradients(model, client_trainloader, data_mean, data_std, configs=config, optuna_trial_data=trial_data)
    optuna_config = OptunaConfig()
    optuna_config.n_trials = 100
    attack_object.run_with_optuna()
    
