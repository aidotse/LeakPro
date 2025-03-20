"""Inverting on a single image."""
import torchvision
from celebA import get_celeba_loader
from torch import nn
from torch.nn import BCEWithLogitsLoss

from leakpro.attacks.gia_attacks.invertinggradients import InvertingConfig
from leakpro.fl_utils.gia_train import train
from leakpro.run import run_inverting

if __name__ == "__main__":
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=40)

    client_dataloader, train_loader, val_loader, data_mean, data_std = get_celeba_loader(
        num_images=1, batch_size=1, num_workers=2, start_idx = 12)

    # Meta train function designed to work with GIA
    train_fn = train

    configs = InvertingConfig()
    configs.at_iterations = 24000
    configs.tv_reg = 1.0e-06
    configs.attack_lr = 0.1
    configs.top10norms = True
    configs.median_pooling = True
    configs.criterion = BCEWithLogitsLoss() #CrossEntropyLoss()
    configs.epochs = 1

    result = run_inverting(model, client_dataloader, train_fn, data_mean, data_std, configs)

