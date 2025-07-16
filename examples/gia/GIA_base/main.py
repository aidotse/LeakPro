"""Inverting on a single image."""

from cifar import get_cifar10_loader

from leakpro.attacks.gia_attacks.gia_estimate import GIABase
from leakpro.schemas import OptunaConfig
from leakpro.utils.seed import seed_everything
from model import ResNet, PreActBlock

if __name__ == "__main__":
    seed_everything(1234)
    # This attack needs pre activation batch normalization to function properly
    model = ResNet(PreActBlock, [2, 2, 2, 2], num_classes=10, width_factor=2)
    client_dataloader, data_mean, data_std = get_cifar10_loader(start_idx=0, num_images=16, batch_size=16, num_workers=2)
    proxy_loader, data_mean, data_std = get_cifar10_loader(start_idx=20, num_images=16, batch_size=16, num_workers=2)
    trial_data = []
    for i in range(0,16*10,16):
        # only use two loaders for client here
        client_idx = 0 if i > 16*4 else 16
        loader, _, _ = get_cifar10_loader(start_idx=i,
                                                num_images=16,
                                                batch_size=16,
                                                num_workers=2)
        proxy, _, _ = get_cifar10_loader(start_idx=i+32,
                                                num_images=16,
                                                batch_size=16,
                                                num_workers=2)
        trial_data.append((loader, proxy))

    # meta train function designed to work with GIA
    # baseline config
    attack_object = GIABase(model, client_dataloader, data_mean, data_std,proxy_loader=proxy_loader,optuna_trial_data=trial_data)
    optuna_config = OptunaConfig(n_trials=100)
    attack_object.run_with_optuna(optuna_config)
