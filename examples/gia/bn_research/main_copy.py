import torch
from cifar import get_cifar10_loader

from attack_modifications import gradient_closure, prepare_attack, prepare_attack2
from leakpro.attacks.gia_attacks.huang import Huang, HuangConfig
from leakpro.attacks.gia_attacks.invertinggradients import InvertingConfig, InvertingGradients
from leakpro.run import run_gia_attack
from leakpro.schemas import OptunaConfig
from leakpro.utils.seed import seed_everything
from model import ResNet, BasicBlock

if __name__ == "__main__":
    # seed_everything(1234)
    model = ResNet(BasicBlock, [5, 5, 5], num_classes=10, base_width=16 * 10)

    client_dataloader, data_mean, data_std = get_cifar10_loader(num_images=1, batch_size=1, num_workers=2)
    trial_data = []
    for i in range(0,16*5,16):
        loader, _, _ = get_cifar10_loader(start_idx=i,
                                                num_images=1,
                                                batch_size=1,
                                                num_workers=2)
        trial_data.append(loader)

    attack_object = InvertingGradients(model, client_dataloader, data_mean, data_std,optuna_trial_data=trial_data)
    # update functions and reset attack
    attack_object.gradient_closure = gradient_closure.__get__(attack_object, InvertingGradients)
    attack_object.prepare_attack = prepare_attack.__get__(attack_object, InvertingGradients)
    # optuna_config = OptunaConfig(n_trials=100)
    # attack_object.run_with_optuna(optuna_config)

    # TEST!!
    bn_used_stats_client = []
    bn_used_stats_test = []
    def bn_forward_hook(module, input, output):
        if module.training:
            batch_mean = input[0].mean([0, 2, 3])
            batch_var = input[0].var([0, 2, 3], unbiased=False)
            bn_used_stats_client.append((batch_mean.detach().cpu(), batch_var.detach().cpu()))
        elif module.client:
            # During training, BN uses batch stats
            bn_used_stats_client.append((module.running_mean.detach().cpu(), module.running_var.detach().cpu()))
        else:
            # During eval, BN uses running stats
            bn_used_stats_test.append((module.running_mean.detach().cpu(), module.running_var.detach().cpu()))

    # Register hooks
    hooks = []
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.client = True
            hooks.append(module.register_forward_hook(bn_forward_hook))

    # Run prepare_attack
    attack_object.reset_attack(None)

    # Remove hooks
    for h in hooks:
        h.remove()
    # Compare the recorded stats
    for idx, ((mean_client, var_client), (mean_test, var_test)) in enumerate(zip(bn_used_stats_client, bn_used_stats_test)):
        mean_diff = (mean_client - mean_test).abs().max()
        var_diff = (var_client - var_test).abs().max()
        print(f"Layer {idx}: mean diff = {mean_diff}, var diff = {var_diff}")
    run_gia_attack(attack_object)