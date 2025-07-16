import time
import torch
from cifar import get_cifar10_loader

from attack_modifications import gradient_closure, prepare_attack
from leakpro.attacks.gia_attacks.huang import Huang, HuangConfig
from leakpro.attacks.gia_attacks.invertinggradients import InvertingConfig, InvertingGradients
from leakpro.run import run_gia_attack
from leakpro.schemas import OptunaConfig
from leakpro.utils.seed import seed_everything
from model import ResNet, BasicBlock

if __name__ == "__main__":
    # seed_everything(1234)
    model = ResNet(BasicBlock, [5, 5, 5], num_classes=10, base_width=16 * 10)
    gpu_or_cpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(gpu_or_cpu)
    bn_channel_element_counts = []
    def bn_forward_hook(module, input, output):
        # input[0]: shape [B, C, H, W]
        batch_mean = input[0].mean([0, 2, 3])
        batch_var = input[0].var([0, 2, 3], unbiased=False)
        # batch_var = input[0].var([0, 2, 3], unbiased=True)
        rm = module.running_mean.data
        rv = module.running_var.data
        print(batch_var)
        print(rv)
        time.sleep(10000)
    # Register hooks
    hooks = []
    pre_step_running_statistics = []
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.client = True
            pre_step_running_statistics.append((module.running_mean.data.clone(), module.running_var.data.clone()))
            hooks.append(module.register_forward_hook(bn_forward_hook))

    client_dataloader, data_mean, data_std = get_cifar10_loader(num_images=1, batch_size=1, num_workers=2)
    post_step_running_statistics = []

    for x, y in client_dataloader:
        x = x.to(gpu_or_cpu)
        model.train()
        out_train = model(x)
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.client = False
                post_step_running_statistics.append((module.running_mean.data.clone(), module.running_var.data.clone()))
            used_statistics = []

        for (rm_pre, rv_pre), (rm_post, rv_post) in zip(pre_step_running_statistics, post_step_running_statistics):
            used_mean = rm_post # 10 * rm_post - 9 * rm_pre
            used_var = rv_post # 10 * rv_post - 9 * rv_pre
            used_statistics.append((used_mean, used_var))
        # Put the used statistics as running statistics and set model to eval
        stat_idx = 0
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                used_mean, used_var = used_statistics[stat_idx]
                print(f"mean allclose: {torch.allclose(module.running_mean.data, used_mean, atol=1e-8)}")
                # print(module.running_mean.data)
                # print(used_mean)
                n = bn_channel_element_counts[stat_idx]
                correction_factor = 1 # (n - 1) / n
                print(f"var allclose: {torch.allclose(module.running_var.data, used_var * correction_factor, atol=1e-8)}")
                stat_idx += 1
        model.eval()
        out_eval = model(x)
        print(out_train)
        print(out_eval)
        print(torch.allclose(out_eval, out_train, atol=1e-6))

    # trial_data = []
    # for i in range(0,16*5,16):
    #     loader, _, _ = get_cifar10_loader(start_idx=i,
    #                                             num_images=1,
    #                                             batch_size=1,
    #                                             num_workers=2)
    #     trial_data.append(loader)

    # attack_object = InvertingGradients(model, client_dataloader, data_mean, data_std,optuna_trial_data=trial_data)
    # # update functions and reset attack
    # attack_object.gradient_closure = gradient_closure.__get__(attack_object, InvertingGradients)
    # attack_object.prepare_attack = prepare_attack.__get__(attack_object, InvertingGradients)
    # # optuna_config = OptunaConfig(n_trials=100)
    # # attack_object.run_with_optuna(optuna_config)

    # # TEST!!
    # def bn_forward_hook(module, input, output):
    #     batch_mean = input[0].mean([0, 2, 3])
    #     batch_var = input[0].var([0, 2, 3], unbiased=False)

    #     eps = module.eps
    #     invstd = (batch_var + eps).rsqrt()

    #     # Recompute normalized output
    #     normalized = (input[0] - batch_mean[None, :, None, None]) * invstd[None, :, None, None]
    #     if module.affine:
    #         normalized = normalized * module.weight[None, :, None, None] + module.bias[None, :, None, None]

    #     # Compare with actual output
    #     diff = (normalized - output).abs().max()
    #     print(f'Max training difference with actual output: {diff.item()}')
    #     input_tensor = input[0]
    #     B, C, H, W = input_tensor.shape
    #     n = B * H * W  # elements per channel
    #     rm, rv = module.running_mean.data.clone(), module.running_var.data.clone()
    #     used_mean = 10 * rm - 9 * 0.0
    #     used_var = 10 * rv - 9 * 1.0
    #     correction_factor = (n - 1) / n
    #     used_var = used_var*correction_factor

    #     eps = module.eps
    #     invstdi = (used_var + eps).rsqrt()

    #     # Recompute normalized output
    #     normalizedi = (input[0] - used_mean[None, :, None, None]) * invstdi[None, :, None, None]
    #     if module.affine:
    #         normalizedi = normalizedi * module.weight[None, :, None, None] + module.bias[None, :, None, None]

    #     # Compare with actual output
    #     diff = (normalizedi - output).abs().max()
    #     print(f'Max inferred values difference with actual output: {diff.item()}')
    #     # Compare with previous
    #     diff = (normalizedi - normalized).abs().max()
    #     print(f'Max inferred values difference with actual training: {diff.item()}')

    # # Register hooks
    # hooks = []
    # for module in model.modules():
    #     if isinstance(module, torch.nn.BatchNorm2d):
    #         module.client = True
    #         hooks.append(module.register_forward_hook(bn_forward_hook))
    # attack_object.reset_attack(None)

    # # run_gia_attack(attack_object)