import os
import torch
from cifar import get_cifar10_loader

from attack_modifications import gradient_closure, prepare_attack
from cifar100 import get_cifar100_loader
from pre_train import pre_train
from leakpro.attacks.gia_attacks.huang import Huang, HuangConfig
from leakpro.attacks.gia_attacks.invertinggradients import InvertingConfig, InvertingGradients
from leakpro.fl_utils.gia_train import train_nostep
from leakpro.run import run_gia_attack
from leakpro.schemas import OptunaConfig
from leakpro.utils.seed import seed_everything
from model import ResNet, BasicBlock
import torch
import torch.nn.functional as F

if __name__ == "__main__":
    seed_everything(1234)
    model = ResNet(BasicBlock, [5, 5, 5], num_classes=100, base_width=16 * 10)

    # client_dataloader, data_mean, data_std = get_cifar10_loader(num_images=16, batch_size=16, num_workers=2)
    client_trainloader, pre_train_loader, data_mean, data_std = get_cifar100_loader(start_idx=None,
                                                                                    num_images=16,
                                                                                    client_batch_size=16,
                                                                                    pre_train_batch_size=64,
                                                                                    num_workers=2)
    pre_train_epochs = 10
    model_path = "model_epochs_" + str(pre_train_epochs) + ".pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
    else:
        print("No saved model found. Training from scratch...")
        pre_train(model, pre_train_loader, epochs=pre_train_epochs)
        torch.save(model.state_dict(), model_path)
        print(f"Model trained and saved to {model_path}")
    # trial_data = []
    # for i in range(0,16*5,16):
    #     loader, _, _ = get_cifar10_loader(start_idx=i,
    #                                             num_images=1,
    #                                             batch_size=1,
    #                                             num_workers=2)
    #     trial_data.append(loader)
    config = InvertingConfig()
    config.at_iterations = 12000
    attack_object = InvertingGradients(model, client_trainloader, data_mean, data_std, train_fn=train2)#,optuna_trial_data=trial_data)
    # update functions and reset attack
    attack_object.gradient_closure = gradient_closure.__get__(attack_object, InvertingGradients)
    attack_object.prepare_attack = prepare_attack.__get__(attack_object, InvertingGradients)
    run_gia_attack(attack_object)