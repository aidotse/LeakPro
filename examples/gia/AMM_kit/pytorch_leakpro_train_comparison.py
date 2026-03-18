import copy
import random
from typing import Literal

import numpy as np
import torch

from leakpro.attacks.gia_attacks.invertinggradients import InvertingConfig
from leakpro.fl_utils import gia_optimizers
from leakpro.fl_utils.gia_train import train as train_leakpro

from train_scaleout import train as train_scaleout_fn
from cifar import get_cifar10_loader
from model import CNN, ResNet
from torchvision.models.resnet import BasicBlock


OptimizerName = Literal["adam", "sgd"]


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensor_list_l2norm(tensors: list[torch.Tensor]) -> float:
    return sum(torch.norm(t).item() for t in tensors)


def compare_tensor_lists(tensors1: list[torch.Tensor], tensors2: list[torch.Tensor]) -> float:
    return sum(torch.norm(t1 - t2).item() for t1, t2 in zip(tensors1, tensors2))


def pairwise_differences(runs: list[list[torch.Tensor]]) -> list[float]:
    return [
        compare_tensor_lists(runs[i], runs[j])
        for i in range(len(runs))
        for j in range(i + 1, len(runs))
    ]


def format_differences(values: list[float]) -> str:
    return ", ".join(f"{v:.2e}" for v in values)


def build_leakpro_optimizer(
    name: OptimizerName,
    learning_rate: float,
    differentiable: bool = True,
):
    if name == "adam":
        return gia_optimizers.MetaAdam(
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
            foreach=False,
            fused=False,
            decoupled_weight_decay=False,
            differentiable=differentiable,
        )

    if name == "sgd":
        return gia_optimizers.MetaSGD(
            lr=learning_rate,
            foreach=False,
        )

    raise ValueError(f"Unsupported optimizer: {name}")


def build_torch_optimizer(
    name: OptimizerName,
    model: torch.nn.Module,
    learning_rate: float,
):
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
            foreach=False,
            fused=False,
        )

    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            foreach=False,
        )

    raise ValueError(f"Unsupported optimizer: {name}")


def create_model() -> torch.nn.Module:
    # return ResNet(BasicBlock, [5, 5, 5], num_classes=10, base_width=16)
    return CNN(num_classes=10)


def run_leakpro_train(
    model: torch.nn.Module,
    epochs: int,
    learning_rate: float,
    dataloader,
    optimizer_name: OptimizerName,
    device: torch.device,
    differentiable: bool,
) -> list[torch.Tensor]:
    configs = InvertingConfig()
    configs.epochs = epochs
    configs.optimizer = build_leakpro_optimizer(
        optimizer_name,
        learning_rate,
        differentiable=differentiable,
    )

    model_leakpro = copy.deepcopy(model)

    _, updated_params = train_leakpro(
        model_leakpro,
        dataloader,
        configs.optimizer,
        configs.criterion,
        configs.epochs,
        device=device,
    )

    return [param.detach().cpu().clone() for param in updated_params]


def run_scaleout_train(
    model: torch.nn.Module,
    epochs: int,
    learning_rate: float,
    dataloader,
    optimizer_name: OptimizerName,
    device: torch.device,
) -> list[torch.Tensor]:
    model_scaleout = copy.deepcopy(model)
    optimizer = build_torch_optimizer(optimizer_name, model_scaleout, learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    trained_model = train_scaleout_fn(
        model=model_scaleout,
        data=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=epochs,
        device=device,
    )

    return [param.detach().cpu().clone() for param in trained_model.parameters()]


def test_leakpro_train_comparison() -> None:
    epochs = 10
    learning_rate = 0.001
    optimizer_name: OptimizerName = "adam"
    leakpro_differentiable = False
    num_runs = 5

    dataloader, data_mean, data_std = get_cifar10_loader(
        num_images=25,
        batch_size=5,
        num_workers=0,
    )

    device = torch.device("cpu")
    model = create_model()
    model.to(device)
    model.eval()

    print(f"Optimizer: {optimizer_name}")
    print(f"LeakPro differentiable: {leakpro_differentiable}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}")
    print()

    leakpro_runs = [
        run_leakpro_train(
            model,
            epochs,
            learning_rate,
            dataloader,
            optimizer_name,
            device,
            differentiable=leakpro_differentiable,
        )
        for _ in range(num_runs)
    ]

    scaleout_runs = [
        run_scaleout_train(
            model,
            epochs,
            learning_rate,
            dataloader,
            optimizer_name,
            device,
        )
        for _ in range(num_runs)
    ]

    orig_norm = tensor_list_l2norm([p.detach().cpu() for p in model.parameters()])
    leakpro_mean_norm = np.mean([tensor_list_l2norm(run) for run in leakpro_runs])
    scaleout_mean_norm = np.mean([tensor_list_l2norm(run) for run in scaleout_runs])

    cross_diffs = [
        compare_tensor_lists(leakpro_run, scaleout_run)
        for leakpro_run, scaleout_run in zip(leakpro_runs, scaleout_runs)
    ]
    scaleout_diffs = pairwise_differences(scaleout_runs)
    leakpro_diffs = pairwise_differences(leakpro_runs)

    print(f"L2 norm orig gradients: {orig_norm}")
    print(f"Mean L2 norm from LeakPro training: {leakpro_mean_norm}")
    print(f"Mean L2 norm from PyTorch training: {scaleout_mean_norm}")
    print(f"LeakPro vs PyTorch diffs: {format_differences(cross_diffs)}")
    print(f"PyTorch vs PyTorch diffs: {format_differences(scaleout_diffs)}")
    print(f"LeakPro vs LeakPro diffs: {format_differences(leakpro_diffs)}")


if __name__ == "__main__":
    set_seed(0)
    test_leakpro_train_comparison()