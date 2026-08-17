#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Quick test - just run iDLG which is faster than DLG."""

import torch
import torch.nn as nn
from collections import OrderedDict
from pathlib import Path
import matplotlib.pyplot as plt

from leakpro.attacks.gia_attacks.modular.presets import dlg_attack, huang_attack, idlg_attack, inverting_gradients_attack, gia_running_attack, gia_estimate_attack
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.step_strategies import StandardStepStrategy
from leakpro.fl_utils.fl_client_simulator import FLClientSimulator
from leakpro.utils.seed import seed_everything
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.training_simulator import (
    TrainingSettings,
)

from cifar import get_cifar10_loader
from model import ResNet, PreActBlock
from visualize import visualize_multiple_attacks

def pretrain_and_extract_state(model, dataloader, optimizer_type, lr, steps, loss_fn, device):
    """Quickly train `model` in place, then extract the optimizer state as a warm-start dict.

    Simulates a client whose optimizer has already run for several FL rounds: we train the model
    with a real torch optimizer for `steps` steps, then pull its state into an optimizer_state dict
    keyed by parameter name ({"m", "v", "t"} for adam, {"m", "t"} for momentum). The model is left
    with the corresponding already-trained weights, so weights and optimizer state stay consistent.

    Returns None for SGD (stateless).
    """
    model.to(device).train()
    if optimizer_type == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, foreach=False)
    elif optimizer_type == "momentum":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, foreach=False)
    elif optimizer_type == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, foreach=False)
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}")

    done = 0
    while done < steps:
        for x, y in dataloader:
            if done >= steps:
                break
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
            done += 1
    print(f"Pretrained {done} steps to build the client's optimizer state")

    if optimizer_type == "sgd":
        return None  # stateless

    # torch keys optimizer state by parameter tensor; remap to parameter name.
    name_by_param = {p: n for n, p in model.named_parameters()}
    m, v, t = OrderedDict(), OrderedDict(), 0
    for p, st in opt.state.items():
        name = name_by_param[p]
        if optimizer_type == "adam":
            m[name] = st["exp_avg"].detach().clone()
            v[name] = st["exp_avg_sq"].detach().clone()
            step = st["step"]
            t = int(step.item()) if torch.is_tensor(step) else int(step)
        else:  # momentum
            m[name] = st["momentum_buffer"].detach().clone()
            t = done
    return {"m": m, "v": v, "t": t} if optimizer_type == "adam" else {"m": m, "t": t}


def main():
    print("="*60)
    print("Quick Geiping Test on CIFAR-10")
    print("="*60)
    
    seed_everything(1234)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Model
    print("Creating model...")
    model = ResNet(PreActBlock, [2, 2, 2, 2], num_classes=10, width_factor=2)
    
    # Data
    print("Loading CIFAR-10...")
    dataloader, data_mean, data_std = get_cifar10_loader(
        num_images=2,
        batch_size=2,
        num_workers=0
    )

    proxy_dataloader, _, _ = get_cifar10_loader(
        start_idx=32,
        num_images=4,
        batch_size=4,
        num_workers=0
    )
    
    data_mean = data_mean.to(device)
    data_std = data_std.to(device)

    optimizer_type = "adam"
    learning_rate = 0.001
    loss_fn = nn.CrossEntropyLoss()

    # Scenario: the client's optimizer has already run for several rounds. Pretrain the model and
    # extract its optimizer state (m/v/t) to warm-start the attack's training simulation. This state
    # flows client -> observations -> attacker, so both sides use the same start.
    # (Pass optimizer_state=None instead for the default zero-init behavior.)
    #
    # IMPORTANT: pretrain on proxy data, NOT the client's target images. Pretraining on the targets
    # would drive the loss to ~0, making the client update Δθ ≈ 0 (uninformative) — the gradient-
    # matching attack then converges to a degenerate solution that does not resemble the originals.
    optimizer_state = pretrain_and_extract_state(
        model, proxy_dataloader, optimizer_type, lr=learning_rate, steps=20, loss_fn=loss_fn, device=device,
    )

    training_settings = TrainingSettings(
        epochs=1,
        optimizer_type=optimizer_type,
        training_batch_size=2,
        compute_mode="updates",
        model_mode="train",
        shuffle_mode="client",
        optimizer_state=optimizer_state,
        learning_rate=learning_rate,
    )

    client_simulator = FLClientSimulator(
        client_data=dataloader,
        data_mean=data_mean,
        data_std=data_std,
        device=device,
        num_classes=10,
    )

    client_observation = client_simulator.train_and_observe(
        server_model=model,
        training_settings=training_settings,
        loss_fn=loss_fn,
        send_labels_to_server=True,
        threat_model="gia_running",
    )

    input_shape = client_simulator.original_inputs.shape


    attacks = []
    results = []

    #geiping = inverting_gradients_attack()
    #attacks.append(("Geiping", geiping))

    #huang = huang_attack()
    #attacks.append(("Huang", huang))

    gia_running = gia_running_attack()
    attacks.append(("GIA Running", gia_running))

    #gia_base_attack = gia_estimate_attack()
    #attacks.append(("GIA Base", gia_base_attack))

    for attack_name, attack in attacks:
        print(f"\n{'='*20} Running {attack_name} Attack {'='*20}\n")
        attack.max_iterations = 1000
        attack.log_interval = 10
        attack = attack.build(client_observations=client_observation)
        reconstruction, attack_config = attack.run_attack(
            target_model=model,
            input_shape=input_shape,
            device=device,
            client_observations=client_observation,
            proxy_dataloader=proxy_dataloader
        )


        results.append(client_simulator.compute_metrics(reconstruction, attack_config))

    
    visualize_multiple_attacks(results, client_simulator.original_labels, data_mean, data_std, file="main_results.png")
    

if __name__ == "__main__":
    main()