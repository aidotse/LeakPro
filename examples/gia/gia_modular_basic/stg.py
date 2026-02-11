"""Quick test - just run iDLG which is faster than DLG."""

import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt

from leakpro.attacks.gia_attacks.modular.presets import dlg_attack, huang_attack, idlg_attack, inverting_gradients_attack, gia_running_attack, gia_estimate_attack, see_through_gradients_attack
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.step_strategies import StandardStepStrategy
from leakpro.fl_utils.fl_client_simulator import FLClientSimulator
from leakpro.utils.seed import seed_everything
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.training_simulator import (
    MultiEpochTrainingSimulation,
)

from cifar import get_cifar10_loader
from model import ResNet, PreActBlock
from visualize import visualize_multiple_attacks

def main():
    print("="*60)
    print("Quick Geiping Test on CIFAR-10")
    print("="*60)
    
    seed_everything(1234)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Model
    print("Creating model...")
    model = ResNet(PreActBlock, [2, 2, 2, 2], num_classes=10, width_factor=1)
    
    # Data
    print("Loading CIFAR-10...")
    dataloader, data_mean, data_std = get_cifar10_loader(
        num_images=4,
        batch_size=4,
        num_workers=0
    )

    data_mean = data_mean.to(device)
    data_std = data_std.to(device)
    
    training_simulator = MultiEpochTrainingSimulation(epochs=1, 
                                                      compute_mode="updates", 
                                                      model_mode="train",
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
        training_simulator=training_simulator,
        loss_fn=nn.CrossEntropyLoss(),
        send_labels_to_server=True, 
        threat_model="gia_running",
    )

    input_shape = client_simulator.original_inputs.shape


    attacks = []
    results = []

    stg1 = see_through_gradients_attack()
    stg1.num_seeds_per_image = 1
    stg1.max_iterations = 10000
    attacks.append(("See Through Gradients", stg1))

    stg4 = see_through_gradients_attack()
    stg4.num_seeds_per_image = 4
    stg4.max_iterations = 2500
    attacks.append(("STG 4 Seeds", stg4))


    for attack_name, attack in attacks:
        print(f"\n{'='*20} Running {attack_name} Attack {'='*20}\n")
        # attack.bn_weight = 0.0
        attack.gradient_loss_type = "cosine"
        attack.gradient_noise_std = 0.0
        attack.bn_strategy = "running"
        attack.tv_weight = 0.052
        attack.bn_weight = 0.00016
        attack.learning_rate = 1.0
        attack = attack.build(training_simulator=training_simulator)
        reconstruction, attack_config = attack.run_attack(
            target_model=model,
            input_shape=input_shape,
            device=device,
            client_observations=client_observation,
        )


        results.append(client_simulator.compute_metrics(reconstruction, attack_config))

    
    visualize_multiple_attacks(results, client_simulator.original_labels, data_mean, data_std, file="stg_1_vs_4_seeds.png")
    

if __name__ == "__main__":
    main()