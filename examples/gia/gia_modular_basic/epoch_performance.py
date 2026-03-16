"""Example: Dimitrov FedAvg Multi-Epoch Attack - Strategy Comparison

This example compares different strategies for handling multi-epoch FedAvg attacks
as described in Dimitrov et al. (2022). It demonstrates 5 different approaches:

1. **Dimitrov (prior)**: Full Dimitrov method with order-invariant prior (L_inv) and 
   simulation-based reconstruction (L_sim) with separate optimization variables per epoch.

2. **Dimitrov (no prior)**: Same as (1) but without the epoch order-invariant prior L_inv.

3. **Shared (Geiping et al.)**: Assumes same order of batches across epochs, allowing
   sharing of optimization variables without needing the prior.

4. **FedSGD-Epoch**: Simulates FedAvg with single batch per epoch (Bc=1), so no
   explicit regularization is needed.

5. **FedSGD (Geng et al.)**: Disregards multi-epoch simulation and reconstructs from
   the averaged parameter update like in standard FedSGD.

Reference:
    Dimitrov, D. I., Balunovic, M., Konstantinov, N., & Vechev, M. (2022).
    Data Leakage in Federated Averaging. Transactions on Machine Learning Research.
"""

import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt

from leakpro.attacks.gia_attacks.modular.presets import dimitrov_fedavg_attack
from leakpro.fl_utils.fl_client_simulator import FLClientSimulator
from leakpro.utils.seed import seed_everything
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.training_simulator import (
    TrainingSettings,
)

from cifar import get_cifar10_loader
from model import ResNet, PreActBlock
from visualize import visualize_multiple_attacks


def run_strategy(
    model: nn.Module,
    client_observations,
    device: str,
    epochs: int = 3,
    batch_size: int = 4,
    max_iterations: int = 300,
    client_simulator: FLClientSimulator = None,
) -> dict:
    """Run a single attack strategy and return results.
    
    Args:
        model: Target model
        client_observations: Observations from client
        device: Device to use
        epochs: Number of epochs
        batch_size: Batch size per epoch
        max_iterations: Maximum optimization iterations
        client_simulator: Optional client simulator for metric computation
    Returns:
        Dictionary with reconstruction and metrics
    """
    
    # Create base config
    config = dimitrov_fedavg_attack()
    config.max_iterations = max_iterations
    config.learning_rate = 1.0
    
    # Full Dimitrov: separate images per epoch + order-invariant prior
    config.epoch_handling_strategy = "multi_epoch_separate"
    config.fedavg_lambda_inv = 1e-6  # Enable prior
        


    config.bn_weight=0.00016
    config.bn_strategy="inferred"
    config.bn_momentum=0.1
    config.threat_model_type="model_e"
    # Build attack - all strategies now use client_observations
    # Config flags control which settings are actually used
    attack = config.build(client_observations=client_observations)
    
    reconstruction, attack_config = attack.run_attack(
        target_model=model,
        client_observations=client_observations,
        device=device,
    )
    results = client_simulator.compute_metrics(reconstruction, attack_config)
    
    return results


def main():
    print("=" * 80)
    print("Dimitrov FedAvg Strategy Comparison")
    print("=" * 80)
    print()
    print("Comparing performance under different numbers of epochs:")
    print()
    
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # ==========================================================================
    # Setup: Model and Data
    # ==========================================================================
    print("Setup")
    print("-" * 80)
    
    print("Creating ResNet model...")
    model = ResNet(PreActBlock, [2, 2, 2, 2], num_classes=10, width_factor=1)
    model = model.to(device)
    
    print("Loading CIFAR-10 data...")
    num_images = 6
    batch_size = 3
    
    dataloader, data_mean, data_std = get_cifar10_loader(
        num_images=num_images,
        batch_size=num_images,
        num_workers=0
    )
    data_mean = data_mean.to(device)
    data_std = data_std.to(device)
    
    print(f"✓ Setup complete")
    print(f"  Total images: {num_images}")
    print(f"  Batch size: {batch_size}")
    print()
    
    results = []
    # ==========================================================================
    # Client-side training simulation
    # ==========================================================================
    for epochs in [1,2,4,8]:
        print(f"[CLIENT SIDE] Epochs: {epochs}")
        print("Simulating FedAvg local training...")
    
        training_settings = TrainingSettings(
            epochs=epochs,
            optimizer_type="sgd",
            training_batch_size=batch_size,
            compute_mode="updates",
            model_mode="train",
            shuffle_mode="client",  # Realistic shuffling
        )
        
        client_simulator = FLClientSimulator(
            client_data=dataloader,
            data_mean=data_mean,
            data_std=data_std,
            device=device,
            num_classes=10,
        )
        
        client_observations = client_simulator.train_and_observe(
            server_model=model,
            training_settings=training_settings,
            loss_fn=nn.CrossEntropyLoss(),
            send_labels_to_server=True,
            threat_model="gia_running",
        )
        
        print(f"✓ Client training complete")
        print(f"  Parameter updates sent to server")
        print()
        
        # ==========================================================================
        # Run all strategies
        # ==========================================================================
        print("[ATTACKER SIDE]")
        print("Running all attack strategies...")
        
        max_iterations = 1000 // epochs  # Scale iterations inversely with epochs for fair comparison
    
        print(f"\n{'='*80}")
        print(f"{'='*80}\n")
        strategy_results = run_strategy(
            model=model,
            client_observations=client_observations,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            max_iterations=max_iterations,
            client_simulator=client_simulator,
        )
        results.append(strategy_results)
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print(f"\n{'='*80}")
    print("SUMMARY: Strategy Comparison")
    print(f"{'='*80}\n")
    visualize_multiple_attacks(
        results=results,
        labels=client_observations.labels,
        data_mean=data_mean,
        data_std=data_std,
        file="epoch_number_comparison.png",
    )

if __name__ == "__main__":
    main()
