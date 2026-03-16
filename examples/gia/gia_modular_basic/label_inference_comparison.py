"""Example: Comparing Label Inference Strategies for Multi-Epoch FedAvg

This example compares different label inference strategies:
1. Oracle - Ground truth labels (upper bound)
2. iDLG - Standard gradient-based label inference
3. Dimitrov - Analytical label inference using interpolated statistics

The comparison shows which method works best for multi-epoch federated averaging
scenarios where standard iDLG may struggle.

Reference:
    Dimitrov, D. I., Balunovic, M., Konstantinov, N., & Vechev, M. (2022).
    Data Leakage in Federated Averaging. TMLR.
"""

import torch
import torch.nn as nn
from pathlib import Path

from leakpro.attacks.gia_attacks.modular.components.label_inference import (
    IDLGLabelInference,
    GengLabelInference,
    DimitrovLabelInference,
    OracleLabels,
)
from leakpro.fl_utils.fl_client_simulator import FLClientSimulator
from leakpro.utils.seed import seed_everything
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.training_simulator import (
    TrainingSettings,
)

from cifar import get_cifar10_loader
from model import ResNet, PreActBlock


def test_label_inference_strategy(
    strategy_name: str,
    label_strategy,
    model: nn.Module,
    client_observations,
    true_labels: torch.Tensor,
    device: str,
    learning_rate: float = 0.1,
) -> dict:
    """Test a single label inference strategy.
    
    Args:
        strategy_name: Name of the strategy
        label_strategy: Label inference strategy instance
        model: Target model
        client_observations: Client gradient observations
        true_labels: Ground truth labels
        device: Computation device
        
    Returns:
        Dictionary with results
    """
    print(f"\n{'='*60}")
    print(f"Testing: {strategy_name}")
    print(f"{'='*60}")
    
    # Convert gradients to list if it's a dict (from training simulator)
    gradients = client_observations.gradients
    if isinstance(gradients, dict):
        gradients = list(gradients.values())
    
    # Determine if these are updates or gradients
    is_update = (client_observations.training_settings and 
                 client_observations.training_settings.compute_mode == "updates")

    # Infer labels
    result = label_strategy.infer_labels(
        gradients=gradients,
        model=model,
        num_samples=len(true_labels),
        true_labels=true_labels if isinstance(label_strategy, OracleLabels) else None,
        is_update=is_update,
    )
    
    inferred_labels = result.labels.cpu()
    
    # Calculate accuracy
    correct = 0
    total = len(true_labels)
    
    # For unordered batch, check if inferred label set matches true label set
    true_counts = torch.bincount(true_labels.cpu(), minlength=10)
    inferred_counts = torch.bincount(inferred_labels, minlength=10)
    
    # Count correct label occurrences
    correct_per_class = torch.min(true_counts, inferred_counts)
    correct = correct_per_class.sum().item()
    
    accuracy = correct / total * 100
    
    # Calculate per-label accuracy
    label_errors = (true_counts - inferred_counts).abs()
    
    print(f"✓ Label Inference Complete")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"  Correct labels: {correct}/{total}")
    print(f"  True label counts: {true_counts.tolist()}")
    print(f"  Inferred counts:   {inferred_counts.tolist()}")
    print(f"  Absolute errors:   {label_errors.tolist()}")
    
    return {
        "strategy": strategy_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "true_counts": true_counts,
        "inferred_counts": inferred_counts,
        "errors": label_errors,
        "success": True,
    }

def main():
    print("=" * 80)
    print("Label Inference Strategy Comparison for Multi-Epoch FedAvg")
    print("=" * 80)
    print()
    
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # ==========================================================================
    # Setup
    # ==========================================================================
    print("Setup")
    print("-" * 80)
    
    # Model
    print("Creating ResNet model...")
    model = ResNet(PreActBlock, [2, 2, 2, 2], num_classes=10, width_factor=1)
    model = model.to(device)
    model.eval()
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Data
    num_images = 30
    batch_size = 4
    epochs = 20
    learning_rate = 0.01
    
    print(f"Loading CIFAR-10 data: {num_images} images...")
    client_dataloader, data_mean, data_std = get_cifar10_loader(
        num_images=num_images,
        batch_size=num_images,
        num_workers=2,
        # excluded_classes=[0,1,2,3,4],
    )
    
    # Get a batch
    images, labels = next(iter(client_dataloader))
    images = images.to(device)
    labels = labels.to(device)
    print(f"✓ Data loaded - Labels: {labels.tolist()}")
    
    # ==========================================================================
    # Simulate Client Training (FedAvg)
    # ==========================================================================
    print(f"\nSimulating Client Training")
    print("-" * 80)
    
    client_simulator = FLClientSimulator(
        client_data=client_dataloader,
        data_mean=data_mean,
        data_std=data_std,
        device=device,
        num_classes=10,
    )
    settings=TrainingSettings(
            epochs=epochs,
            training_batch_size=batch_size,
            compute_mode="updates",
            optimizer_type="sgd",
            model_mode="eval",
            shuffle_mode="client",
        )
    client_observations = client_simulator.train_and_observe(
        server_model=model,
        training_settings=settings,
        loss_fn=nn.CrossEntropyLoss(),
        send_labels_to_server=True,
    )
    print(f"✓ Client simulation complete - {epochs} epochs, LR={learning_rate}")
    
    # ==========================================================================
    # Test Label Inference Strategies
    # ==========================================================================
    print(f"\nTesting Label Inference Strategies")
    print("-" * 80)
    
    strategies = [
        ("Oracle (Ground Truth)", OracleLabels()),
        ("iDLG", IDLGLabelInference()),
        ("Geng", GengLabelInference(num_dummy_samples=10)),
        ("Dimitrov (start)", DimitrovLabelInference(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_dummy_samples=10,
            strategy="start",
        )),
        ("Dimitrov (end)", DimitrovLabelInference(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_dummy_samples=10,
            strategy="end",
        )),
        ("Dimitrov (avg)", DimitrovLabelInference(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_dummy_samples=10,
            strategy="avg",
        )),
    ]
    
    results = []
    for strategy_name, strategy_instance in strategies:
        result = test_label_inference_strategy(
            strategy_name=strategy_name,
            label_strategy=strategy_instance,
            model=model,
            client_observations=client_observations,
            true_labels=labels,
            device=device,
            learning_rate=learning_rate,
        )
        results.append(result)
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    
    print("\nLabel Inference Accuracy:")
    for result in results:
        if result.get("success", False):
            print(f"  {result['strategy']:25s}: {result['accuracy']:5.1f}%")
        else:
            print(f"  {result['strategy']:25s}: FAILED - {result.get('error', 'Unknown error')}")
    



if __name__ == "__main__":
    main()
