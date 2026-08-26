#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Quick test - just run iDLG which is faster than DLG."""

import torch
import torch.nn as nn
from collections import OrderedDict
from datetime import datetime
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


def plot_loss_history(loss_history, file, reference=None, start_iteration=0):
    """Plot the total loss and each component loss as a function of GIA iteration.

    One subplot per loss term, each y-axis autoscaled to its own (windowed) data so the progress
    is visible even when terms have very different magnitudes.

    Args:
        start_iteration: Skip iterations before this, so the large early transient does not
            compress the y-axis and hide later progress (e.g. start_iteration=200).
        reference: Optional {name: value} dict (e.g. from compute_reference_loss on the ground
            truth); drawn as a horizontal dashed line per subplot.
    """
    history = [entry for entry in loss_history if entry["iteration"] >= start_iteration]
    if not history:
        print(f"No loss history at/after iteration {start_iteration}; skipping {file}.")
        return
    iterations = [entry["iteration"] for entry in history]
    keys = [k for k in history[0] if k != "iteration"]
    colors = plt.cm.tab10.colors
    n = len(keys)
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(9, 2.4 * n))
    if n == 1:
        axes = [axes]
    for ax, k, color in zip(axes, keys, colors):
        values = [entry[k] for entry in history]
        ax.plot(iterations, values, color=color, linewidth=2.0 if k == "total" else 1.5)
        ax.set_title(k, loc="left", fontsize=9)
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        if reference is not None and k in reference:
            ax.axhline(reference[k], color=color, linestyle="--", linewidth=1.0, alpha=0.7,
                       label=f"reference (original) = {reference[k]:.4f}")
            ax.legend(fontsize=8, loc="best")
        # Autoscale y to the windowed data (set last so the reference line can't blow up the view)
        lo, hi = min(values), max(values)
        margin = 0.05 * (hi - lo) if hi > lo else max(abs(hi) * 0.05, 1e-8)
        ax.set_ylim(lo - margin, hi + margin)
    axes[-1].set_xlabel("Iteration")
    title = "GIA loss per iteration" + (f" (from iteration {start_iteration})" if start_iteration else "")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    plt.savefig(file, dpi=120)
    plt.close()
    print(f"Saved loss curve to: {file}")


def plot_metric_history(iterations, psnr, ssim, file):
    """Plot PSNR and SSIM (one point per loss improvement) on twin y-axes."""
    if not iterations:
        print("No metric history to plot.")
        return
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(iterations, psnr, color="tab:blue", marker="o", label="PSNR (dB)")
    ax1.set_xlabel("Iteration (at each loss improvement)")
    ax1.set_ylabel("PSNR (dB)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(iterations, ssim, color="tab:red", marker="s", label="SSIM")
    ax2.set_ylabel("SSIM", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    plt.title("Reconstruction quality per improvement")
    fig.tight_layout()
    plt.savefig(file, dpi=120)
    plt.close()
    print(f"Saved metric curve to: {file}")


def save_reconstruction_series(snapshots, client_simulator, device, file):
    """Stack the per-improvement reconstruction snapshots and score each against ground truth.

    Saves a single file with:
        reconstructions: [S, N, C, H, W]   (S = number of loss improvements)
        iterations:      [S]               iteration each improvement occurred at
        losses:          [S]               total loss at each improvement
        psnr:            [S]               PSNR (dB) of each snapshot vs the originals
        ssim:            [S]               SSIM of each snapshot vs the originals
    """
    if not snapshots:
        print("No reconstruction snapshots to save.")
        return None
    reconstructions = torch.stack([s["reconstruction"] for s in snapshots])  # [S, N, C, H, W]
    iterations = torch.tensor([s["iteration"] for s in snapshots])
    losses = torch.tensor([s["loss"] for s in snapshots])
    psnr, ssim = [], []
    for s in snapshots:
        psnr_score, ssim_score, _ = client_simulator.compute_scores(s["reconstruction"].to(device))
        psnr.append(psnr_score)
        ssim.append(ssim_score)
    series = {
        "reconstructions": reconstructions,
        "iterations": iterations,
        "losses": losses,
        "psnr": torch.tensor(psnr),
        "ssim": torch.tensor(ssim),
        # ground truth + normalization constants, so a viewer can show originals next to the
        # reconstructions in the same (denormalized) color space
        "originals": client_simulator.original_inputs.detach().cpu(),
        "data_mean": client_simulator.data_mean.detach().cpu(),
        "data_std": client_simulator.data_std.detach().cpu(),
    }
    torch.save(series, file)
    print(f"Saved {len(snapshots)} reconstruction snapshots (with PSNR/SSIM) to: {file}  "
          f"[final PSNR={psnr[-1]:.2f} dB, SSIM={ssim[-1]:.3f}]")
    return series


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
        num_images=4,
        batch_size=4,
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

    optimizer_type = "sgd"
    learning_rate = 0.01
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
        model, proxy_dataloader, optimizer_type, lr=learning_rate, steps=0, loss_fn=loss_fn, device=device,
    )

    training_settings = TrainingSettings(
        epochs=1,
        optimizer_type=optimizer_type,
        training_batch_size=4,
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
    #gia_running.bn_weight = 0.00016        # <- tar bort BNStatisticsRegularization helt
    #gia_running.tv_weight = 0.01
    #gia_running.gradient_loss_type = "l2"
    attacks.append(("GIA Running", gia_running))

    #gia_base_attack = gia_estimate_attack()
    #attacks.append(("GIA Base", gia_base_attack))

    # Each experiment gets its own timestamped folder under outputs/ (nothing is overwritten).
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(__file__).parent / "outputs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing experiment outputs to: {run_dir}")

    for attack_name, attack in attacks:
        print(f"\n{'='*20} Running {attack_name} Attack {'='*20}\n")
        attack.max_iterations = 4000
        attack.log_interval = 10
        attack = attack.build(client_observations=client_observation)
        reconstruction, attack_config = attack.run_attack(
            target_model=model,
            input_shape=input_shape,
            device=device,
            client_observations=client_observation,
            proxy_dataloader=proxy_dataloader
        )

        # Reference: the loss on the ORIGINAL (ground-truth) image, through the same pipeline.
        # Gradient matching should be ~0; the regularizers reveal their floor value.
        reference_loss = attack.compute_reference_loss(client_simulator.original_inputs)
        print(f"Reference loss on ORIGINAL image: "
              + ", ".join(f"{k}={v:.4f}" for k, v in reference_loss.items()))

        # Save the per-iteration loss curve (with reference lines) and the per-improvement
        # reconstruction series (stacked tensor + PSNR/SSIM measured against ground truth).
        tag = attack_name.replace(" ", "_")
        # Full curve, plus a zoomed one that skips the early transient so later progress is visible.
        plot_loss_history(attack_config["loss_history"], run_dir / f"loss_curve_{tag}.png",
                          reference=reference_loss)
        zoom_start = min(200, len(attack_config["loss_history"]) // 5)
        plot_loss_history(attack_config["loss_history"], run_dir / f"loss_curve_{tag}_zoom.png",
                          reference=reference_loss, start_iteration=zoom_start)
        series = save_reconstruction_series(
            attack.reconstruction_snapshots, client_simulator, device,
            run_dir / f"reconstruction_series_{tag}.pt",
        )
        if series is not None:
            plot_metric_history(series["iterations"].tolist(), series["psnr"].tolist(),
                                series["ssim"].tolist(), run_dir / f"metric_curve_{tag}.png")

        results.append(client_simulator.compute_metrics(reconstruction, attack_config))

    
    visualize_multiple_attacks(results, client_simulator.original_labels, data_mean, data_std,
                               file=f"{run_name}/main_results.png")
    

if __name__ == "__main__":
    main()