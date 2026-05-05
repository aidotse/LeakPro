#
# Copyright 2023-2026 AI Sweden
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Federated Learning client simulator for gradient inversion attack evaluation.

This module provides a clear separation between client-side and server-side operations
in a federated learning setting. The client simulator runs on a "separate machine" and
only exposes information according to the FL protocol and threat model.

Architecture:
- Client has private data and trains locally
- Client sends observations (gradients, BN stats) to server
- Server (attacker) tries to reconstruct from observations
- Client receives reconstruction and computes metrics
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.training_simulator import (
    MultiEpochTrainingSimulation,
)
from leakpro.fl_utils.matching_utils import match_images
from leakpro.metrics.attack_result import GIAResults

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.training_simulator import (
        TrainingSettings,
    )

@dataclass
class ClientObservations:
    """Information observable by the server from a client update.

    Different threat models expose different information:
    - Standard FL: Only gradients
    - Huang (Model B): Gradients + post-training BN statistics
    - GIA Running (Model B+): Gradients + pre/post BN statistics + num_images
    - GIA Estimate (Model C): Gradients only (attacker uses proxy data)
    """

    # Always sent in FL
    gradients: List[torch.Tensor]

    # Optionally sent depending on protocol/threat model
    post_bn_stats: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    pre_bn_stats: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

    # Number of images the client trained on (needed for BN correction and reconstruction)
    num_images: Optional[int] = None

    # Shape of input data [N, C, H, W] - needed for initialization
    input_shape: Optional[tuple[int, ...]] = None

    # Training configuration - allows attacker to match client's training setup
    training_settings: Optional["TrainingSettings"] = None

    # Optional: labels for oracle mode (unrealistic but useful for testing)
    labels: Optional[torch.Tensor] = None
    data_mean: Optional[torch.Tensor] = None
    data_std: Optional[torch.Tensor] = None
    num_classes: Optional[int] = None


class FLClientSimulator:
    """Simulates a federated learning client on a separate machine.

    This class represents the client side of FL and maintains strict separation
    from the server (attacker). It:
    1. Receives a model from the server
    2. Trains locally on private data
    3. Returns only observations defined by the threat model
    4. Computes reconstruction quality metrics when it receives results

    The client NEVER shares its private data directly with the server.
    """

    def __init__(
        self,
        client_data: DataLoader,
        data_mean: torch.Tensor,
        data_std: torch.Tensor,
        num_classes: int,
        device: torch.device,
    ) -> None:
        """Initialize FL client simulator.

        Args:
            client_data: Client's private training data
            data_mean: Mean for denormalization (for metrics)
            data_std: Std for denormalization (for metrics)
            num_classes: Number of classes in the dataset
            device: Device to run on

        """
        self.client_data = client_data
        self.data_mean = data_mean
        self.data_std = data_std
        self.num_classes = num_classes
        self.device = device

        # Store original data for metric computation (client-side only)
        self.original_inputs = None
        self.original_labels = None
        self._cache_original_data()

    def _cache_original_data(self) -> None:
        """Cache the original data for later metric computation."""
        data, labels = next(iter(self.client_data))
        self.original_inputs = data.to(self.device)
        self.original_labels = labels.to(self.device)

    def train_and_observe(
        self,
        server_model: nn.Module,
        training_settings: "TrainingSettings",
        loss_fn: nn.Module,
        threat_model: str = "gradients_only",
        send_labels_to_server: bool = False,
    ) -> ClientObservations:
        """Simulate client training and return observable information.

        This is the key method that maintains the client/server boundary.
        The server_model is NOT modified - we work on a local copy.

        Args:
            server_model: Model received from server
            training_settings: Training configuration (epochs, optimizer, etc.)
            loss_fn: Loss function to use for training
            threat_model: What information the protocol/threat model exposes:
                - "gradients_only": Standard FL (most realistic)
                - "huang": Model B - gradients + post-training BN stats
                - "gia_running": Model B+ - gradients + pre/post BN stats + num_images
            send_labels_to_server: If True, include labels (for oracle mode testing)

        Returns:
            ClientObservations containing only what the protocol exposes

        """
        # CLIENT WORKS ON A LOCAL COPY (separate machine)
        client_model = deepcopy(server_model)
        client_model.to(self.device)

        observations = ClientObservations(
            gradients=[],
            data_mean=self.data_mean,
            data_std=self.data_std,
            num_classes=self.num_classes,
            training_settings=training_settings,
            input_shape=self.original_inputs.shape,
        )

        # Capture pre-training BN statistics if needed
        if threat_model == "gia_running":
            observations.pre_bn_stats = self._capture_bn_stats(client_model)
            observations.num_images = self.client_data.batch_size

        # Create training simulator for client-side training
        # Client always uses "client" mode (realistic with shuffling)
        training_simulator = MultiEpochTrainingSimulation(
            epochs=training_settings.epochs,
            optimizer_type=training_settings.optimizer_type,
            batch_size=training_settings.training_batch_size,
            compute_mode=training_settings.compute_mode,
            model_mode=training_settings.model_mode,
            shuffle_mode="client",  # Client uses realistic shuffling
        )

        # CLIENT TRAINS LOCALLY (server cannot observe this process)
        gradients = training_simulator.simulate_training(
            model=client_model,
            input_data=self.original_inputs,
            labels=self.original_labels,
            loss_fn=loss_fn,
        )
        observations.gradients = gradients

        # Capture post-training BN statistics if protocol exposes them
        if threat_model in ["huang", "gia_running"]:
            observations.post_bn_stats = self._capture_bn_stats(client_model)

        # Optionally include labels (for oracle mode - unrealistic but useful for testing)
        if send_labels_to_server:
            observations.labels = self.original_labels

        # Return only what the FL protocol would transmit to server
        return observations

    @staticmethod
    def _capture_bn_stats(model: nn.Module) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Capture running_mean/var from all BatchNorm2d layers."""
        stats = []
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                stats.append((
                    module.running_mean.data.clone().detach(),
                    module.running_var.data.clone().detach()
                ))
        return stats

    def compute_metrics(
        self,
        reconstruction: torch.Tensor,
        attack_config: dict,
    ) -> GIAResults:
        """Compute reconstruction quality metrics on client's private data.

        This is called AFTER the server returns a reconstruction. The client
        compares the reconstruction against its private data to evaluate quality.

        Args:
            reconstruction: Reconstructed data from server (attacker)
            attack_config: Configuration dict from attack

        Returns:
            GIAResults with metrics computed against private data

        """
        num_images = reconstruction.shape[0]

        # Denormalize for metric computation
        denorm_reconstruction = reconstruction * self.data_std.to(self.device) + self.data_mean.to(self.device)
        denorm_original = self.original_inputs * self.data_std.to(self.device) + self.data_mean.to(self.device)
        denorm_reconstruction = torch.clamp(denorm_reconstruction, 0.0, 1.0)
        denorm_original = torch.clamp(denorm_original, 0.0, 1.0)

        # Match reconstructions to ground truth (handles permutation invariance)
        matched_indices = None
        if num_images > 1:
            matched_indices, _ = match_images(
                denorm_reconstruction,
                denorm_original,
                metric="mse"
            )
            denorm_reconstruction = denorm_reconstruction[matched_indices]
        else:
            matched_indices = [0]

        # Compute metrics
        psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        psnr_score = psnr(denorm_reconstruction, denorm_original).item()
        ssim_score = ssim(denorm_reconstruction, denorm_original).item()

        return GIAResults(
            original_data=self.original_inputs,
            recreated_data=reconstruction[matched_indices],
            psnr_score=psnr_score,
            ssim_score=ssim_score,
            config=attack_config,
            images=True,
        )


__all__ = ["ClientObservations", "FLClientSimulator"]
