#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""RAPID membership inference (He et al., ACM CCS 2024, arXiv:2409.00426).

Offline RAPID uses the original loss and its difference from the mean OUT
reference loss. The online variant adds the difference from the mean IN loss.
Each reference model takes a turn as the shadow target while fitting the
scorer. Target membership labels are used only for reporting.
"""

from copy import deepcopy
from typing import Literal, Optional, Union

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
from torch.utils.data import DataLoader, TensorDataset

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler, audit_points_with_reference_models
from leakpro.input_handler.mia_handler import MIAHandler, ordered_query_dataloader
from leakpro.reporting.mia_result import MIAResult
from leakpro.signals.signal_extractor import PytorchModel, align_loss_targets
from leakpro.utils.device import get_device, mark_step
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger
from leakpro.utils.seed import seed_everything


class AttackRAPID(AbstractMIA):
    """Calibrate classification losses against a shared pool of reference models."""

    requires_distillation_models = False
    precompute_shadow_logits = False

    class AttackConfig(BaseModel):
        """Configuration for the RAPID attack."""

        model_config = ConfigDict(extra="forbid")
        num_reference_models: int = Field(default=4, ge=1, description="Number of references per shadow rotation")
        online: bool = Field(default=False, description="Use both IN and OUT reference losses")
        shadow_data_fraction: float = Field(default=0.5, gt=0.0, lt=1.0,
                                           description="Fraction of auxiliary records used to fit and validate the scorer")
        training_data_fraction: float = Field(default=0.5, gt=0.0, lt=1.0,
                                             description="Training fraction of each reference model")
        query_num: int = Field(default=8, ge=1,
                               description="Direct queries per model. One query suffices for deterministic inputs.")
        augment_strength: Optional[Literal["none", "easy", "medium", "strong"]] = Field(
            default=None, description="Optional query augmentation supported by the user dataset; None inherits its transforms."
        )
        batch_size: int = Field(default=128, ge=1, description="Batch size for querying models and fitting the MIA classifier")
        mia_classifier_patience: int = Field(default=5, ge=1, description="Epochs without validation improvement before stopping")
        mia_classifier_val_fraction: float = Field(default=0.2, gt=0.0, lt=1.0,
                                                   description="Fraction of scorer records held out for validation")
        learning_rate: float = Field(default=0.0002, gt=0.0, allow_inf_nan=False, description="MIA classifier Adam learning rate")
        weight_decay: float = Field(default=5e-4, ge=0.0, allow_inf_nan=False, description="MIA classifier weight decay")
        random_seed: int = Field(default=42, ge=0, le=2**32 - 1, description="Inherits audit.random_seed unless overridden.")
        use_attack: None = Field(default=None, exclude=True, description="Legacy selector for ordinary RAPID")

        @field_validator("use_attack", mode="before")
        @classmethod
        def normalize_use_attack(cls, value:object) -> None:
            """Accept legacy selectors for ordinary RAPID."""
            if value is None or isinstance(value, str) and value.strip().lower() in {"none", "rapid"}:
                return
            raise ValueError("RAPID computes losses directly. Remove the use_attack setting.")

        @model_validator(mode="after")
        def validate_reference_count(self:Self) -> Self:
            """Require enough references for both sides of online calibration."""
            if self.online and self.num_reference_models < 2:
                raise ValueError("Online RAPID needs at least two references per rotation for IN and OUT coverage.")
            return self

    def __init__(self: Self, handler: MIAHandler, configs: Optional[dict] = None) -> None:
        """Initialize the RAPID attack.

        Args:
        ----
            handler (MIAHandler): The input handler object.
            configs (dict): Configuration parameters for the attack.

        """
        logger.info("Configuring RAPID attack")
        seed_config = {"random_seed": handler.configs.audit.random_seed}
        self.configs = self.AttackConfig(**{**seed_config, **(configs if configs is not None else {})})
        for key, value in self.configs.model_dump().items():
            setattr(self, key, value)
        super().__init__(handler)
        self.scoring_model = None
        self.signal_data = None

    def description(self: Self) -> dict:
        """Return a description of the attack."""
        return {
            "title_str": "RAPID",
            "reference": "He et al. Is Difficulty Calibration All We Need? Towards More Practical Membership Inference Attacks. "
                         "ACM CCS 2024. https://arxiv.org/abs/2409.00426",
            "summary": "A learned scorer combines original and reference-calibrated losses.",
            "detailed": "RAPID trains on leave-one-model-out rotations and validates on separate auxiliary records. "
                        "Offline calibration uses OUT references. Online calibration uses IN and OUT references.",
        }

    def prepare_attack(self: Self) -> None:
        """Train auxiliary models and fit the scorer without target membership labels."""
        logger.info("Preparing RAPID attack")
        self.scoring_model = None
        self.signal_data = None
        seed_everything(self.random_seed)
        reference_models = self._prepare_reference_models()

        # Cache hits skip model training. Use the same seed for queries and scorer fitting.
        seed_everything(self.random_seed)
        membership = ShadowModelHandler().get_in_indices_mask(self.reference_model_indices, self.shadow_population).T
        losses = self._query_losses(reference_models, self.shadow_population)
        features, covered = self._loss_features(losses, losses, membership, rotate=True)
        validation = self._validation_records(membership, covered)
        train_rows, val_rows = covered & ~validation, covered & validation
        self.scoring_model = self._train_mia_classifier(
            features[train_rows], membership[train_rows], features[val_rows], membership[val_rows],
        )
        modes = [(module, module.training) for module in self.target_model.model_obj.modules()]
        try:
            self.signal_data = self._get_mia_features(self.target_model, reference_models, self.audit_dataset["data"])
        finally:
            for module, training in modes:
                module.training = training

    def _prepare_reference_models(self: Self) -> list[PytorchModel]:
        """Reuse a single reference pool for the audit and all scorer rotations."""
        auxiliary = self.sample_indices_from_population(include_train_indices=False, include_test_indices=False)
        split = int(len(auxiliary) * self.shadow_data_fraction)
        self.shadow_population = auxiliary[:split]
        if split < 2:
            raise ValueError("RAPID needs at least two auxiliary records for scorer training and validation.")
        models = ShadowModelHandler()
        self.reference_model_indices = models.create_shadow_models(
            self.num_reference_models + 1, np.arange(self.population_size), training_fraction=self.training_data_fraction,
            precompute_logits=False, seed=self.random_seed,
        )
        reference_models, _ = models.get_shadow_models(self.reference_model_indices)
        return reference_models

    def _validation_records(self: Self, membership: np.ndarray, covered: np.ndarray) -> np.ndarray:
        """Hold out entire records and keep both classes in each scorer split."""
        by_class = np.stack(((covered & membership).any(axis=0), (covered & ~membership).any(axis=0)))
        if (by_class.sum(axis=1) < 2).any():
            raise ValueError("RAPID needs members and non-members in both scorer splits after reference coverage filtering. "
                             "Increase num_reference_models or shadow_data_fraction, or adjust training_data_fraction.")

        mixed = np.flatnonzero(by_class.all(axis=0))
        validation = np.zeros(membership.shape[1], dtype=bool)
        validation[mixed[:1] if mixed.size else by_class.argmax(axis=1)] = True
        available = by_class.any(axis=0)
        size = int(available.sum() * self.mia_classifier_val_fraction)
        # Reserve one record of each class for training before filling the holdout.
        available[(by_class & ~validation).argmax(axis=1)] = False
        extra = max(0, size - validation.sum())
        validation[np.flatnonzero(available & ~validation)[:extra]] = True
        return validation

    def _get_query_dataloader(self:Self, indices:np.ndarray) -> DataLoader:
        """Construct an ordered query loader with the configured dataset transforms."""
        params = None
        if self.augment_strength is not None:
            params = self.population.return_params().copy()
            # Rebuild the dataset's transforms while preserving normalization.
            params.pop("augment", None)
            params.pop("erase_post_norm", None)
            params["augment_strength"] = self.augment_strength
        return ordered_query_dataloader(self.handler.get_dataloader, indices, params, self.batch_size)

    def _get_mean_loss(self: Self, model: PytorchModel, indices: np.ndarray) -> np.ndarray:
        """Average the model's unweighted classification loss over repeated dataloader passes."""
        loader = self._get_query_dataloader(indices)
        total = np.zeros(len(indices), dtype=np.float64)
        for _ in range(self.query_num):
            losses = [self._classification_losses(model.get_logits(data), labels) for data, labels in loader]
            query_loss = np.concatenate(losses) if losses else np.array([])
            if query_loss.shape != total.shape:
                raise ValueError("RAPID's query dataloader must yield every requested sample; disable drop_last.")
            total += query_loss
        return total / self.query_num

    @staticmethod
    def _classification_losses(logits:np.ndarray, labels:Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Compute per-record classification losses from each query."""
        logits = torch.as_tensor(logits)
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)
        labels = torch.atleast_1d(torch.as_tensor(labels))
        if logits.ndim != 2 or logits.shape[0] != len(labels) or logits.shape[1] < 1:
            raise ValueError("RAPID requires classification logits shaped (batch, classes) and one label per sample.")
        if not torch.isfinite(logits).all():
            raise ValueError("RAPID received non-finite model logits.")
        if logits.shape[1] == 1:
            labels = align_loss_targets(logits, labels, nn.BCEWithLogitsLoss)
            if labels.shape != logits.shape or not torch.all((labels >= 0) & (labels <= 1)):
                raise ValueError("RAPID requires binary labels in [0, 1] for single-logit models.")
            loss = binary_cross_entropy_with_logits(logits, labels, reduction="none").reshape(-1)
        else:
            labels = align_loss_targets(logits, labels, nn.CrossEntropyLoss)
            if labels.shape == logits.shape and labels.is_floating_point():
                if not torch.all((labels >= 0) & (labels <= 1)) or not torch.allclose(
                    labels.sum(dim=1), torch.ones_like(labels[:, 0])
                ):
                    raise ValueError("RAPID requires class probabilities that sum to one per sample.")
                labels = labels.to(logits.dtype)
            elif labels.ndim != 1 or not torch.all((labels == labels.long()) & (labels >= 0) & (labels < logits.shape[1])):
                raise ValueError("RAPID requires integer class labels within the model's class range.")
            else:
                labels = labels.long()
            loss = cross_entropy(logits, labels, reduction="none")
        if not torch.isfinite(loss).all():
            raise ValueError("RAPID received non-finite classification losses.")
        return loss.numpy()

    def _get_mia_features(self: Self, model: PytorchModel, references: list[PytorchModel], indices: np.ndarray) -> np.ndarray:
        """Calibrate the audited model's losses against the full reference pool."""
        membership = ShadowModelHandler().get_in_indices_mask(self.reference_model_indices, indices).T
        if not audit_points_with_reference_models(~membership, require_in_models=self.online).all():
            raise ValueError("RAPID references must cover every requested sample on the required IN/OUT sides.")
        losses = self._query_losses([model, *references], indices)
        features, _ = self._loss_features(losses[0], losses[1:], membership)
        return features

    def _loss_features(self: Self, losses: np.ndarray, references: np.ndarray, membership: np.ndarray,
                       *, rotate: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Build calibrated features and mark rows with enough reference coverage."""
        if references.shape != membership.shape:
            raise ValueError("Reference losses must match RAPID's model and sample counts.")
        columns = [losses]
        covered = np.ones(losses.shape, dtype=bool)
        sides = (~membership, membership) if self.online else (~membership,)
        for side in sides:
            selected = np.where(side, references, 0.0)
            count, total = side.sum(axis=0), selected.sum(axis=0, dtype=np.float64)
            if rotate:
                # In each row, remove the model playing the shadow target.
                count, total = count - side, total - selected
            covered &= count > 0
            mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
            columns.append(losses - mean)
        features = np.stack(columns, axis=-1).astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError("RAPID features must be finite.")
        return features, covered

    def _query_losses(self: Self, models: list[PytorchModel], indices: np.ndarray) -> np.ndarray:
        """Query each model for mean per-record losses."""
        return np.array([self._get_mean_loss(model, indices) for model in models])

    def _train_mia_classifier(self: Self, features: np.ndarray, membership: np.ndarray,
                              val_features: np.ndarray, val_membership: np.ndarray) -> nn.Module:
        """Fit the MLP until held-out validation loss stops improving."""
        scorer = nn.Sequential(
            nn.Linear(features.shape[1], 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1),
        )
        for layer in scorer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        device = get_device()
        scorer.to(device)
        scale = np.maximum(np.abs(features).max(axis=0), 1.0)
        train = TensorDataset(torch.from_numpy(features / scale), torch.tensor(membership, dtype=torch.float32).reshape(-1, 1))
        validation = TensorDataset(
            torch.from_numpy(val_features / scale), torch.tensor(val_membership, dtype=torch.float32).reshape(-1, 1),
        )
        loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(validation, batch_size=self.batch_size)
        optimizer = torch.optim.Adam(scorer.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        best_loss, stale_epochs = np.inf, 0
        while stale_epochs < self.mia_classifier_patience:
            scorer.train()
            for data, labels in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(scorer(data.to(device)), labels.to(device))
                if not torch.isfinite(loss):
                    raise ValueError("RAPID scorer training produced a non-finite loss.")
                loss.backward()
                optimizer.step()
                mark_step(device)
            scorer.eval()
            with torch.no_grad():
                val_loss = sum(criterion(scorer(data.to(device)), labels.to(device)).item() * len(labels)
                               for data, labels in val_loader) / len(validation)
            if not np.isfinite(val_loss):
                raise ValueError("RAPID scorer validation produced a non-finite loss.")
            if val_loss < best_loss - 1e-4:
                best_loss, stale_epochs = val_loss, 0
                best_weights = deepcopy(scorer.state_dict())
            else:
                stale_epochs += 1
        scorer.load_state_dict(best_weights)
        # Fold the training scale into the first layer so audits can pass losses directly.
        with torch.no_grad():
            scorer[0].weight.div_(torch.from_numpy(scale).to(device))
        return scorer.to("cpu").eval()

    def run_attack(self: Self) -> MIAResult:
        """Score samples in audit order and use target labels only for evaluation."""
        logger.info("Running RAPID attack")
        if self.scoring_model is None or self.signal_data is None:
            raise RuntimeError("Call prepare_attack() before running RAPID.")
        scores = []
        with torch.no_grad():
            for features in torch.from_numpy(self.signal_data).split(self.batch_size):
                scores.append(torch.sigmoid(self.scoring_model(features)).numpy().reshape(-1))
        signal_values = np.concatenate(scores)
        if not np.isfinite(signal_values).all():
            raise ValueError("RAPID membership scores must be finite.")
        true_membership = np.zeros(self.audit_size, dtype=int)
        true_membership[self.train_indices] = 1
        return MIAResult.from_full_scores(true_membership=true_membership, signal_values=signal_values,
                                          result_name="RAPID", metadata=self.configs.model_dump(),
                                          audit_indices=self.scored_population_indices())
