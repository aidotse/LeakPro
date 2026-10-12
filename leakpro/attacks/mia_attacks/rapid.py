#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""RAPID membership inference (He et al., ACM CCS 2024, arXiv:2409.00426).

This implementation follows the paper's offline attack and the reference
implementation's positive-loss features: [loss, loss - mean reference loss].
Both inputs have the opposite sign to the scores in Equations 3 and 7, an
equivalent convention for the learned scorer. Target membership labels are
used only for reporting, never for fitting or selecting the scorer.
"""

from typing import Literal, Optional

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler, audit_points_with_reference_models
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.signals import functional
from leakpro.signals.signal import ModelLogits
from leakpro.signals.signal_extractor import PytorchModel
from leakpro.utils.device import get_device, mark_step
from leakpro.utils.import_helper import Self, Tuple
from leakpro.utils.logger import logger
from leakpro.utils.seed import seed_everything


class AttackRAPID(AbstractMIA):
    """Implementation of RAPID with optional LiRA/RMIA input reuse."""

    class AttackConfig(BaseModel):
        """Configuration for the RAPID attack."""

        model_config = ConfigDict(extra="forbid")
        num_reference_models: int = Field(default=4, ge=1, description="Number of reference models")
        shadow_data_fraction: float = Field(default=0.5, gt=0.0, lt=1.0,
                                           description="Fraction of auxiliary data in the shadow pool")
        training_data_fraction: float = Field(default=0.5, gt=0.0, lt=1.0, description="Training fraction of each auxiliary pool")
        query_num: int = Field(default=8, ge=1, description="Number of augmented queries per model and sample")
        augment_strength: Optional[Literal["none", "easy", "medium", "strong"]] = Field(
            default=None, description="Optional query augmentation supported by the user dataset; None inherits its transforms."
        )
        batch_size: int = Field(default=128, ge=1, description="Batch size for querying models and fitting the MIA classifier")
        mia_classifier_epochs: int = Field(default=150, ge=1, description="Number of epochs for training the MIA classifier")
        learning_rate: float = Field(default=0.0002, gt=0.0, allow_inf_nan=False, description="MIA classifier Adam learning rate")
        weight_decay: float = Field(default=5e-4, ge=0.0, allow_inf_nan=False, description="MIA classifier weight decay")
        random_seed: int = Field(default=42, ge=0, le=2**32 - 1, description="Inherits audit.random_seed unless overridden.")
        use_attack: Optional[Literal["lira", "rmia"]] = Field(default=None,
            description="Reuse LiRA/RMIA inputs or scores; None or RAPID selects ordinary RAPID.")
        use_attack_output: Literal["scores", "logits"] = Field(default="scores",
            description="Use source scores or calculate RAPID loss features from the source logits")
        use_attack_config: dict = Field(default_factory=dict, description="Parameters for the selected source attack")

        @field_validator("use_attack", mode="before")
        @classmethod
        def normalize_use_attack(cls, value:object) -> object:
            """Normalize ordinary RAPID aliases before validating source names."""
            if isinstance(value, str):
                value = value.strip().lower()
                if value in {"none", "rapid"}:
                    return None
            return value

        @model_validator(mode="after")
        def validate_source_config(self:Self) -> Self:
            """Keep reference ownership and source configuration explicit."""
            if self.use_attack is None and self.use_attack_config:
                raise ValueError("use_attack_config requires use_attack.")
            if self.use_attack is None and self.use_attack_output != "scores":
                raise ValueError("use_attack_output requires use_attack.")
            controlled = {"num_shadow_models", "training_data_fraction"} & self.use_attack_config.keys()
            if controlled:
                raise ValueError(f"RAPID owns reference training; configure {sorted(controlled)} through RAPID instead.")
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
        super().__init__(handler)
        for key, value in self.configs.model_dump().items():
            setattr(self, key, value)
        self.scoring_model = None
        self.signal_data = None
        self.source_attack = None
        self.source_results = {}
        if self.use_attack is not None:
            # RAPID's factory already initialized the shared handlers. Reuse its
            # registered attack class without creating another set of handlers.
            from leakpro.attacks.mia_attacks.attack_factory_mia import AttackFactoryMIA  # noqa: PLC0415
            source_config = {**self.use_attack_config, "num_shadow_models": self.num_reference_models,
                             "training_data_fraction": self.training_data_fraction}
            self.source_attack = AttackFactoryMIA.attack_classes[self.use_attack](handler, source_config)
            effective = self.source_attack.configs.model_dump()
            for key in ("num_shadow_models", "training_data_fraction"):
                effective.pop(key)
            self.configs = self.configs.model_copy(update={"use_attack_config": effective})
            self._hash_attack()

    def description(self: Self) -> dict:
        """Return a description of the attack."""
        return {
            "title_str": "RAPID",
            "reference": "He et al. Is Difficulty Calibration All We Need? Towards More Practical Membership Inference Attacks. "
                         "ACM CCS 2024. https://arxiv.org/abs/2409.00426",
            "summary": "A learned scorer combines original losses with "
                       f"{self.use_attack.upper() + ' scores' if self.use_attack and self.use_attack_output == 'scores' else 'reference-calibrated losses'}.",  # noqa: E501
            "detailed": "RAPID fits a binary MLP using only shadow membership. It averages original and reference losses "
                        "or reuses source scores. Reference membership follows the selected source's offline or online mode.",
        }

    def prepare_attack(self: Self) -> None:
        """Train auxiliary models and fit the scorer without target membership labels."""
        logger.info("Preparing RAPID attack")
        self.scoring_model = None
        self.signal_data = None
        self.source_results = {}
        seed_everything(self.random_seed)
        shadow_model, reference_models = self._prepare_shadow_models()

        # Cache hits skip model training and consume different RNG state. Reset
        # before querying and fitting so cached runs use the same augmentations
        # and classifier initialization with the same auxiliary models.
        seed_everything(self.random_seed)
        self.shadow_features = self._get_mia_features(
            shadow_model, reference_models, self.shadow_population, self.shadow_membership
        )
        self.scoring_model = self._train_mia_classifier(self.shadow_features, self.shadow_membership)
        membership = np.zeros(self.audit_size, dtype=int)
        membership[self.train_indices] = 1
        self.signal_data = self._get_mia_features(self.target_model, reference_models, self.audit_dataset["data"], membership)

    def _prepare_shadow_models(self: Self) -> Tuple[PytorchModel, list[PytorchModel]]:
        """Train the shadow model on auxiliary data and references under the source's membership policy."""
        auxiliary = self.sample_indices_from_population(include_train_indices=False, include_test_indices=False)
        split = int(len(auxiliary) * self.shadow_data_fraction)
        self.shadow_population = auxiliary[:split]
        self.reference_population = auxiliary[split:]
        online = self.source_attack is not None and self.source_attack.online
        reference_training_pool = np.concatenate((auxiliary, self.audit_dataset["data"])) if online else self.reference_population
        shadow_handler = ShadowModelHandler()
        for pool in (self.shadow_population, reference_training_pool):
            train_size = int(len(pool) * self.training_data_fraction)
            shadow_handler._validate_training_size(train_size, len(pool), self.training_data_fraction)
        if online:
            total_inclusions = int(len(reference_training_pool) * self.training_data_fraction) * self.num_reference_models
            max_inclusions = (self.num_reference_models - 1) * len(reference_training_pool)
            if not len(reference_training_pool) <= total_inclusions <= max_inclusions:
                raise ValueError("Online RAPID sources require IN and OUT references for every point; "
                                 "increase num_reference_models or move training_data_fraction towards 0.5.")

        self.shadow_model_indices = shadow_handler.create_shadow_models(
            1, self.shadow_population, training_fraction=self.training_data_fraction, sampling_method="random"
        )
        self.reference_model_indices = shadow_handler.create_shadow_models(
            self.num_reference_models, reference_training_pool,
            training_fraction=self.training_data_fraction, sampling_method="balanced" if online else "random"
        )
        if online:
            out_indices = ~shadow_handler.get_in_indices_mask(self.reference_model_indices, reference_training_pool).T
            if not audit_points_with_reference_models(out_indices, require_in_models=True).all():
                raise ValueError("Cached online references must provide IN and OUT models for every point.")
        shadow_models, _ = shadow_handler.get_shadow_models(self.shadow_model_indices)
        reference_models, _ = shadow_handler.get_shadow_models(self.reference_model_indices)
        self.shadow_membership = shadow_handler.get_in_indices_mask(
            self.shadow_model_indices, self.shadow_population
        )[:, 0].astype(np.float32)
        if np.unique(self.shadow_membership).size != 2:
            raise ValueError("RAPID's shadow pool must contain both members and non-members.")
        return shadow_models[0], reference_models

    def _get_query_dataloader(self:Self, indices:np.ndarray) -> DataLoader:
        """Construct an ordered query loader with the configured dataset transforms."""
        params = None
        if self.augment_strength is not None:
            params = self.population.return_params().copy()
            # Rebuild the dataset's transforms while preserving normalization.
            params.pop("augment", None)
            params.pop("erase_post_norm", None)
            params["augment_strength"] = self.augment_strength
        loader = self.handler.get_dataloader(indices, params=params, batch_size=self.batch_size, shuffle=False)
        if not isinstance(loader.sampler, SequentialSampler) or (
            loader.num_workers > 0 and not getattr(loader, "in_order", True)
        ):
            raise ValueError("RAPID requires a sequential query dataloader with in_order=True to preserve audit order.")
        return loader

    def _get_mean_loss(self: Self, model: PytorchModel, indices: np.ndarray) -> np.ndarray:
        """Average unweighted classification loss, validating binary and multiclass outputs."""
        loader = self._get_query_dataloader(indices)
        total = np.zeros(len(indices), dtype=np.float64)
        for _ in range(self.query_num):
            losses = []
            for data, labels in loader:
                logits = torch.as_tensor(model.get_logits(data))
                labels = torch.as_tensor(labels).reshape(-1)
                self._validate_classification_logits(logits, labels)
                # The stable shared signal keeps unweighted NLL finite for large logits.
                losses.append(functional.loss(logits.numpy(), labels.long().numpy(), stable=True))
            query_loss = np.concatenate(losses) if losses else np.array([])
            if query_loss.shape != total.shape:
                raise ValueError("RAPID's query dataloader must yield every requested sample; disable drop_last.")
            if not np.isfinite(query_loss).all():
                raise ValueError("RAPID received non-finite classification losses.")
            total += query_loss
        return total / self.query_num

    def _validate_classification_logits(self:Self, logits:torch.Tensor, labels:torch.Tensor) -> None:
        """Validate the same classification contract for losses and composed attack inputs."""
        if logits.ndim != 2 or logits.shape[0] != labels.numel() or logits.shape[1] < 1:
            raise ValueError("RAPID requires classification logits shaped (batch, classes) and one label per sample.")
        if not torch.isfinite(logits).all():
            raise ValueError("RAPID received non-finite model logits.")
        if logits.shape[1] == 1:
            if not torch.all((labels == 0) | (labels == 1)):
                raise ValueError("RAPID requires binary labels 0 or 1 for single-logit models.")
        elif not torch.all((labels == labels.long()) & (labels >= 0) & (labels < logits.shape[1])):
            raise ValueError("RAPID requires integer class labels within the model's class range.")

    def _get_mia_features(self: Self, model: PytorchModel, references: list[PytorchModel], indices: np.ndarray,
                          membership:Optional[np.ndarray]=None) -> np.ndarray:
        """Pair original loss with calibration from source logits or source scores."""
        if self.source_attack is None:
            original = self._get_mean_loss(model, indices)
            reference = np.zeros_like(original)
            for reference_model in references:
                reference += self._get_mean_loss(reference_model, indices)
            calibrated = original - reference / len(references)
        elif self.use_attack_output == "logits":
            original = np.zeros(len(indices), dtype=np.float64)
            calibrated = np.zeros_like(original)
            for _ in range(self.query_num):
                target_logits, reference_logits, labels, out_indices = self._get_use_attack_logits(model, references, indices)
                loss = functional.loss(target_logits, labels, stable=True)
                reference_losses = np.array([functional.loss(values, labels, stable=True) for values in reference_logits])
                reference = np.sum(np.where(out_indices, reference_losses, 0.0), axis=0) / np.sum(out_indices, axis=0)
                original += loss
                calibrated += loss - reference
            original /= self.query_num
            calibrated /= self.query_num
        else:
            original = self._get_mean_loss(model, indices)
            if membership is None:
                raise ValueError("RAPID needs membership labels to align the selected attack's reported rows.")
            result = self._run_use_attack(model, references, indices, membership)
            order = np.concatenate((np.flatnonzero(membership), np.flatnonzero(membership == 0)))
            calibrated = np.empty_like(original)
            if result.signal_values.shape != original.shape or not np.isfinite(result.signal_values).all():
                raise ValueError("The selected attack must return one finite score per RAPID sample.")
            calibrated[order] = result.signal_values
            self.source_results["shadow" if np.array_equal(indices, self.shadow_population) else "target"] = result
        features = np.column_stack((original, calibrated)).astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError("RAPID features must be finite.")
        return features

    def _get_use_attack_logits(self:Self, model:PytorchModel, references:list[PytorchModel], indices:np.ndarray
                              ) -> Tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
        """Query the source's aligned model inputs without using target membership labels."""
        loader = self._get_query_dataloader(indices)
        logits = ModelLogits()([model, *references], self.handler, indices, data_loader=loader)
        labels = torch.as_tensor(loader.dataset.targets).reshape(-1)
        for values in logits:
            self._validate_classification_logits(torch.as_tensor(values), labels)
        out_indices = ~ShadowModelHandler().get_in_indices_mask(self.reference_model_indices, indices).T
        if out_indices.shape != (len(references), len(indices)) or not audit_points_with_reference_models(
            out_indices, require_in_models=self.source_attack.online
        ).all():
            raise ValueError("RAPID source references must cover every requested sample on the required IN/OUT sides.")
        return logits[0], list(logits[1:]), labels.long().numpy(), out_indices

    def _run_use_attack(self:Self, model:PytorchModel, references:list[PytorchModel], indices:np.ndarray,
                         membership:np.ndarray) -> MIAResult:
        """Run the selected MIA on the supplied model's logits and return its ordinary result."""
        source = self.source_attack
        source.audit_dataset = {"data": indices, "in_members": np.flatnonzero(membership),
                                "out_members": np.flatnonzero(membership == 0)}
        target_logits, reference_logits, labels, out_indices = self._get_use_attack_logits(model, references, indices)
        if self.use_attack == "lira":
            source.prepare_from_logits(target_logits, reference_logits, labels, out_indices)
        else:
            n_z = int(source.attack_data_fraction * len(self.reference_population))
            if n_z < 1:
                raise ValueError("RMIA composition needs at least one population sample; increase attack_data_fraction.")
            z_indices = np.random.choice(self.reference_population, n_z, replace=False)
            z_loader = self._get_query_dataloader(z_indices)
            z_logits = ModelLogits()([model], self.handler, z_indices, data_loader=z_loader)[0]
            z_reference_logits = list(ModelLogits()(references, self.handler, z_indices, data_loader=z_loader))
            z_labels = torch.as_tensor(z_loader.dataset.targets).reshape(-1)
            for values in [z_logits, *z_reference_logits]:
                self._validate_classification_logits(torch.as_tensor(values), z_labels)
            z_labels = z_labels.long().numpy()
            source.prepare_from_logits(target_logits, reference_logits, labels, out_indices,
                                       z_logits, z_reference_logits, z_labels)
        source.bayesian_optimization = False
        return source.run_attack()

    def _train_mia_classifier(self: Self, features: np.ndarray, membership: np.ndarray) -> nn.Module:
        """Fit the reference four-layer MLP for a fixed number of epochs."""
        scorer = nn.Sequential(
            nn.Linear(2, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1),
        )
        for layer in scorer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        device = get_device()
        scorer.to(device)
        loader = DataLoader(TensorDataset(torch.from_numpy(features), torch.from_numpy(membership).reshape(-1, 1)),
                            batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(scorer.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        scorer.train()
        for epoch in range(self.mia_classifier_epochs):
            for data, labels in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(scorer(data.to(device)), labels.to(device))
                if not torch.isfinite(loss):
                    raise ValueError("RAPID scorer training produced a non-finite loss.")
                loss.backward()
                optimizer.step()
                mark_step(device)
            logger.debug("RAPID scorer epoch %s/%s", epoch + 1, self.mia_classifier_epochs)
        return scorer.to("cpu").eval()

    def run_attack(self: Self) -> MIAResult:
        """Score samples in audit order and use target labels only for evaluation."""
        logger.info("Running RAPID attack")
        if self.scoring_model is None or self.signal_data is None:
            raise RuntimeError("Call prepare_attack() before running RAPID.")
        loader = DataLoader(TensorDataset(torch.from_numpy(self.signal_data)), batch_size=self.batch_size, shuffle=False)
        scores = []
        with torch.no_grad():
            for (features,) in loader:
                scores.append(torch.sigmoid(self.scoring_model(features)).numpy().reshape(-1))
        signal_values = np.concatenate(scores)
        if not np.isfinite(signal_values).all():
            raise ValueError("RAPID membership scores must be finite.")
        true_membership = np.zeros(self.audit_size, dtype=int)
        true_membership[self.train_indices] = 1
        return MIAResult.from_full_scores(true_membership=true_membership, signal_values=signal_values,
                                          result_name="RAPID", metadata=self.configs.model_dump())
