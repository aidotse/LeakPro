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

from contextlib import contextmanager
from copy import copy, deepcopy
from pathlib import Path
from types import MethodType
from typing import Iterator, Literal, Optional, Union

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
from torch.utils.data import DataLoader, TensorDataset

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.model_handler import LogitQuery
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler, audit_points_with_reference_models
from leakpro.input_handler.mia_handler import MIAHandler, ordered_query_dataloader
from leakpro.reporting.mia_result import MIAResult
from leakpro.signals.signal_extractor import PytorchModel, align_loss_targets
from leakpro.utils.device import get_device, mark_step
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger
from leakpro.utils.save_load import unwrapped_state_dict
from leakpro.utils.seed import seed_everything


class AttackRAPID(AbstractMIA):
    """Implementation of RAPID with optional registered MIA input reuse."""

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
        use_attack: Optional[str] = Field(default=None,
            description="Registered MIA source; None or RAPID selects ordinary RAPID.")
        use_attack_output: Optional[Literal["logits"]] = Field(default=None,
            description="Source output for computing RAPID's loss features. Logits are used when a source is configured.")
        use_attack_config: dict = Field(default_factory=dict, description="Parameters for the selected source attack")

        @field_validator("use_attack", mode="before")
        @classmethod
        def normalize_use_attack(cls, value:object) -> object:
            """Normalize ordinary RAPID aliases before validating source names."""
            if isinstance(value, str):
                value = value.strip().lower()
                if value in {"none", "rapid"}:
                    return None
                from leakpro.attacks.mia_attacks.attack_factory_mia import AttackFactoryMIA  # noqa: PLC0415
                for name in AttackFactoryMIA.attack_classes:
                    if name.lower() == value:
                        return name
                raise ValueError(f"Unknown MIA source: {value}")
            if value is not None:
                raise ValueError("use_attack must be a registered MIA name or None.")
            return value

        @model_validator(mode="after")
        def validate_source_config(self:Self) -> Self:
            """Keep reference ownership and source configuration explicit."""
            if self.use_attack is None and self.use_attack_config:
                raise ValueError("use_attack_config requires use_attack.")
            if self.use_attack is None and self.use_attack_output is not None:
                raise ValueError("use_attack_output requires use_attack.")
            if self.online and self.num_reference_models < 2:
                raise ValueError("Online RAPID needs at least two references per rotation for IN and OUT coverage.")
            if self.use_attack is not None:
                from leakpro.attacks.mia_attacks.attack_factory_mia import AttackFactoryMIA  # noqa: PLC0415
                source = AttackFactoryMIA.attack_classes[self.use_attack]
                if not source.requires_shadow_models or not callable(getattr(source, "get_audit_logits", None)):
                    raise ValueError(f"{self.use_attack} does not provide audit reference logits.")
            controlled = {"num_shadow_models", "training_data_fraction", "online"} & self.use_attack_config.keys()
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
        for key, value in self.configs.model_dump().items():
            setattr(self, key, value)
        self.source_configs = None
        if self.use_attack is not None:
            from leakpro.attacks.mia_attacks.attack_factory_mia import AttackFactoryMIA  # noqa: PLC0415
            config_cls = AttackFactoryMIA.attack_classes[self.use_attack].AttackConfig
            source_config = self.use_attack_config.copy()
            controlled = {"num_shadow_models": self.num_reference_models + 1,
                          "training_data_fraction": self.training_data_fraction, "online": self.online}
            for key, value in controlled.items():
                if key in config_cls.model_fields:
                    source_config[key] = value
            self.source_configs = config_cls(**source_config)
            effective = self.source_configs.model_dump()
            for key in controlled:
                effective.pop(key, None)
            self.configs = self.configs.model_copy(update={"use_attack_config": effective})
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
        original, references = self._query_losses(
            reference_models[0], reference_models[1:], self.reference_model_indices[1:], self.shadow_population,
        )
        losses = np.concatenate((original[None, :], references))
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
        # A logits source loads the other pool members through its cache.
        model_indices = self.reference_model_indices[:1] if self.use_attack else self.reference_model_indices
        reference_models, _ = models.get_shadow_models(model_indices)
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
        """Compute per-record classification losses for direct or source queries."""
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
        original, reference_losses = self._query_losses(model, references, self.reference_model_indices, indices)
        features, _ = self._loss_features(original, reference_losses, membership)
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

    def _query_losses(self: Self, model: PytorchModel, references: list[PytorchModel],
                      reference_indices: list[int], indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Query each requested model once per loss pass."""
        if self.use_attack is not None:
            return self._get_source_losses(model, reference_indices, indices)
        return self._get_mean_loss(model, indices), np.array([self._get_mean_loss(ref, indices) for ref in references])

    @staticmethod
    def _source_positions(source_ids:Optional[np.ndarray], indices:np.ndarray) -> np.ndarray:
        """Align source rows by their population IDs."""
        if source_ids is None:
            raise ValueError("Source results must identify every requested sample exactly once.")
        order = np.argsort(source_ids)
        sorted_ids = source_ids[order]
        if not np.array_equal(sorted_ids, np.sort(indices)) or np.any(sorted_ids[:-1] == sorted_ids[1:]):
            raise ValueError("Source results must identify every requested sample exactly once.")
        return order[np.searchsorted(sorted_ids, indices)]

    def _get_source_losses(self:Self, model:PytorchModel, reference_indices:list[int],
                           indices:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Use the logit inputs that the selected source reads when scoring."""
        with self._source_attack(model, indices) as source:
            positions = self._source_positions(source.audit_dataset["data"], indices)

            def loss(query: LogitQuery) -> np.ndarray:
                if not isinstance(query, tuple) or len(query) != 2:
                    raise ValueError("RAPID's logits source must provide labels from each model query.")
                values = self._classification_losses(*query)
                if values.shape != indices.shape:
                    raise ValueError("Source queries must return one logit and label row per requested record.")
                return values[positions]

            target, references = source.get_audit_logits(reference_indices)
            original = loss(target)
            reference_losses = np.array([loss(query) for query in references])
            if len(reference_losses) != len(reference_indices):
                raise ValueError("Source queries must return one prediction array per reference model.")
            return original, reference_losses

    def _get_source_handler(self:Self, model:PytorchModel, indices:np.ndarray) -> MIAHandler:
        """Copy mutable source settings and partition record IDs without membership labels."""
        order = np.random.default_rng(self.random_seed).permutation(np.argsort(indices))
        source_handler = copy(self.handler)
        # Input handlers attach trainer methods to each instance.
        for name, method in vars(self.handler).items():
            if isinstance(method, MethodType) and method.__self__ is self.handler:
                setattr(source_handler, name, MethodType(method.__func__, source_handler))
        split = len(indices) // 2
        source_handler.train_indices = indices[order[:split]]
        source_handler.test_indices = indices[order[split:]]
        # Recreate the model so that caller-registered hooks stay on the caller's instance.
        models = ShadowModelHandler()
        if model is self.target_model:
            blueprint = self.handler.target_model_blueprint
            init_params = self.handler.target_model_metadata.init_params or {}
        else:
            blueprint, init_params = models.model_blueprint, models.init_params
        source_handler.target_model = models.load_model_state(blueprint(**init_params), unwrapped_state_dict(model.model_obj))
        source_handler._criterion = deepcopy(model.loss_fn)
        source_handler.target_model_metadata = self.handler.target_model_metadata.model_copy(update={
            "train_indices": source_handler.train_indices, "test_indices": source_handler.test_indices
        }, deep=True)
        source_handler.dataloader_config = deepcopy(self.handler.dataloader_config)
        source_handler.configs = self.handler.configs.model_copy(deep=True)
        source_handler.configs.audit.random_seed = self.random_seed
        stage = "shadow" if np.array_equal(indices, self.shadow_population) else "target"
        # Use a separate cache for source queries with verified loader order.
        source_handler.configs.audit.output_dir = str(Path(self.handler.configs.audit.output_dir) /
                                                    "attack_objects" / "rapid" / "sources" / self.attack_id / stage)
        get_loader = source_handler.get_dataloader

        def source_loader(dataset_indices:np.ndarray, params:Optional[dict]=None, batch_size:Optional[int]=None,
                          shuffle:Optional[bool]=None) -> DataLoader:
            """Preserve source query order and delegate training loaders as requested."""
            if shuffle is False:
                return ordered_query_dataloader(get_loader, dataset_indices, params, batch_size)
            return get_loader(dataset_indices, params=params, batch_size=batch_size, shuffle=shuffle)

        source_handler.get_dataloader = source_loader
        return source_handler

    @contextmanager
    def _source_attack(self:Self, model:PytorchModel, indices:np.ndarray) -> Iterator[AbstractMIA]:
        """Scope the factory's shared handlers to a private source audit."""
        from leakpro.attacks.mia_attacks.attack_factory_mia import AttackFactoryMIA  # noqa: PLC0415

        source_handler = self._get_source_handler(model, indices)
        source_cls = AttackFactoryMIA.attack_classes[self.use_attack]
        storage_path = ShadowModelHandler().storage_path
        with AttackFactoryMIA._model_handlers(source_cls, source_handler, isolated=True) as (shadow_handler, _):
            # The source reads shared weights and keeps its own query cache.
            shadow_handler.storage_path = storage_path
            yield source_cls(source_handler, self.source_configs.model_dump())

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
