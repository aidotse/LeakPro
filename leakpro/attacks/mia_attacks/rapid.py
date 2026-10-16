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

import random
from contextlib import ExitStack, contextmanager
from copy import copy, deepcopy
from pathlib import Path
from types import MethodType
from typing import Iterator, Literal, Optional

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.distillation_model_handler import DistillationModelHandler
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler, audit_points_with_reference_models
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.signals.signal import ModelLogits, Signal
from leakpro.signals.signal_extractor import PytorchModel
from leakpro.utils.device import get_device, hpu_is_installed, mark_step
from leakpro.utils.import_helper import Any, Self, Tuple
from leakpro.utils.logger import logger
from leakpro.utils.save_load import hash_indices
from leakpro.utils.seed import seed_everything


class AttackRAPID(AbstractMIA):
    """Implementation of RAPID with optional registered MIA input reuse."""

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
        use_attack: Optional[str] = Field(default=None,
            description="Registered MIA source; None or RAPID selects ordinary RAPID.")
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
        self.source_indices = {}
        self.source_configs = None
        if self.use_attack is not None:
            from leakpro.attacks.mia_attacks.attack_factory_mia import AttackFactoryMIA  # noqa: PLC0415
            config_cls = AttackFactoryMIA.attack_classes[self.use_attack].AttackConfig
            source_config = self.use_attack_config.copy()
            for key, value in {"num_shadow_models": self.num_reference_models,
                               "training_data_fraction": self.training_data_fraction}.items():
                if key in config_cls.model_fields:
                    source_config[key] = value
            self.source_configs = config_cls(**source_config)
            effective = self.source_configs.model_dump()
            for key in ("num_shadow_models", "training_data_fraction"):
                effective.pop(key, None)
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
        with self._random_state():
            logger.info("Preparing RAPID attack")
            self.scoring_model = None
            self.signal_data = None
            self.source_results = {}
            self.source_indices = {}
            seed_everything(self.random_seed)
            shadow_model, reference_models = self._prepare_shadow_models()

            # Cache hits skip model training and consume different RNG state. Reset
            # before querying and fitting so cached runs use the same augmentations
            # and classifier initialization with the same auxiliary models.
            seed_everything(self.random_seed)
            self.shadow_features = self._get_mia_features(
                shadow_model, reference_models, self.shadow_population
            )
            self.scoring_model = self._train_mia_classifier(self.shadow_features, self.shadow_membership)
            self.signal_data = self._get_mia_features(self.target_model, reference_models, self.audit_dataset["data"])

    def _prepare_shadow_models(self: Self) -> Tuple[PytorchModel, list[PytorchModel]]:
        """Train the shadow model on auxiliary data and references under the source's membership policy."""
        auxiliary = self.sample_indices_from_population(include_train_indices=False, include_test_indices=False)
        split = int(len(auxiliary) * self.shadow_data_fraction)
        self.shadow_population = auxiliary[:split]
        self.reference_population = auxiliary[split:]
        online = getattr(self.source_configs, "online", False)
        needs_references = self.use_attack is None or self.use_attack_output == "logits"
        reference_training_pool = np.concatenate((auxiliary, self.audit_dataset["data"])) if online else self.reference_population
        shadow_handler = ShadowModelHandler()
        pools = (self.shadow_population, reference_training_pool) if needs_references else (self.shadow_population,)
        for pool in pools:
            train_size = int(len(pool) * self.training_data_fraction)
            shadow_handler._validate_training_size(train_size, len(pool), self.training_data_fraction)
        if online and needs_references:
            total_inclusions = int(len(reference_training_pool) * self.training_data_fraction) * self.num_reference_models
            max_inclusions = (self.num_reference_models - 1) * len(reference_training_pool)
            if not len(reference_training_pool) <= total_inclusions <= max_inclusions:
                raise ValueError("Online RAPID sources require IN and OUT references for every point; "
                                 "increase num_reference_models or move training_data_fraction towards 0.5.")

        with self._model_cache("shadow", self.shadow_population) as models:
            self.shadow_model_indices = models.create_shadow_models(
                1, self.shadow_population, training_fraction=self.training_data_fraction, sampling_method="random"
            )
            shadow_models, _ = models.get_shadow_models(self.shadow_model_indices)
            self.shadow_membership = models.get_in_indices_mask(
                self.shadow_model_indices, self.shadow_population
            )[:, 0].astype(np.float32)
        self.reference_training_pool = reference_training_pool
        self.reference_model_indices, reference_models = [], []
        if needs_references:
            with self._model_cache("references", reference_training_pool) as models:
                self.reference_model_indices = models.create_shadow_models(
                    self.num_reference_models, reference_training_pool, training_fraction=self.training_data_fraction,
                    sampling_method="balanced" if online else "random"
                )
                if online:
                    out_indices = ~models.get_in_indices_mask(self.reference_model_indices, reference_training_pool).T
                    if not audit_points_with_reference_models(out_indices, require_in_models=True).all():
                        raise ValueError("Cached online references must provide IN and OUT models for every point.")
                reference_models, _ = models.get_shadow_models(self.reference_model_indices)
        if np.unique(self.shadow_membership).size != 2:
            raise ValueError("RAPID's shadow pool must contain both members and non-members.")
        return shadow_models[0], reference_models

    @contextmanager
    def _random_state(self:Self) -> Iterator[None]:
        """Restore the audit's random streams and backend flags after RAPID, including failures."""
        python_state, numpy_state = random.getstate(), np.random.get_state()
        deterministic, benchmark = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
        with ExitStack() as stack:
            stack.enter_context(torch.random.fork_rng(devices=range(torch.cuda.device_count())))
            if torch.backends.mps.is_available():
                stack.callback(torch.mps.set_rng_state, torch.mps.get_rng_state())
            if hpu_is_installed():
                import habana_frameworks.torch.hpu.random as htrandom  # type: ignore[import-not-found]  # noqa: PLC0415
                stack.callback(htrandom.set_rng_state_all, htrandom.get_rng_state_all())
            try:
                yield
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)
                torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = deterministic, benchmark

    @contextmanager
    def _model_cache(self:Self, role:str, pool:np.ndarray) -> Iterator[Any]:
        """Keep RAPID's roles and eligible training pools in separate existing handler caches."""
        models = ShadowModelHandler()
        storage, logits = models.storage_path, models.attack_cache_folder_path
        key = hash_indices(pool, np.array([], dtype=pool.dtype))
        directory = Path(self.handler.configs.audit.output_dir) / "attack_objects" / "rapid" / f"{role}_{key}"
        try:
            models.storage_path = str(directory)
            models.attack_cache_folder_path = str(directory / "attack_cache")
            Path(models.attack_cache_folder_path).mkdir(parents=True, exist_ok=True)
            yield models
        finally:
            models.storage_path, models.attack_cache_folder_path = storage, logits

    def _classification_loss(self:Self, logits:np.ndarray, labels:np.ndarray) -> np.ndarray:
        """Compute RAPID's stable unweighted classification loss using the existing Torch losses."""
        values, targets = torch.as_tensor(logits), torch.as_tensor(labels).long()
        if values.shape[1] == 1:
            return binary_cross_entropy_with_logits(values[:, 0], targets.to(values.dtype), reduction="none").numpy()
        return cross_entropy(values, targets, reduction="none").numpy()

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
                losses.append(self._classification_loss(logits.numpy(), labels.long().numpy()))
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

    def _get_mia_features(self: Self, model: PytorchModel, references: list[PytorchModel], indices: np.ndarray) -> np.ndarray:
        """Pair original loss with calibration from source logits or source scores."""
        if self.use_attack is None:
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
                loss = self._classification_loss(target_logits, labels)
                reference_losses = np.array([self._classification_loss(values, labels) for values in reference_logits])
                reference = np.sum(np.where(out_indices, reference_losses, 0.0), axis=0) / np.sum(out_indices, axis=0)
                original += loss
                calibrated += loss - reference
            original /= self.query_num
            calibrated /= self.query_num
        else:
            original = self._get_mean_loss(model, indices)
            result, source_indices = self._run_use_attack(model, indices)
            if not isinstance(result, MIAResult) or result.signal_values is None:
                raise ValueError("The selected attack must provide per-sample scores; use use_attack_output=logits otherwise.")
            source_scores = np.asarray(result.signal_values)
            if source_scores.shape != original.shape or not np.isfinite(source_scores).all():
                raise ValueError("The selected attack must return one finite score per RAPID sample.")
            if source_indices is None or source_indices.shape != indices.shape or not np.array_equal(
                np.sort(source_indices), np.sort(indices)
            ) or len(np.unique(source_indices)) != len(indices):
                raise ValueError("Source scores must identify every requested sample exactly once; "
                                 "use logits for unindexed results.")
            calibrated = source_scores[np.argsort(source_indices)[np.searchsorted(np.sort(source_indices), indices)]]
            self.source_results["shadow" if np.array_equal(indices, self.shadow_population) else "target"] = result
            self.source_indices["shadow" if np.array_equal(indices, self.shadow_population) else "target"] = source_indices
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
        with self._model_cache("references", self.reference_training_pool) as models:
            out_indices = ~models.get_in_indices_mask(self.reference_model_indices, indices).T
        if out_indices.shape != (len(references), len(indices)) or not audit_points_with_reference_models(
            out_indices, require_in_models=getattr(self.source_configs, "online", False)
        ).all():
            raise ValueError("RAPID source references must cover every requested sample on the required IN/OUT sides.")
        return logits[0], list(logits[1:]), labels.long().numpy(), out_indices

    def _run_use_attack(self:Self, model:PytorchModel, indices:np.ndarray) -> Tuple[MIAResult, Optional[np.ndarray]]:
        """Run a registered source normally, isolating shared state and withholding true membership.

        Source reporting uses a seeded arbitrary partition of sorted sample IDs.
        Its labels are not the target or shadow model's true membership. The source
        must return a finite score for every requested sample; RAPID alone fits on
        known shadow membership and evaluates against target membership.
        """
        from leakpro.attacks.mia_attacks.attack_factory_mia import AttackFactoryMIA  # noqa: PLC0415

        order = np.random.default_rng(self.random_seed).permutation(np.argsort(indices))
        source_handler = copy(self.handler)
        for name, method in vars(self.handler).items():
            if isinstance(method, MethodType) and method.__self__ is self.handler:
                setattr(source_handler, name, MethodType(method.__func__, source_handler))
        split = len(indices) // 2
        source_handler.train_indices = indices[order[:split]]
        source_handler.test_indices = indices[order[split:]]
        source_handler.target_model = deepcopy(model.model_obj)
        source_handler.target_model_metadata = self.handler.target_model_metadata.model_copy(update={
            "train_indices": source_handler.train_indices, "test_indices": source_handler.test_indices
        }, deep=True)
        source_handler.configs = self.handler.configs.model_copy(deep=True)
        stage = "shadow" if np.array_equal(indices, self.shadow_population) else "target"
        source_handler.configs.audit.output_dir = str(Path(self.handler.configs.audit.output_dir) /
                                                    "attack_objects" / "rapid" / self.attack_id / stage)
        shared = {key: getattr(AbstractMIA, key) for key in
                  ("population", "population_size", "target_model", "audit_dataset", "handler", "_initialized")}
        handlers = [AttackFactoryMIA.shadow_model_handler, AttackFactoryMIA.distillation_model_handler]
        states = [vars(handler).copy() for handler in handlers]
        queries = []
        get_loader = source_handler.get_dataloader

        def record_loader(dataset_indices:np.ndarray, *args:Any, **kwargs:Any) -> Any:
            """Retain the IDs of ordered source queries without altering the source attack."""
            loader = get_loader(dataset_indices, *args, **kwargs)
            if isinstance(loader.sampler, SequentialSampler) and not loader.drop_last and (
                loader.num_workers == 0 or getattr(loader, "in_order", True)
            ):
                ids = np.asarray(dataset_indices).copy()
                queries.append(ids)
            return loader

        source_handler.get_dataloader = record_loader
        try:
            shadow_handler = ShadowModelHandler(source_handler)
            DistillationModelHandler(source_handler)
            create = shadow_handler.create_shadow_models

            def create_models(*args:Any, **kwargs:Any) -> Any:
                """Reset the source stream after either cached reuse or reference training."""
                models = create(*args, **kwargs)
                seed_everything(self.random_seed)
                return models

            shadow_handler.create_shadow_models = create_models
            # Reseed after shared cache setup, before a source constructor can
            # initialize fitted components from a cache-dependent random stream.
            seed_everything(self.random_seed)
            source_cls = AttackFactoryMIA.attack_classes[self.use_attack]
            self.source_attack = source_cls(source_handler, self.source_configs.model_dump())
            seed_everything(self.random_seed)
            self.source_attack.prepare_attack()
            return self._score_source(self.source_attack, queries)
        finally:
            for key, value in shared.items():
                setattr(AbstractMIA, key, value)
            for handler, state in zip(handlers, states):
                vars(handler).clear()
                vars(handler).update(state)
            seed_everything(self.random_seed)

    def _score_source(self:Self, source:AbstractMIA, queries:list[np.ndarray]) -> Tuple[MIAResult, Optional[np.ndarray]]:
        """Require returned scalar scores to retain a verifiable record mapping."""
        queries.clear()
        signals = []
        signal = getattr(source, "signal", None)
        if isinstance(signal, Signal):
            def capture_signal(*args:Any, **kwargs:Any) -> Any:
                """Retain a shared signal's actual per-point values and ordered record queries."""
                start = len(queries)
                values = signal(*args, **kwargs)
                if isinstance(values, (np.ndarray, list)):
                    flat = np.asarray(values).reshape(-1)
                    orders = queries[start:]
                    if orders and all(np.array_equal(ids, orders[0]) for ids in orders) and len(flat) == len(orders[0]):
                        signals.append((orders[0].copy(), flat.copy()))
                return values

            source.signal = capture_signal
        result = source.run_attack()
        result_indices = self._source_score_indices(source, result, signals)
        return result, result_indices

    def _source_score_indices(self:Self, source:AbstractMIA, result:MIAResult,
                              signals:list[Tuple[np.ndarray, np.ndarray]]) -> Optional[np.ndarray]:
        """Adapt existing indexed group results or identical shared-signal values; refuse opaque results."""
        explicit = getattr(result, "sample_indices", None)
        if explicit is not None:
            return np.asarray(explicit)
        if getattr(result, "signal_values", None) is None:
            return None
        inside, outside = getattr(source, "in_member_signals", None), getattr(source, "out_member_signals", None)
        if inside is not None and outside is not None and np.array_equal(
            np.ravel(result.signal_values), np.ravel(np.concatenate((inside, outside)))
        ):
            data = getattr(source, "audit_data_indices", source.audit_dataset["data"])
            in_rows = getattr(source, "in_members", source.audit_dataset["in_members"])
            out_rows = getattr(source, "out_members", source.audit_dataset["out_members"])
            return np.asarray(data)[np.concatenate((in_rows, out_rows))]
        orders = [ids for ids, values in signals if np.array_equal(values, np.ravel(result.signal_values))]
        if orders and all(np.array_equal(ids, orders[0]) for ids in orders):
            return orders[0]
        return None

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
        with self._random_state():
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
