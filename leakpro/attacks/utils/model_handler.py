#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Abstract class for the model handler."""

import os
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import BadZipFile

import joblib
import numpy as np
from torch import load
from torch.nn import Module

from leakpro.input_handler.mia_handler import MIAHandler, _fix_bn_inplace
from leakpro.input_handler.user_imports import (
    get_class_from_module,
    get_criterion_mapping,
    get_optimizer_mapping,
    import_module_from_file,
)
from leakpro.signals.signal import ModelLogits
from leakpro.signals.signal_extractor import PytorchModel
from leakpro.utils.import_helper import Any, Optional, Self, Tuple, Union
from leakpro.utils.logger import logger
from leakpro.utils.save_load import hash_config, hash_indices, hash_model


class ModelHandler():
    """Class to handle models used in attacks."""

    def __init__(
        self:Self,
        handler: MIAHandler,
        caller:str=None
    )->None:
        """Initialize the ModelHandler class."""

        self.handler = handler

        caller_configs = getattr(handler.configs, caller) if caller is not None else None
        self.use_target_model_setup = caller_configs is None
        optimizer_name, criterion_name = self._load_model_setup(caller_configs)

        self.optimizer_name = optimizer_name.lower()
        self.criterion_name = criterion_name.lower()

        # Get optimizer class
        self.optimizer_class = self._get_optimizer_class(optimizer_name)

        # Get criterion class
        self.criterion_class = self._get_criterion_class(criterion_name)

        # Set the storage paths for objects created by the handler
        storage_path = handler.configs.audit.output_dir
        if storage_path is not None:
            self.storage_path = f"{storage_path}/attack_objects/{caller}"
            if not os.path.exists(self.storage_path):
                # Create the folder
                os.makedirs(self.storage_path)
                logger.info(f"Created folder {self.storage_path}")
        else:
            raise ValueError("Storage path not provided")

        target = handler.configs.target
        self.target_model_hash = hash_config({
            "weights": hash_model(handler.target_model), "class": target.model_class,
            "definition": sha256(Path(target.module_path).read_bytes()).hexdigest(),
            "init_params": handler.target_model_metadata.init_params or {},
        })

        self.population_hash = hash_indices(handler.train_indices, handler.test_indices)

        # Use a fresh audit directory when the dataset or preprocessing changes.
        self.attack_cache_folder_path = f"{storage_path}/attack_cache"
        os.makedirs(self.attack_cache_folder_path, exist_ok=True)

    def _load_model_setup(self:Self, caller_configs) -> Tuple[str, str]:  # noqa: ANN001
        """Load the effective model, optimizer, and criterion setup."""
        target_setup = self.handler.target_model_metadata

        if self.use_target_model_setup:
            self.model_path = self.handler.configs.target.module_path
            self.model_class = self.handler.target_model_blueprint.__name__
            self.model_blueprint = self.handler.target_model_blueprint
            self.init_params = (target_setup.init_params or {}).copy()
            self.optimizer_config = (target_setup.optimizer.params or {}).copy()
            self.loss_config = (target_setup.criterion.params or {}).copy()
            self.epochs = target_setup.epochs
            self.batch_size = target_setup.data_loader.params.get("batch_size", 1)
            return target_setup.optimizer.name, target_setup.criterion.name

        # Allow partial shadow model config by inheriting from the target setup.
        self.model_path = self._get_config_value(caller_configs, "module_path") or self.handler.configs.target.module_path
        self.model_class = self._get_config_value(caller_configs, "model_class") or self.handler.configs.target.model_class
        try:
            self.model_blueprint = self._import_model_from_path(self.model_path, self.model_class)
        except Exception as e:
            raise ValueError(f"Failed to create model blueprint from {self.model_class} in {self.model_path}") from e

        # Inherit defaults from target and apply caller overrides when present.
        caller_init_params = self._get_config_value(caller_configs, "init_params")
        target_init_params = (target_setup.init_params or {}).copy()
        if caller_init_params is None:
            self.init_params = target_init_params
        elif caller_init_params:
            self.init_params = target_init_params
            self.init_params.update(caller_init_params)
        else:
            self.init_params = {}
        optimizer_cfg = self._get_config_value(caller_configs, "optimizer") or target_setup.optimizer
        criterion_cfg = self._get_config_value(caller_configs, "criterion") or target_setup.criterion
        self.optimizer_config = (optimizer_cfg.params or {}).copy()
        self.loss_config = (criterion_cfg.params or {}).copy()
        caller_epochs = self._get_config_value(caller_configs, "epochs")
        self.epochs = caller_epochs if caller_epochs is not None else target_setup.epochs
        target_batch_size = target_setup.data_loader.params.get("batch_size", 1)
        caller_batch_size = self._get_config_value(caller_configs, "batch_size")
        self.batch_size = caller_batch_size if caller_batch_size is not None else target_batch_size
        return optimizer_cfg.name, criterion_cfg.name

    def _get_config_value(self:Self, configs:Any, field_name:str) -> Any:
        """Read config values safely from Pydantic models, DotMaps, and dictionaries."""
        if configs is None:
            return None
        if isinstance(configs, dict):
            return configs.get(field_name)
        if hasattr(configs, "get"):
            return configs.get(field_name, None)
        return getattr(configs, field_name, None)



    def cache_logits(self:Self, model:Union[Module, list[Module]], name:str) -> np.ndarray:
        """Compute predictions and publish them together with their record order."""
        indices = np.concatenate((self.handler.train_indices, self.handler.test_indices))
        models = model if isinstance(model, list) else [model]
        logits = np.array(ModelLogits()(models, self.handler, indices)).squeeze(axis=0)
        destination = Path(self.attack_cache_folder_path) / f"{name}.npz"
        with TemporaryDirectory(dir=destination.parent) as staging:
            pending = Path(staging) / destination.name
            np.savez(pending, logits=logits, indices=indices)
            pending.replace(destination)
        return logits

    def load_logits(self:Self, name:str) -> Optional[np.ndarray]:
        """Load predictions in audit order, computing the target on its first use."""
        target = f"target_{self.target_model_hash}"
        if name == "target":
            name = target
        values = self._read_cached_logits(name)
        if values is None and name == target:
            model = PytorchModel(self.handler.target_model, self.handler.get_criterion())
            return self.cache_logits(model, name)
        return values

    def _read_cached_logits(self:Self, name:str) -> Optional[np.ndarray]:
        """Read an intact cache entry for the current records."""
        path = Path(self.attack_cache_folder_path) / f"{name}.npz"
        try:
            with np.load(path, allow_pickle=False) as cached:
                indices = cached["indices"]
                current = np.concatenate((self.handler.train_indices, self.handler.test_indices))
                if indices.shape != current.shape:
                    return None
                aligned = None
                if not np.array_equal(indices, current):
                    order = indices.argsort(kind="stable")
                    current_order = current.argsort(kind="stable")
                    if not np.array_equal(indices[order], current[current_order]):
                        return None
                    aligned = np.empty_like(order)
                    aligned[current_order] = order
                logits = cached["logits"]
        except FileNotFoundError:
            return None
        except (BadZipFile, EOFError, KeyError, ValueError, TypeError):
            logger.warning(f"Recomputing unreadable cached predictions at {path}")
            return None
        if logits.ndim < 1 or len(logits) != len(indices):
            return None
        return logits if aligned is None else logits[aligned]

    def _import_model_from_path(self:Self, module_path:str, model_class:str)->None:
        """Import the model from the given path.

        Args:
        ----
            module_path (str): The path to the module.
            model_class (str): The name of the model class.

        """
        try:
            module = import_module_from_file(module_path)
            return get_class_from_module(module, model_class)
        except Exception as e:
            raise ValueError(f"Failed to create model blueprint from {model_class} in {module_path}") from e

    def _get_optimizer_class(self:Self, optimizer_name:str) -> None:
        """Get the optimizer class based on the optimizer name.

        Args:
        ----
            optimizer_name (str): The name of the optimizer.

        """
        optimizer_name = optimizer_name.lower()
        try:
            return get_optimizer_mapping()[optimizer_name]
        except Exception as e:
            raise ValueError(f"Failed to create optimizer from {optimizer_name}") from e

    def _get_criterion_class(self:Self, criterion_name:str)->None:
        """Get the criterion class based on the criterion name.

        Args:
        ----
            criterion_name (str): The name of the criterion.

        """
        criterion_name = criterion_name.lower()
        try:
            return get_criterion_mapping()[criterion_name]
        except Exception as e:
            raise ValueError(f"Failed to create criterion from {criterion_name}") from e

    def _get_model_criterion_optimizer(self:Self) -> Tuple[Module, Module, Module]:
        """Get the model, criterion, and optimizer from the handler or config."""

        # Set up shadow model from config file
        if self.use_target_model_setup:
            model, criterion, optimizer = self.handler.get_target_replica()
        else:
            model = self.model_blueprint(**self.init_params)
            optimizer = self.optimizer_class(model.parameters(), **self.optimizer_config)
            criterion = self.criterion_class(**self.loss_config)
        return model, criterion, optimizer

    def get_criterion(self:Self) -> Module:
        """Get the criterion from the handler or config."""

        return self.criterion_class(**self.loss_config)

    def _load_model(self:Self, model_path:str) -> Tuple[Module, Module]:
        """Load a model from a path.

        Args:
        ----
            model_path (str): The path to the saved model.

        Returns:
        -------
            Module: The loaded shadow model.
            Module: The loaded criterion.

        """
        # First create the blueprint to inject the weights
        try:
            model = self.model_blueprint(**self.init_params)  # noqa: E501
            criterion = self.handler.get_criterion() if self.use_target_model_setup else self.criterion_class(**self.loss_config)
        except Exception as e:
            raise ValueError("Failed to create model from blueprint") from e

        # Then load the weights and insert them into the model
        try:
            with open(model_path, "rb") as f:
                state_dict = load(f)
            try:
                model.load_state_dict(state_dict)
            except RuntimeError:
                _fix_bn_inplace(model)
                model.load_state_dict(state_dict)
                logger.info(f"Applied BN→GN fix when loading {model_path}")
            logger.info(f"Loaded model from {model_path}")
            return model, criterion
        except FileNotFoundError as e:
            raise ValueError(f"Model file not found at {model_path}") from e

    def _load_metadata(self:Self, metadata_path:str) -> dict:
        """Load metadata from a saved state.

        Args:
        ----
            metadata_path (str): The path to the saved metadata.

        Returns:
        -------
            dict: The loaded metadata.

        """

        try:
            with open(metadata_path, "rb") as f:
                return joblib.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Metadata at {metadata_path} not found") from e
