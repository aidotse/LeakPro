#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Handler for memorization extraction attacks.

Extraction only queries the target: it generates text and scores likelihoods. So unlike
:class:`~leakpro.input_handler.mia_handler.MIAHandler` there is no population split, no
criterion, no training and no evaluation. The handler provides:

* ``target_model``: built from ``target.module_path`` / ``target.model_class``. If
  ``target.target_folder`` holds ``model_metadata.pkl`` (and ``target_model.pkl``), the model is
  rebuilt from its ``init_params`` and saved weights, exactly as for MIA. Otherwise the class is
  instantiated without arguments, which suits a wrapper around an off-the-shelf checkpoint.
* ``tokenizer``: from the user handler's ``get_tokenizer(self)`` if it defines one, else
  ``transformers.AutoTokenizer`` on the model's ``pretrained_name_or_path``.
* ``prefixes()``: the prefix pool for conditional prompting, a JSON-lines file at
  ``target.data_path`` with one ``{"prefix": "..."}`` record per line.
"""

import json
import types
from pathlib import Path
from typing import Any, List, Optional

import joblib
import torch
from pydantic import BaseModel

from leakpro.input_handler.user_imports import get_class_from_module, import_module_from_file
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class ExtractionHandler:
    """Target model, tokenizer and prefix pool for extraction attacks.

    Args:
    ----
        configs: The validated LeakPro config.
        user_input_handler: The user's handler class. Only an optional ``get_tokenizer`` is used.

    """

    def __init__(self: Self, configs: Any, user_input_handler: Optional[type] = None) -> None:  # noqa: ANN401
        self.configs = configs
        self._user_handler = user_input_handler
        self._tokenizer = None
        self._prefixes: Optional[List[str]] = None
        self._load_model_class()
        self._load_target_model()

    def _load_model_class(self: Self) -> None:
        module_path = self.configs.target.module_path
        model_class = self.configs.target.model_class
        try:
            module = import_module_from_file(module_path)
            self.target_model_blueprint = get_class_from_module(module, model_class)
        except Exception as e:
            raise ValueError(f"Failed to load the target model class {model_class} from {module_path}") from e
        logger.info(f"Target model blueprint created from {model_class} in {module_path}.")

    def _load_target_model(self: Self) -> None:
        folder = Path(self.configs.target.target_folder)
        metadata_path = folder / "model_metadata.pkl"
        weights_path = folder / "target_model.pkl"
        if metadata_path.exists():
            with metadata_path.open("rb") as f:
                metadata = joblib.load(f)
            init_params = metadata.init_params if isinstance(metadata, BaseModel) else metadata["init_params"]
            self.target_model = self.target_model_blueprint(**init_params)
            if weights_path.exists():
                with weights_path.open("rb") as f:
                    self.target_model.load_state_dict(torch.load(f))
            logger.info(f"Loaded target model from {folder}")
        else:
            self.target_model = self.target_model_blueprint()
            logger.info(f"No model_metadata.pkl in {folder}; instantiated {type(self.target_model).__name__}() as is")
        self.target_model.eval()

    @property
    def tokenizer(self: Self) -> Any:  # noqa: ANN401
        """The target's tokenizer (loaded on first use)."""
        if self._tokenizer is None:
            get_tokenizer = getattr(self._user_handler, "get_tokenizer", None)
            if callable(get_tokenizer):
                self._tokenizer = types.MethodType(get_tokenizer, self)()
            else:
                name = getattr(self.target_model, "pretrained_name_or_path", None)
                if name is None:
                    raise ValueError(
                        "Cannot find a tokenizer: define get_tokenizer(self) on the input handler, or give the "
                        "target model a `pretrained_name_or_path` attribute."
                    )
                from transformers import AutoTokenizer  # noqa: PLC0415
                self._tokenizer = AutoTokenizer.from_pretrained(name)
        return self._tokenizer

    def prefixes(self: Self) -> List[str]:
        """Prefix texts for conditional prompting, in file order."""
        if self._prefixes is None:
            path = Path(self.configs.target.data_path)
            if not path.exists():
                raise FileNotFoundError(f"Prefix pool not found at {path}")
            with path.open(encoding="utf-8") as f:
                self._prefixes = [json.loads(line)["prefix"] for line in f if line.strip()]
            logger.info(f"Loaded {len(self._prefixes)} prefixes from {path}")
        return self._prefixes
