#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Serializable extraction result contract."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from torch import Tensor

from leakpro.attacks.extraction_attacks.utils_generative import json_safe


class CandidateRecord(BaseModel):
    """Metadata for one retained candidate."""

    image_index: int = Field(
        ge=0, validation_alias=AliasChoices("image_index", "sample_index"), serialization_alias="sample_index"
    )
    source: str
    score: Optional[float] = None
    support_indices: List[int] = Field(default_factory=list)
    nearest_reference_index: Optional[int] = Field(default=None, ge=0)
    nearest_reference_distance: Optional[float] = Field(default=None, ge=0)
    verified: Optional[bool] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")

    @property
    def sample_index(self) -> int:
        """Return the retained sample's index."""
        return self.image_index


class ExtractionResult:
    """LeakPro-compatible result containing samples, metrics, and provenance."""

    def __init__(
        self,
        *,
        name: str,
        result_id: str,
        config: BaseModel | dict[str, Any],
        images: Tensor | None = None,
        samples: Tensor | None = None,
        candidates: list[CandidateRecord],
        metrics: dict[str, Any],
        provenance: dict[str, Any],
        execution_trace: list[dict[str, Any]] | None = None,
        overwrite: bool = False,
    ) -> None:
        if re.fullmatch(r"[A-Za-z0-9._-]+", result_id) is None:
            raise ValueError("result_id contains unsafe path characters.")
        if (images is None) == (samples is None):
            raise ValueError("Provide exactly one of images or samples.")
        if images is not None:
            if images.ndim != 4:
                raise ValueError("ExtractionResult.images must be BCHW.")
            samples = images.to(dtype=torch.float32)
        if samples.ndim < 1:
            raise ValueError("ExtractionResult.samples must have a batch dimension.")
        if not torch.isfinite(samples).all():
            raise ValueError("ExtractionResult.samples contains NaN or infinity.")
        if len(candidates) != samples.shape[0]:
            raise ValueError("Candidate metadata count must match the number of samples.")
        if any(candidate.sample_index >= samples.shape[0] for candidate in candidates):
            raise ValueError("Candidate sample index is outside the sample batch.")
        self.name = name
        self.id = result_id
        self.config = config.model_dump(mode="json") if isinstance(config, BaseModel) else dict(config)
        self.samples = samples.detach().cpu().contiguous()
        self._array_key = "images" if images is not None else "samples"
        self.candidates = candidates
        self.metrics = json_safe(metrics)
        self.provenance = json_safe(provenance)
        self.execution_trace = json_safe(execution_trace or [])
        self.overwrite = overwrite

    @property
    def images(self) -> Tensor:
        """Return image samples for existing image consumers."""
        if self.samples.ndim != 4:
            raise ValueError("ExtractionResult.images must be BCHW.")
        return self.samples

    @property
    def result(self) -> dict[str, Any]:
        """Return a report-friendly metadata view without embedding samples."""
        return {
            "name": self.name,
            "id": self.id,
            "config": json_safe(self.config),
            "metrics": self.metrics,
            "provenance": self.provenance,
            "execution_trace": self.execution_trace,
            "candidate_count": len(self.candidates),
            "candidates": [
                candidate.model_dump(mode="json", by_alias=self._array_key == "samples") for candidate in self.candidates
            ],
        }

    def save(self, attack_obj: object | None = None, output_dir: str | os.PathLike[str] = "./leakpro_output") -> None:  # noqa: ARG002
        """Save candidate samples and metadata, replacing existing results only when requested."""
        output_root = Path(output_dir).resolve()
        results_dir = output_root / "results"
        result_dir = results_dir / self.id
        data_dir = output_root / "data_objects"
        data_path = data_dir / f"{self.id}.json"
        results_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        if not self.overwrite and (result_dir.exists() or data_path.exists()):
            raise FileExistsError(
                f"Extraction result {self.id!r} already exists. Set overwrite_results=true to replace it."
            )
        with tempfile.TemporaryDirectory(dir=output_root, prefix=f".{self.id}-") as temporary_dir:
            staging = Path(temporary_dir)
            staged_result = staging / "result"
            staged_result.mkdir()
            metadata = json.dumps(self.result, indent=2, sort_keys=True, allow_nan=False) + "\n"
            (staged_result / "result.json").write_text(metadata, encoding="utf-8")
            np.savez_compressed(staged_result / "candidates.npz", **{self._array_key: self.samples.numpy()})
            staged_data = staging / "data.json"
            staged_data.write_text(metadata, encoding="utf-8")
            if result_dir.exists():
                shutil.rmtree(result_dir)
            try:
                os.replace(staged_result, result_dir)
                os.replace(staged_data, data_path)
            except BaseException:
                shutil.rmtree(result_dir, ignore_errors=True)
                data_path.unlink(missing_ok=True)
                raise
