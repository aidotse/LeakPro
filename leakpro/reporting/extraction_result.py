#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Serializable extraction images and candidate metadata."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from leakpro.attacks.extraction_attacks.utils_generative import json_safe


class CandidateRecord(BaseModel):
    """Metadata for one retained image."""

    image_index: int = Field(ge=0)
    source: str
    score: Optional[float] = None
    support_indices: List[int] = Field(default_factory=list)
    nearest_reference_index: Optional[int] = Field(default=None, ge=0)
    nearest_reference_distance: Optional[float] = Field(default=None, ge=0)
    verified: Optional[bool] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")


class ExtractionResult:
    """Result containing generated images, metrics, and provenance."""

    def __init__(
        self,
        *,
        name: str,
        result_id: str,
        config: BaseModel | dict[str, Any],
        images: Tensor,
        candidates: list[CandidateRecord],
        metrics: dict[str, Any],
        provenance: dict[str, Any],
    ) -> None:
        if result_id in {".", ".."} or re.fullmatch(r"[A-Za-z0-9._-]+", result_id) is None:
            raise ValueError("result_id contains unsafe path characters.")
        if images.ndim != 4:
            raise ValueError("ExtractionResult.images must be BCHW.")
        images = images.to(dtype=torch.float32)
        if not torch.isfinite(images).all():
            raise ValueError("ExtractionResult.images contains NaN or infinity.")
        self._validate_candidate_indices([candidate.image_index for candidate in candidates], images.shape[0])
        self.name = name
        self.id = result_id
        self.config = config.model_dump(mode="json") if isinstance(config, BaseModel) else dict(config)
        self.images = images.detach().cpu().contiguous()
        self.candidates = candidates
        self.metrics = json_safe(metrics)
        self.provenance = json_safe(provenance)

    @staticmethod
    def _validate_candidate_indices(indices: list[int], image_count: int) -> None:
        if (
            len(indices) != image_count
            or any(type(index) is not int or not 0 <= index < image_count for index in indices)
            or len(set(indices)) != image_count
        ):
            raise ValueError("Each output image must have exactly one candidate record.")

    @property
    def result(self) -> dict[str, Any]:
        """Return metadata without embedding the image arrays."""
        candidates = [candidate.model_dump(mode="json") for candidate in self.candidates]
        self._validate_candidate_indices([candidate["image_index"] for candidate in candidates], self.images.shape[0])
        return {
            "name": self.name,
            "id": self.id,
            "config": json_safe(self.config),
            "metrics": self.metrics,
            "provenance": self.provenance,
            "candidate_count": len(candidates),
            "candidates": candidates,
        }

    def save(self, attack_obj: object | None = None, output_dir: str | os.PathLike[str] = "./leakpro_output") -> None:  # noqa: ARG002
        """Write metadata and images using LeakPro's result directories.

        The scheduler checks overwrite policy before compute. Like other result
        writers, direct saves replace existing files and propagate write errors.
        """
        metadata = json.dumps(self.result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        result_dir = Path(output_dir) / "results" / self.id
        data_dir = Path(output_dir) / "data_objects"
        result_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(result_dir / "candidates.npz", images=self.images.numpy())
        (result_dir / "result.txt").write_text(metadata, encoding="utf-8")
        (data_dir / f"{self.id}.json").write_text(metadata, encoding="utf-8")
