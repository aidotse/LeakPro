#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""ComponentSpec — a serialisable pointer to a registered component."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ComponentSpec(BaseModel):
    """Serialisable reference to a registered component.

    ``type`` is the registry key (e.g. ``"constraint.clip"``).
    ``params`` are forwarded as keyword arguments to the registered factory.
    ``id`` is an optional handle used by ``build_component(live_overrides=…)``
    to substitute a pre-built object instead of calling the registry — useful
    when a heavy model (GAN, diffusion UNet) is already loaded in memory.
    """

    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None
    model_config = ConfigDict(extra="forbid")

    def __repr__(self) -> str:  # noqa: D105
        parts = [f"type={self.type!r}"]
        if self.params:
            parts.append(f"params={self.params!r}")
        if self.id is not None:
            parts.append(f"id={self.id!r}")
        return f"ComponentSpec({', '.join(parts)})"


__all__ = ["ComponentSpec"]
