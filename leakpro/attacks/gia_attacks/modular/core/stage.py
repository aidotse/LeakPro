#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Optimization stage: a representation, losses and a driver run on a state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from leakpro.attacks.gia_attacks.modular.core.interfaces import (
    Aggregator,
    Callback,
    Driver,
    LossTerm,
    Representation,
    TargetSource,
)

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.state import Context, State


@dataclass(frozen=True)
class OptimStage:
    """Frozen config for one optimization stage — also its own executor."""

    losses: dict[str, tuple[LossTerm, float]]
    rep: Representation
    driver: Driver
    target_source: TargetSource
    aggregator: Aggregator | None = None

    def tunables(self) -> dict[str, float]:
        """Flat dotted-key HPO surface. Driver contributes its own tunables under 'driver.*'."""
        flat: dict[str, float] = {f"loss.{n}.weight": w for n, (_, w) in self.losses.items()}
        if hasattr(self.driver, "tunables"):
            flat.update({f"driver.{k}": v for k, v in self.driver.tunables().items()})
        return flat

    def with_overrides(self, flat: dict[str, float]) -> OptimStage:
        """Returns a copy with loss weights and driver settings taken from the overrides."""
        new_losses = {n: (t, flat.get(f"loss.{n}.weight", w)) for n, (t, w) in self.losses.items()}
        new_driver = self.driver
        if hasattr(self.driver, "with_overrides"):
            driver_flat = {k[len("driver."):]: v for k, v in flat.items() if k.startswith("driver.")}
            if driver_flat:
                new_driver = self.driver.with_overrides(driver_flat)
        return replace(self, losses=new_losses, driver=new_driver)

    def run(
        self,
        state: State,
        ctx: Context,
        stage_idx: int = 0,
        callbacks: list[Callback] | None = None,
    ) -> State:
        """Runs the driver on the state, firing callbacks around the stage."""
        cbs = callbacks or []

        for cb in cbs:
            cb.on_stage_start(stage_idx, state, ctx)

        # Every run resource lives on bound copies and on ctx.session, never on this config.
        ctx.session.begin_stage(ctx)
        rep = self.rep.bind(state, ctx)
        target_source = self.target_source.bind(state, ctx)
        losses = {n: (t.bind(ctx), w) for n, (t, w) in self.losses.items()}
        bound = replace(self, rep=rep, target_source=target_source, losses=losses)
        try:
            result = self.driver.run(bound, ctx, callbacks=cbs, stage_idx=stage_idx)
            if self.aggregator is not None:
                result = self.aggregator.aggregate(result, ctx, bound)
        finally:
            for term, _ in losses.values():
                term.release()
        state.input = result.input
        # A private copy: the next stage binds from it and must not share tensors with this one.
        state.target = result.target.detach().clone() if result.target is not None else None
        state.artifacts = result.artifacts
        state.loss = result.loss
        state.iteration += result.iterations

        for cb in cbs:
            cb.on_stage_end(stage_idx, state, ctx)

        return state
