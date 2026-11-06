#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""CMA-ES runs are reproducible: pycma's sampler is seeded from the run's torch RNG or an explicit seed."""
from __future__ import annotations

import pytest
import torch
from torch import nn

pytest.importorskip("cma")

from leakpro.attacks.gia_attacks.modular.components.drivers.cma_es import CMAESDriver  # noqa: E402
from leakpro.attacks.gia_attacks.modular.components.losses import L2GradientLoss  # noqa: E402
from leakpro.attacks.gia_attacks.modular.components.representations.gan import GANRepresentation  # noqa: E402
from leakpro.attacks.gia_attacks.modular.components.targets.fixed import FixedTarget  # noqa: E402
from leakpro.attacks.gia_attacks.modular.core.axes import AxisManager  # noqa: E402
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage  # noqa: E402
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, Context, State, TargetInfo  # noqa: E402
from leakpro.tests.gia.stand_ins import TinyGenerator  # noqa: E402


class _Task:
    def simulate_dummy_signal(self, model_input, target, ctx, *, create_graph=True):  # noqa: ANN001, ANN201, ARG002
        return ClientObservation(gradients=[model_input.sum().view(1)])


class _Identity:
    def to_model_input(self, data):  # noqa: ANN001, ANN201
        return data


def _run(driver: CMAESDriver, torch_seed: int) -> torch.Tensor:
    torch.manual_seed(torch_seed)
    ctx = Context(model=nn.Module(), observation=ClientObservation(gradients=[torch.ones(1)]), modality=_Identity(),
                  task=_Task(), input_shape=(1, 4, 4), device=torch.device("cpu"), axes=AxisManager())
    target_source = FixedTarget()
    target_source._value = TargetInfo(value=torch.zeros(1))
    rep = GANRepresentation(TinyGenerator(4, (1, 4, 4)), latent_dim=4).bind(State(), ctx)
    stage = OptimStage(losses={"l2": (L2GradientLoss(), 1.0)}, rep=rep, driver=driver, target_source=target_source)
    return driver.run(stage, ctx).artifacts["latent"]


def test_same_torch_seed_gives_the_same_search() -> None:
    driver = CMAESDriver(population_size=4, max_generations=3)
    torch.testing.assert_close(_run(driver, 0), _run(driver, 0), rtol=0, atol=0)


def test_explicit_seed_gives_the_same_search() -> None:
    driver = CMAESDriver(population_size=4, max_generations=3, seed=123)
    a, b = _run(driver, 0), _run(driver, 0)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_seed_must_be_positive() -> None:
    with pytest.raises(ValueError, match="seed"):
        CMAESDriver(seed=0)
