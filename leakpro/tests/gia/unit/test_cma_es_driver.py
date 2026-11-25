#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import torch
from torch import nn

cma = pytest.importorskip("cma")

from leakpro.attacks.gia_attacks.modular.components.drivers.cma_es import CMAESDriver
from leakpro.attacks.gia_attacks.modular.components.losses import L2GradientLoss
from leakpro.attacks.gia_attacks.modular.components.representations.gan import GANRepresentation
from leakpro.attacks.gia_attacks.modular.components.targets.fixed import FixedTarget
from leakpro.attacks.gia_attacks.modular.core.axes import AxisManager
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, Context, State, TargetInfo
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.tests.gia.stand_ins import TinyGenerator


class _FakeTask:
    def simulate_dummy_signal(self, model_input, target, ctx, *, create_graph=True):
        return ClientObservation(gradients=[model_input.sum().view(1)])


class _IdentityModality:
    def to_model_input(self, data):
        return data


class TestCMAESDriver:
    def test_run_returns_correct_shape(self):
        device = torch.device("cpu")
        rep = GANRepresentation(TinyGenerator(4, (1, 4, 4)), latent_dim=4)
        target_source = FixedTarget()
        # FixedTarget stores its resolved signal on `_value` (set via bind());
        # here we set it directly to avoid needing a populated State.target.
        target_source._value = TargetInfo(value=torch.zeros(1))
        ctx = Context(
            model=nn.Module(), observation=ClientObservation(gradients=[torch.zeros(1)]),
            modality=_IdentityModality(), task=_FakeTask(),
            input_shape=(1, 4, 4), device=device, axes=AxisManager(),
        )
        rep = rep.bind(State(), ctx)
        stage = OptimStage(
            losses={"l2": (L2GradientLoss(), 1.0)},
            rep=rep, driver=CMAESDriver(population_size=4, max_generations=2),
            target_source=target_source, aggregator=None,
        )
        driver = CMAESDriver(population_size=4, max_generations=2)
        out = driver.run(stage, ctx)
        assert out.input.shape == (1, 1, 4, 4)

    def test_missing_cma_raises_clear_import_error(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "cma":
                raise ImportError("no cma")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        driver = CMAESDriver()
        with pytest.raises(ImportError, match="pip install cma"):
            driver.run(None, None)
