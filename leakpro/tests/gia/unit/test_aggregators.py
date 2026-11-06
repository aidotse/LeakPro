#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for MeanAggregator and BestLossAggregator.

aggregate() maps a DriverResult to a DriverResult. BestLossAggregator selects by the driver's
per-slice losses (``slice_losses``), so these tests pass the per-seed losses directly.
"""
from __future__ import annotations

import types

import torch
from torch import Tensor


from leakpro.attacks.gia_attacks.modular.core.axes import Axis, AxisManager
from leakpro.attacks.gia_attacks.modular.core.interfaces import LossTerm, TargetSource
from leakpro.attacks.gia_attacks.modular.core.loss_inputs import LossInputs
from leakpro.attacks.gia_attacks.modular.core.result import DriverResult
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, State, TargetInfo
from leakpro.attacks.gia_attacks.modular.components.aggregators import BestLossAggregator, MeanAggregator
from leakpro.attacks.gia_attacks.modular.components.drivers import GradientDriver
from leakpro.attacks.gia_attacks.modular.components.representations.pixel import PixelRepresentation
from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget


INPUT_SHAPE = (1, 3, 4, 4)
_DEFAULT_TARGET = TargetInfo(value=torch.tensor([0]))
_EMPTY_OBS = ClientObservation(gradients=[])


# ── Stub losses ───────────────────────────────────────────────────────────────

class _SumLoss(LossTerm):
    """Returns recon.sum(); ignores dummy_obs / target_obs."""
    def compute(self, inputs: LossInputs) -> Tensor:
        return inputs.recon.sum()


# ── Stub TargetSource for reduce delegation tests ─────────────────────────────

class _RecordingTarget(TargetSource):
    """TargetSource whose reduce() records the call and returns a sentinel TargetInfo."""
    SENTINEL = TargetInfo(value=torch.tensor([99]))

    def __init__(self) -> None:
        self.reduce_calls: list[tuple] = []
        self._value = _DEFAULT_TARGET

    def bind(self, state, ctx) -> None:
        pass

    def current(self) -> TargetInfo:
        return self._value

    def slice(self, target: TargetInfo, dim: int, index: int) -> TargetInfo:
        return target

    def reduce(self, target: TargetInfo, dim: int, method: str) -> TargetInfo:
        self.reduce_calls.append((dim, method))
        return self.SENTINEL


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ctx_stub(axes: AxisManager | None = None) -> types.SimpleNamespace:
    """Minimal context stub.  Includes a dummy task/modality so BestLossAggregator
    can call simulate_dummy_signal without a real model."""
    return types.SimpleNamespace(
        axes=axes or AxisManager(),
        observation=_EMPTY_OBS,
        model=None,
        task=types.SimpleNamespace(
            simulate_dummy_signal=lambda *a, **kw: _EMPTY_OBS,
        ),
        modality=types.SimpleNamespace(to_model_input=lambda x: x),
    )


def _bound_fixed_target(target: TargetInfo | None = None) -> FixedTarget:
    ft = FixedTarget()
    ft = ft.bind(State(target=target or _DEFAULT_TARGET), None)
    return ft


def _make_stage(losses: dict, target_source=None) -> OptimStage:
    return OptimStage(
        losses=losses,
        rep=PixelRepresentation(),
        driver=GradientDriver(lr=0.0, max_iter=0, return_best=False, schedule=None),
        target_source=target_source or _bound_fixed_target(),
    )


def _agg(agg, recon: Tensor, target: TargetInfo, ctx, stage: OptimStage,
         slice_losses: Tensor | None = None) -> tuple[Tensor, TargetInfo]:
    """Run ``agg`` on a DriverResult built from ``recon``/``target``; return the collapsed (input, target)."""
    result = DriverResult(input=recon, target=target, artifacts={}, loss=0.0, iterations=0, slice_losses=slice_losses)
    out = agg.aggregate(result, ctx, stage)
    return out.input, out.target


def _sums(recon: Tensor) -> Tensor:
    """Per-seed ``recon.sum()`` (what _SumLoss gives each seed slice of a [S, ...] reconstruction)."""
    return recon.flatten(1).sum(dim=1)


def _seeded_recon(seed_values: list[float]) -> Tensor:
    """Build a [S, *INPUT_SHAPE] tensor with each seed filled with a distinct constant."""
    return torch.stack([torch.full(INPUT_SHAPE, v) for v in seed_values])


# ── MeanAggregator ────────────────────────────────────────────────────────────

class TestMeanAggregator:
    def setup_method(self):
        self.agg = MeanAggregator()

    def test_no_seed_axis_returns_recon_unchanged(self):
        recon = torch.rand(INPUT_SHAPE)
        recon_out, _ = _agg(self.agg, recon, _DEFAULT_TARGET, _ctx_stub(), _make_stage({}))
        assert recon_out is recon

    def test_no_seed_axis_returns_target_unchanged(self):
        target = _DEFAULT_TARGET
        _, target_out = _agg(self.agg, torch.rand(INPUT_SHAPE), target, _ctx_stub(), _make_stage({}))
        assert target_out is target

    def test_with_seed_axis_removes_seed_dimension(self):
        N = 4
        recon = torch.rand(N, *INPUT_SHAPE)
        ctx = _ctx_stub(AxisManager(Axis("seed", N)))
        recon_out, _ = _agg(self.agg, recon, _DEFAULT_TARGET, ctx, _make_stage({}))
        assert recon_out.shape == INPUT_SHAPE

    def test_mean_is_mathematically_correct(self):
        """Mean of two seeds [0.0, 2.0] must be exactly 1.0 everywhere."""
        recon = _seeded_recon([0.0, 2.0])
        ctx = _ctx_stub(AxisManager(Axis("seed", 2)))
        recon_out, _ = _agg(self.agg, recon, _DEFAULT_TARGET, ctx, _make_stage({}))
        assert torch.allclose(recon_out, torch.ones(*INPUT_SHAPE), atol=1e-6)

    def test_mean_of_identical_seeds_equals_that_value(self):
        V, N = 0.75, 5
        recon = _seeded_recon([V] * N)
        ctx = _ctx_stub(AxisManager(Axis("seed", N)))
        recon_out, _ = _agg(self.agg, recon, _DEFAULT_TARGET, ctx, _make_stage({}))
        assert torch.allclose(recon_out, torch.full(INPUT_SHAPE, V), atol=1e-6)

    def test_non_seed_prefix_axis_is_not_collapsed(self):
        """Epoch axis at dim 0 untouched; seed at dim 1 collapsed."""
        E, S = 2, 3
        recon = torch.rand(E, S, *INPUT_SHAPE)
        ctx = _ctx_stub(AxisManager(Axis("epoch", E), Axis("seed", S)))
        recon_out, _ = _agg(self.agg, recon, _DEFAULT_TARGET, ctx, _make_stage({}))
        assert recon_out.shape == (E, *INPUT_SHAPE)

    def test_mean_matches_torch_mean_reference(self):
        N = 6
        recon = torch.rand(N, *INPUT_SHAPE)
        ctx = _ctx_stub(AxisManager(Axis("seed", N)))
        recon_out, _ = _agg(self.agg, recon, _DEFAULT_TARGET, ctx, _make_stage({}))
        assert torch.allclose(recon_out, recon.mean(dim=0), atol=1e-6)

    def test_delegates_target_reduction_to_target_source(self):
        """MeanAggregator must call target_source.reduce(target, seed_dim, 'mean')."""
        N = 3
        recording_ts = _RecordingTarget()
        stage = _make_stage({}, target_source=recording_ts)
        ctx = _ctx_stub(AxisManager(Axis("seed", N)))
        recon = _seeded_recon([0.5] * N)
        _agg(self.agg, recon, _DEFAULT_TARGET, ctx, stage)
        assert len(recording_ts.reduce_calls) == 1
        assert recording_ts.reduce_calls[0] == (0, "mean")

    def test_returned_target_is_what_reduce_returned(self):
        """The target in the output tuple must be exactly what target_source.reduce() returned."""
        N = 2
        recording_ts = _RecordingTarget()
        stage = _make_stage({}, target_source=recording_ts)
        ctx = _ctx_stub(AxisManager(Axis("seed", N)))
        recon = _seeded_recon([0.5, 0.5])
        _, target_out = _agg(self.agg, recon, _DEFAULT_TARGET, ctx, stage)
        assert target_out is _RecordingTarget.SENTINEL


def test_mean_aggregator_drops_artifacts_carrying_the_collapsed_axis() -> None:
    recon = _seeded_recon([0.0, 2.0])
    result = DriverResult(input=recon, target=_DEFAULT_TARGET, artifacts={"pixels": recon, "tag": "x"},
                          loss=1.5, iterations=3, slice_losses=torch.tensor([1.0, 2.0]))
    out = MeanAggregator().aggregate(result, _ctx_stub(AxisManager(Axis("seed", 2))), _make_stage({}))
    assert out.artifacts == {"tag": "x"}
    assert out.loss == 1.5 and out.iterations == 3 and out.slice_losses is None


# ── BestLossAggregator ────────────────────────────────────────────────────────

class TestBestLossAggregator:
    def setup_method(self):
        self.agg = BestLossAggregator()

    def test_no_seed_axis_returns_recon_unchanged(self):
        recon = torch.rand(INPUT_SHAPE)
        recon_out, _ = _agg(self.agg, recon, _DEFAULT_TARGET, _ctx_stub(), _make_stage({"l": (_SumLoss(), 1.0)}))
        assert recon_out is recon

    def test_no_seed_axis_returns_target_unchanged(self):
        target = _DEFAULT_TARGET
        _, target_out = _agg(self.agg, torch.rand(INPUT_SHAPE), target, _ctx_stub(), _make_stage({"l": (_SumLoss(), 1.0)}))
        assert target_out is target

    def test_selects_seed_with_minimum_loss(self):
        """Seed 1 (value=0.1) has the smallest sum loss and must be selected."""
        BEST = 1
        recon = _seeded_recon([0.5, 0.1, 0.8])
        ctx = _ctx_stub(AxisManager(Axis("seed", 3)))
        recon_out, _ = _agg(self.agg, recon, _DEFAULT_TARGET, ctx, _make_stage({"l": (_SumLoss(), 1.0)}), slice_losses=_sums(recon))
        assert torch.allclose(recon_out, recon[BEST], atol=1e-6)

    def test_selected_recon_has_correct_shape(self):
        recon = _seeded_recon([1.0, 0.2, 0.9])
        ctx = _ctx_stub(AxisManager(Axis("seed", 3)))
        recon_out, _ = _agg(self.agg, recon, _DEFAULT_TARGET, ctx, _make_stage({"l": (_SumLoss(), 1.0)}), slice_losses=_sums(recon))
        assert recon_out.shape == INPUT_SHAPE

    def test_first_seed_wins_when_all_losses_equal(self):
        """Stable selection: when all seeds tie, seed 0 (first minimum) is kept."""
        recon = _seeded_recon([1.0, 1.0, 1.0])
        ctx = _ctx_stub(AxisManager(Axis("seed", 3)))
        recon_out, _ = _agg(self.agg, recon, _DEFAULT_TARGET, ctx, _make_stage({"l": (_SumLoss(), 1.0)}), slice_losses=_sums(recon))
        assert torch.allclose(recon_out, recon[0], atol=1e-6)

    def test_last_seed_selected_when_best(self):
        BEST = 4
        recon = _seeded_recon([1.0, 0.9, 0.8, 0.7, 0.1])
        ctx = _ctx_stub(AxisManager(Axis("seed", 5)))
        recon_out, _ = _agg(self.agg, recon, _DEFAULT_TARGET, ctx, _make_stage({"l": (_SumLoss(), 1.0)}), slice_losses=_sums(recon))
        assert torch.allclose(recon_out, recon[BEST], atol=1e-6)

    def test_loss_weights_affect_selection(self):
        """Weighted sum determines winner even when individual losses disagree.

        loss_a per seed: [1.0, 100.0] — prefers seed 0
        loss_b per seed: [100.0, 1.0] — prefers seed 1
        weight_a=0.01, weight_b=1.0:
          seed 0: 0.01*1.0   + 1.0*100.0 = 100.01
          seed 1: 0.01*100.0 + 1.0*1.0   =   2.0  ← winner
        """
        loss_a = torch.tensor([1.0, 100.0])
        loss_b = torch.tensor([100.0, 1.0])
        recon = _seeded_recon([0.5, 0.5])
        ctx = _ctx_stub(AxisManager(Axis("seed", 2)))
        stage = _make_stage({})
        recon_out, _ = _agg(self.agg, recon, _DEFAULT_TARGET, ctx, stage, slice_losses=0.01 * loss_a + 1.0 * loss_b)
        assert torch.allclose(recon_out, recon[1], atol=1e-6)

    def test_fixed_target_returned_unchanged_in_winner_tuple(self):
        """FixedTarget's value [N] has no seed prefix → slice() returns it unchanged."""
        fixed_val = torch.tensor([7])
        target = TargetInfo(value=fixed_val)
        recon = _seeded_recon([0.5, 0.1])  # seed 1 wins
        ctx = _ctx_stub(AxisManager(Axis("seed", 2)))
        _, target_out = _agg(self.agg, recon, target, ctx, _make_stage({"l": (_SumLoss(), 1.0)}), slice_losses=_sums(recon))
        assert torch.equal(target_out.value, fixed_val)

    def test_fixed_target_returned_unchanged_regardless_of_winner(self):
        """FixedTarget.slice() returns the target unchanged — no seed prefix.

        Seed-sliced target selection requires JointTarget (which carries a seed prefix).
        FixedTarget is documented as 'fixed targets never carry axis prefixes'.
        """
        S = 3
        recon = _seeded_recon([0.5, 0.1, 0.8])
        fixed_value = torch.tensor([42])
        target = TargetInfo(value=fixed_value)
        ctx = _ctx_stub(AxisManager(Axis("seed", S)))
        _, target_out = _agg(self.agg, recon, target, ctx, _make_stage({"l": (_SumLoss(), 1.0)}), slice_losses=_sums(recon))
        assert torch.equal(target_out.value, fixed_value)

    def test_without_slice_losses_on_an_axis_raises(self):
        import pytest

        ctx = _ctx_stub(AxisManager(Axis("seed", 2)))
        with pytest.raises(ValueError, match="slice_losses"):
            _agg(self.agg, _seeded_recon([0.5, 0.1]), _DEFAULT_TARGET, ctx, _make_stage({}))

    def test_selects_artifacts_and_loss_of_the_winning_slice(self):
        recon = _seeded_recon([0.5, 0.1, 0.8])
        latent = torch.arange(3.0).view(3, 1, 1)
        result = DriverResult(input=recon, target=_DEFAULT_TARGET, artifacts={"latent": latent, "tag": "x"},
                              loss=0.0, iterations=4, slice_losses=_sums(recon))
        out = self.agg.aggregate(result, _ctx_stub(AxisManager(Axis("seed", 3))), _make_stage({}))
        assert torch.equal(out.artifacts["latent"], latent[1])
        assert out.artifacts["tag"] == "x"
        assert out.loss == _sums(recon)[1].item()
        assert out.slice_losses is None and out.iterations == 4

    def test_artifact_without_the_axis_prefix_raises_naming_it(self):
        import pytest

        recon = _seeded_recon([0.5, 0.1])
        result = DriverResult(input=recon, target=_DEFAULT_TARGET, artifacts={"latent": torch.zeros(5, 1)},
                              loss=0.0, iterations=0, slice_losses=_sums(recon))
        with pytest.raises(ValueError, match="latent"):
            self.agg.aggregate(result, _ctx_stub(AxisManager(Axis("seed", 2))), _make_stage({}))

    def test_aggregator_integrated_into_stage_run(self):
        """End-to-end: BestLossAggregator wired into OptimStage collapses seed dim in state."""
        import torch.nn as nn
        from leakpro.attacks.gia_attacks.modular.core.state import Context
        from leakpro.attacks.gia_attacks.modular.components.modalities.image import ImageNormalizationModality
        from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
        from leakpro.attacks.gia_attacks.modular.core.task import Task

        N_SEEDS = 3
        DEVICE = torch.device("cpu")
        MEAN = [0.5, 0.5, 0.5]
        STD  = [0.5, 0.5, 0.5]

        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 10)).eval()
        task = Task(CrossEntropy())
        imgs = torch.rand(INPUT_SHAPE)
        obs_raw = task.victim_signal(
            model, imgs, TargetInfo(value=torch.tensor([0]))
        )
        obs = ClientObservation(gradients=[g.detach().clone() for g in obs_raw.gradients])

        ctx = Context(
            model=model,
            observation=obs,
            modality=ImageNormalizationModality(mean=MEAN, std=STD, device=DEVICE),
            task=task,
            input_shape=INPUT_SHAPE,
            device=DEVICE,
            axes=AxisManager(Axis("seed", N_SEEDS)),
        )

        stage = OptimStage(
            losses={"l": (_SumLoss(), 1.0)},
            rep=PixelRepresentation(min_val=0.0, max_val=1.0),
            driver=GradientDriver(lr=0.0, max_iter=1, return_best=False, schedule=None),
            target_source=FixedTarget(),
            aggregator=BestLossAggregator(),
        )
        out = stage.run(State(target=TargetInfo(value=torch.tensor([0]))), ctx)

        assert out.input.shape == INPUT_SHAPE
        assert out.target.value.shape == torch.tensor([0]).shape



def test_chain_aggregator_rejects_a_member_without_collapsed_axes_at_construction():
    import pytest

    from leakpro.attacks.gia_attacks.modular.components.aggregators import ChainAggregator
    from leakpro.attacks.gia_attacks.modular.core.interfaces import Aggregator

    class _Opaque(Aggregator):
        def aggregate(self, result, ctx, stage):  # noqa: ANN001
            return result

    with pytest.raises(NotImplementedError, match="_Opaque"):
        ChainAggregator([BestLossAggregator(), _Opaque()])
