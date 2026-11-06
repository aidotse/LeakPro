#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Every concrete implementation of a framework interface, with a download-free factory for each.

``IMPLEMENTATIONS[interface][implementation]`` is a zero-argument factory returning a *case*: the
implementation instance (``case.component``) plus whatever the interface's contract needs to exercise
it (a context recipe, a state, the expected metric values, ...). ``test_coverage.py`` fails when a
shipped implementation has no entry here, so a new component is contract-tested from its first commit.

This module holds the framework's generic components. Each feature area (generative priors, object
detection, tabular) adds its own in ``registry_<area>.py``, which this module loads at the end.

Implementations behind an optional dependency (or too slow for the default run) stay registered;
``MARKS`` attaches the skip or slow mark inside the parametrisation (``cases``), so coverage still
counts them.
"""
from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from leakpro.attacks.gia_attacks.modular.components.aggregators import (
    BestLossAggregator,
    ChainAggregator,
    EpochMatchingAggregator,
    MeanAggregator,
)
from leakpro.attacks.gia_attacks.modular.components.drivers.grad_transforms import (
    ClipGradNorm,
    GaussianNoise,
    SignGradient,
)
from leakpro.attacks.gia_attacks.modular.components.drivers.gradient import GradientDriver
from leakpro.attacks.gia_attacks.modular.components.drivers.schedules import Cosine, Exponential, LinearDecay, MultiStep
from leakpro.attacks.gia_attacks.modular.components.evaluators import CompositeEvaluator, ImageEvaluator, LabelEvaluator
from leakpro.attacks.gia_attacks.modular.components.initializers import FromState, FromTensor, Gaussian, Uniform
from leakpro.attacks.gia_attacks.modular.components.losses import (
    BNStatsLoss,
    CosineGradientLoss,
    EpochOrderInvariantPrior,
    GroupConsistencyLoss,
    L2GradientLoss,
    L2ImageNormLoss,
    TVLoss,
)
from leakpro.attacks.gia_attacks.modular.components.modalities.image import ImageNormalizationModality
from leakpro.attacks.gia_attacks.modular.components.objectives import BinaryCrossEntropy, CrossEntropy, MeanSquaredError
from leakpro.attacks.gia_attacks.modular.components.representations.fixed import FixedRepresentation
from leakpro.attacks.gia_attacks.modular.components.representations.pixel import PixelRepresentation
from leakpro.attacks.gia_attacks.modular.components.samplers import RandomSampler, SequentialSampler
from leakpro.attacks.gia_attacks.modular.components.target_inference import (
    DimitrovInference,
    GengInference,
    IDLGInference,
    ZeroLabelInference,
)
from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget, JointTarget, NullTarget
from leakpro.attacks.gia_attacks.modular.core.axes import AxisManager
from leakpro.attacks.gia_attacks.modular.core.interfaces import (
    Aggregator,
    BatchSampler,
    ClientProtocol,
    Driver,
    Evaluator,
    GradTransform,
    Initializer,
    LossTerm,
    Modality,
    Objective,
    Representation,
    Schedule,
    TargetInference,
    TargetSource,
)
from leakpro.attacks.gia_attacks.modular.core.meta import MetaAdam, MetaOptimizer, MetaSGD
from leakpro.attacks.gia_attacks.modular.core.protocol import FedAvg, FedSGD
from leakpro.attacks.gia_attacks.modular.core.state import State, TargetInfo
from leakpro.tests.gia import helpers
from leakpro.tests.gia.contract.conftest import epoch_axes, seed_axes


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _needs(module: str) -> pytest.MarkDecorator:
    return pytest.mark.skipif(not _has(module), reason=f"requires {module}")


# ── Shared tiny inputs ─────────────────────────────────────────────────────────────────────────────

# Three rows of five features: two continuous columns, then one one-hot group of three.
TABULAR_SHAPE = (3, 5)


def _image_shape() -> tuple[int, ...]:
    return (2, *helpers.IMAGE_SHAPE)


def _tabular_rows() -> Tensor:
    gen = torch.Generator().manual_seed(3)
    rows = torch.randn(TABULAR_SHAPE, generator=gen)
    rows[:, 2:5] = nn.functional.one_hot(torch.tensor([0, 2, 1]), 3).float()
    return rows


# ── Representation ─────────────────────────────────────────────────────────────────────────────────


@dataclass
class RepresentationCase:
    """A representation and how to bind it.

    Attributes:
        component: The configured representation.
        ctx_kwargs: ``make_ctx`` keywords (modality, input shape) for its domain.
        state: Builds the state it binds from, given the bound context (its axis prefix).
        image: Decodes to [0, 1] images (an observable clamped to the box).
        bounded: ``decode()`` stays in [0, 1] after ``project()`` from any parameter value.
        resumes_from_artifacts: Binding on ``State(artifacts=rep.artifacts())`` resumes the iterate.
            False for a representation documented to bind from ``state.input`` only.

    """

    component: Representation
    ctx_kwargs: dict[str, Any] = field(default_factory=dict)
    state: Callable[[Any], State] = lambda ctx: State()  # noqa: ARG005
    image: bool = True
    bounded: bool = True
    resumes_from_artifacts: bool = True


def _input_state(ctx: Any) -> State:  # noqa: ANN401
    return State(input=torch.rand(*ctx.axes.shape, *ctx.input_shape))


REPRESENTATIONS: dict[type, Callable[[], RepresentationCase]] = {
    PixelRepresentation: lambda: RepresentationCase(PixelRepresentation()),
    FixedRepresentation: lambda: RepresentationCase(FixedRepresentation(), state=_input_state,
                                                    resumes_from_artifacts=False),
}


# ── TargetSource ───────────────────────────────────────────────────────────────────────────────────


@dataclass
class TargetSourceCase:
    """A target source, the state it binds from, and whether it supports an axis prefix."""

    component: TargetSource
    state: Callable[[], State] = lambda: State(target=TargetInfo(value=helpers.client_batch()[1]))
    ctx_kwargs: dict[str, Any] = field(default_factory=dict)
    supports_axes: bool = True


TARGET_SOURCES: dict[type, Callable[[], TargetSourceCase]] = {
    FixedTarget: lambda: TargetSourceCase(FixedTarget()),
    JointTarget: lambda: TargetSourceCase(JointTarget(num_classes=helpers.NUM_CLASSES)),
    NullTarget: lambda: TargetSourceCase(NullTarget()),
}


# ── LossTerm ───────────────────────────────────────────────────────────────────────────────────────


@dataclass
class LossTermCase:
    """A loss term and the context it is evaluated in.

    Attributes:
        component: The configured term.
        ctx_kwargs: ``make_ctx`` keywords; the observation is the standard client batch's.
        matches_gradients: The term is a gradient-matching distance (≈0 at the client batch).

    """

    component: LossTerm
    ctx_kwargs: dict[str, Any] = field(default_factory=dict)
    matches_gradients: bool = False


LOSS_TERMS: dict[type, Callable[[], LossTermCase]] = {
    CosineGradientLoss: lambda: LossTermCase(CosineGradientLoss(), matches_gradients=True),
    L2GradientLoss: lambda: LossTermCase(L2GradientLoss(), matches_gradients=True),
    TVLoss: lambda: LossTermCase(TVLoss()),
    BNStatsLoss: lambda: LossTermCase(BNStatsLoss()),
    GroupConsistencyLoss: lambda: LossTermCase(GroupConsistencyLoss(), ctx_kwargs={"axes": seed_axes()}),
    L2ImageNormLoss: lambda: LossTermCase(L2ImageNormLoss()),
    EpochOrderInvariantPrior: lambda: LossTermCase(
        EpochOrderInvariantPrior(),
        ctx_kwargs={"axes": epoch_axes(), "protocol": FedAvg(local_epochs=2, lr=0.1)}),
}


# ── Driver ─────────────────────────────────────────────────────────────────────────────────────────


@dataclass
class DriverCase:
    """A driver, the stage components it runs, and its configured budget.

    Attributes:
        component: The driver.
        rep: The representation of the stage.
        losses: The stage's ``{name: (term, weight)}``.
        budget: The ``iterations`` the result must report.
        axes: The axis prefix of the run.
        nan_loss: The driver has no objective value (reverse diffusion) and reports NaN.

    """

    component: Driver
    rep: Representation
    losses: dict[str, tuple[LossTerm, float]]
    budget: int
    axes: AxisManager = field(default_factory=AxisManager)
    nan_loss: bool = False


DRIVERS: dict[type, Callable[[], DriverCase]] = {
    GradientDriver: lambda: DriverCase(GradientDriver(lr=0.05, max_iter=3), PixelRepresentation(),
                                       {"grad": (CosineGradientLoss(), 1.0), "tv": (TVLoss(), 0.1)},
                                       budget=3, axes=seed_axes()),
}


# ── Aggregator ─────────────────────────────────────────────────────────────────────────────────────


@dataclass
class AggregatorCase:
    """An aggregator and the axis it collapses."""

    component: Aggregator
    axes: AxisManager = field(default_factory=seed_axes)


AGGREGATORS: dict[type, Callable[[], AggregatorCase]] = {
    MeanAggregator: lambda: AggregatorCase(MeanAggregator()),
    BestLossAggregator: lambda: AggregatorCase(BestLossAggregator()),
    ChainAggregator: lambda: AggregatorCase(ChainAggregator([BestLossAggregator()])),
    EpochMatchingAggregator: lambda: AggregatorCase(EpochMatchingAggregator(), axes=epoch_axes()),
}


# ── Objective ──────────────────────────────────────────────────────────────────────────────────────


@dataclass
class ObjectiveCase:
    """An objective with a model, a model input and a target it scores."""

    component: Objective
    model: nn.Module
    model_input: Tensor
    target: TargetInfo


def _scalar_head() -> nn.Module:
    torch.manual_seed(0)
    return nn.Linear(TABULAR_SHAPE[1], 1)


OBJECTIVES: dict[type, Callable[[], ObjectiveCase]] = {
    CrossEntropy: lambda: ObjectiveCase(
        CrossEntropy(), helpers.tiny_bn_cnn().train(),
        helpers.image_modality().to_model_input(helpers.client_batch()[0]), TargetInfo(value=helpers.client_batch()[1])),
    BinaryCrossEntropy: lambda: ObjectiveCase(
        BinaryCrossEntropy(), _scalar_head(), _tabular_rows(), TargetInfo(value=torch.tensor([0, 1, 1]))),
    MeanSquaredError: lambda: ObjectiveCase(
        MeanSquaredError(), _scalar_head(), _tabular_rows(), TargetInfo(value=torch.tensor([0.5, -1.0, 2.0]))),
}


# ── ClientProtocol ─────────────────────────────────────────────────────────────────────────────────


@dataclass
class ProtocolCase:
    """A protocol observed on the standard classifier and client batch."""

    component: ClientProtocol
    objective: Objective = field(default_factory=CrossEntropy)


PROTOCOLS: dict[type, Callable[[], ProtocolCase]] = {
    FedSGD: lambda: ProtocolCase(FedSGD()),
    FedAvg: lambda: ProtocolCase(FedAvg(local_epochs=2, batch_size=1, lr=0.1)),
}


# ── TargetInference ────────────────────────────────────────────────────────────────────────────────


@dataclass
class TargetInferenceCase:
    """A target inference and the client protocol the observation was produced under."""

    component: TargetInference
    protocol: ClientProtocol = field(default_factory=FedSGD)


TARGET_INFERENCES: dict[type, Callable[[], TargetInferenceCase]] = {
    IDLGInference: lambda: TargetInferenceCase(IDLGInference()),
    DimitrovInference: lambda: TargetInferenceCase(
        DimitrovInference(epochs=2, batch_size=1, learning_rate=0.1, num_dummy_samples=8),
        protocol=FedAvg(local_epochs=2, batch_size=1, lr=0.1)),
    GengInference: lambda: TargetInferenceCase(GengInference(num_dummy_samples=8)),
    ZeroLabelInference: lambda: TargetInferenceCase(ZeroLabelInference()),
}


# ── Modality ───────────────────────────────────────────────────────────────────────────────────────


@dataclass
class ModalityCase:
    """A modality and a data-space sample it maps."""

    component: Modality
    data: Tensor


MODALITIES: dict[type, Callable[[], ModalityCase]] = {
    ImageNormalizationModality: lambda: ModalityCase(helpers.image_modality(), helpers.client_batch()[0]),
}


# ── Evaluator ──────────────────────────────────────────────────────────────────────────────────────


@dataclass
class EvaluatorCase:
    """An evaluator scoring a reconstruction equal to its ground truth.

    Attributes:
        expected: Metric name to the value it must have at ground truth (``pytest.approx``), or to
            a predicate on that value.

    """

    component: Evaluator
    recon: Tensor
    target: Tensor
    state: State | None = None
    expected: dict[str, Any] = field(default_factory=dict)


def _psnr_at_identity(value: float) -> bool:
    return value == float("inf") or value > 60.0


def _label_state() -> State:
    labels = helpers.client_batch()[1]
    return State(target=TargetInfo(value=labels), artifacts={LabelEvaluator.GT_KEY: labels})


_LABEL_EXPECTED = {"accuracy": 1.0, "multiset_accuracy": 1.0}

EVALUATORS: dict[type, Callable[[], EvaluatorCase]] = {
    ImageEvaluator: lambda: EvaluatorCase(ImageEvaluator(), helpers.client_batch()[0], helpers.client_batch()[0],
                                          expected={"psnr": _psnr_at_identity, "ssim": 1.0}),
    LabelEvaluator: lambda: EvaluatorCase(LabelEvaluator(), helpers.client_batch()[0], helpers.client_batch()[0],
                                          state=_label_state(), expected=_LABEL_EXPECTED),
    # registry_detection.py replaces this case with an image + boxes composite.
    CompositeEvaluator: lambda: EvaluatorCase(
        CompositeEvaluator({"image": ImageEvaluator(), "labels": LabelEvaluator()}), helpers.client_batch()[0],
        helpers.client_batch()[0], state=_label_state(),
        expected={"image.psnr": _psnr_at_identity, "image.ssim": 1.0,
                  **{f"labels.{k}": v for k, v in _LABEL_EXPECTED.items()}}),
}


# ── Small components ───────────────────────────────────────────────────────────────────────────────


@dataclass
class InitializerCase:
    """An initializer; ``draws`` is False for one documented to take its start from the state.

    ``seeded`` builds the initializer with a given seed; None for one that draws no random numbers.
    """

    component: Initializer
    draws: bool = True
    seeded: Callable[[int], Initializer] | None = None


INIT_SHAPE = (2, 3, 4, 4)

INITIALIZERS: dict[type, Callable[[], InitializerCase]] = {
    Uniform: lambda: InitializerCase(Uniform(seed=1), seeded=lambda seed: Uniform(seed=seed)),
    Gaussian: lambda: InitializerCase(Gaussian(), seeded=lambda seed: Gaussian(seed=seed)),
    FromTensor: lambda: InitializerCase(FromTensor(torch.rand(INIT_SHAPE))),
    FromState: lambda: InitializerCase(FromState(), draws=False),
}


@dataclass
class SimpleCase:
    """A component whose contract needs nothing but the component itself."""

    component: Any


SCHEDULES: dict[type, Callable[[], SimpleCase]] = {
    Cosine: lambda: SimpleCase(Cosine()),
    MultiStep: lambda: SimpleCase(MultiStep()),
    Exponential: lambda: SimpleCase(Exponential()),
    LinearDecay: lambda: SimpleCase(LinearDecay()),
}

GRAD_TRANSFORMS: dict[type, Callable[[], SimpleCase]] = {
    SignGradient: lambda: SimpleCase(SignGradient()),
    GaussianNoise: lambda: SimpleCase(GaussianNoise(0.1, seed=0)),
    ClipGradNorm: lambda: SimpleCase(ClipGradNorm(1.0)),
}

SAMPLERS: dict[type, Callable[[], SimpleCase]] = {
    SequentialSampler: lambda: SimpleCase(SequentialSampler()),
    RandomSampler: lambda: SimpleCase(RandomSampler()),
}

META_OPTIMIZERS: dict[type, Callable[[], SimpleCase]] = {
    MetaSGD: lambda: SimpleCase(MetaSGD(lr=0.1)),
    MetaAdam: lambda: SimpleCase(MetaAdam(lr=0.1)),
}


IMPLEMENTATIONS: dict[type, dict[type, Callable[[], Any]]] = {
    Representation: REPRESENTATIONS,
    TargetSource: TARGET_SOURCES,
    LossTerm: LOSS_TERMS,
    Driver: DRIVERS,
    Aggregator: AGGREGATORS,
    Objective: OBJECTIVES,
    ClientProtocol: PROTOCOLS,
    TargetInference: TARGET_INFERENCES,
    Modality: MODALITIES,
    Evaluator: EVALUATORS,
    BatchSampler: SAMPLERS,
    Initializer: INITIALIZERS,
    Schedule: SCHEDULES,
    GradTransform: GRAD_TRANSFORMS,
    MetaOptimizer: META_OPTIMIZERS,
}

MARKS: dict[type, tuple[pytest.MarkDecorator, ...]] = {}


def cases(interface: type) -> list[Any]:
    """``pytest.param(implementation, factory)`` for every registered implementation of ``interface``."""
    return [pytest.param(cls, factory, id=cls.__name__, marks=MARKS.get(cls, ()))
            for cls, factory in IMPLEMENTATIONS[interface].items()]


def build(cls: type, factory: Callable[[], Any]) -> Any:  # noqa: ANN401
    """Call ``factory`` and check it built the implementation it is registered for."""
    case = factory()
    assert type(case.component) is cls, f"registry factory for {cls.__name__} built {type(case.component).__name__}"
    return case


__all__ = ["IMPLEMENTATIONS", "MARKS", "build", "cases"]

# Load the feature areas present: each ``registry_<area>.py`` adds its cases (and marks) to the tables above.
for _area in sorted(Path(__file__).parent.glob("registry_*.py")):
    importlib.import_module(f"{__package__}.{_area.stem}")
