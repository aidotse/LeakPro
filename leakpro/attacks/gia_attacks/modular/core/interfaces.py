#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Abstract interfaces for the building blocks of an attack."""

from __future__ import annotations

import abc
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, runtime_checkable

import torch
from torch import Tensor

from leakpro.attacks.gia_attacks.modular.core.evaluation import EvalResult
from leakpro.attacks.gia_attacks.modular.core.loss_inputs import LossInputs, LossScope

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.events import StepEvent
    from leakpro.attacks.gia_attacks.modular.core.protocol import ProtocolOutput
    from leakpro.attacks.gia_attacks.modular.core.requirements import Requirement
    from leakpro.attacks.gia_attacks.modular.core.result import DriverResult
    from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
    from leakpro.attacks.gia_attacks.modular.core.state import Context, State, TargetInfo

Space = Literal["data", "model_input"]
"""Where a tensor lives: the data space x, or the model-input space z the model consumes (e.g. normalized)."""

SPACES: tuple[Space, ...] = ("data", "model_input")

ModelMode = Literal["train", "eval"]
"""The module mode the client trains in (``nn.Module.train(mode == "train")``)."""

MODEL_MODES: tuple[ModelMode, ...] = ("train", "eval")

ModelFn = Callable[..., Any]
"""A model call: the module itself (FedSGD) or a functional call at the current local weights (FedAvg)."""


class BatchSampler(abc.ABC):
    """Yields 1D index tensors covering one epoch worth of minibatches."""

    @abc.abstractmethod
    def yield_batches(
        self, num_samples: int, batch_size: int, epoch: int, device: torch.device
    ) -> Iterator[Tensor]:
        """Yields 1D tensors containing the indices for each minibatch."""


@runtime_checkable
class Optimizable(Protocol):
    """Something that exposes optimizer parameter groups."""

    def parameters(self) -> list[dict]:
        """Parameter groups ``{"name", "params": [Tensor], "lr_scale": float}``."""
        ...


class Representation(abc.ABC):
    """Maps between parameter space (what the optimizer tweaks) and data space."""

    @abc.abstractmethod
    def bind(self, state: State, ctx: Context) -> Representation:
        """Return a bound copy holding this run's parameter leaves; ``self`` is left unchanged.

        The leaves come from state.artifacts (resume), state.input (transition), or from scratch
        when both are absent, with requires_grad set. Keeping run state on the copy lets one
        configuration object be run any number of times with identical results.
        """

    @abc.abstractmethod
    def parameters(self) -> list[dict]:
        """Named param-groups: [{"name": str, "params": [Tensor], "lr_scale": float}]."""

    @abc.abstractmethod
    def decode(self) -> Tensor:
        """Return the reconstruction in **data space**, shaped `[*ctx.axes.shape, *ctx.input_shape]`.

        Data space is the space `Modality.to_model_input` consumes ([0, 1] for images).

        Identity for data-space pixels; G(z) mapped from the generator's output range for a GAN.
        """

    def model_input(self, modality: Modality) -> Tensor:
        """The tensor the model consumes for the current iterate.

        The default maps ``decode()`` through the modality. A representation whose parameter already
        lives in model-input space overrides this to return that parameter itself: a float round trip
        through data space changes the iterate enough to break bit-level agreement with optimizers
        that step in model-input space.
        """
        return modality.to_model_input(self.decode())

    @abc.abstractmethod
    def encode(self, data: Tensor) -> Tensor:
        """Data space → param space (used for cross-rep transitions)."""

    @abc.abstractmethod
    def project(self, ctx: Context | None = None) -> None:
        """In-place projected gradient step on owned parameters."""

    @abc.abstractmethod
    def to_observable(self) -> Tensor:
        """Final detached data-space output."""

    @abc.abstractmethod
    def artifacts(self) -> dict[str, Any]:
        """State to carry across stages for resume / transition."""


class Modality(abc.ABC):
    """Domain-specific mapping between data space and model input space.

    Deliberately *only* a transform.  Scoring a reconstruction is a separate
    concern with a separate axis of variation — the same pixel transform is
    scored by PSNR/SSIM for a natural image and by mAP for a detection task,
    and the same tabular transform is scored by tolerance matching or by
    interval coverage depending on which attack produced it.  Evaluation
    therefore lives behind ``Evaluator``.
    """

    @abc.abstractmethod
    def to_model_input(self, data: Tensor) -> Tensor:
        """Data space → model input (e.g. normalize [0,1] pixels)."""

    @abc.abstractmethod
    def strip_model_input(self, model_input: Tensor) -> Tensor:
        """Model input → data space (inverse of to_model_input)."""

    def client_to_model_input(self, data: Tensor) -> Tensor:
        """The model input of the client's real data, as ``simulate_client`` feeds it to the victim model.

        The default is ``to_model_input``. A modality whose data space relaxes a discrete encoding for the
        attacker (e.g. categorical logits) overrides it, because the client holds the
        discrete values themselves and feeds them to the model as they are.
        """
        return self.to_model_input(data)


class Evaluator(abc.ABC):
    """Scores a reconstruction against ground truth.

    Split out of ``Modality`` so transform and scoring vary independently.
    ``ctx``/``state`` are optional so an evaluator can reach for run context
    (axes, schema) or stage by-products (e.g. interval bounds) when
    it needs them, without every evaluator being forced to accept them.
    """

    @abc.abstractmethod
    def evaluate(
        self,
        recon: Tensor,
        target: Tensor,
        *,
        ctx: Context | None = None,
        state: State | None = None,
    ) -> EvalResult:
        """Return scalar metrics plus non-scalar extras for one reconstruction."""


class Objective(abc.ABC):
    """How a model consumes ``(model_input, target)`` and yields a scalar training loss.

    The objective owns the model call (a detector may need the target at forward time) and the
    loss; the client protocol owns how often and at which weights it is evaluated.
    ``supported_protocols`` names the protocol types whose model call this objective supports.
    """

    supported_protocols: ClassVar[tuple[type[ClientProtocol], ...]]

    def prepare_model(self, model: torch.nn.Module, mode: ModelMode) -> None:
        """Put ``model`` in the module mode the client trains in."""
        model.train(mode == "train")

    @abc.abstractmethod
    def loss(self, model_fn: ModelFn, model_input: Tensor, target: TargetInfo) -> Tensor:
        """Scalar training loss of ``model_fn`` on ``(model_input, target)``."""


class ClientProtocol(abc.ABC):
    """How a client turns its data into the payload the server observes (a gradient or a weight update)."""

    @abc.abstractmethod
    def observe(
        self,
        model: torch.nn.Module,
        objective: Objective,
        model_input: Tensor,
        target: TargetInfo,
        *,
        create_graph: bool,
        has_epoch_prefix: bool,
        base_params: Mapping[str, Tensor] | None = None,
        base_buffers: Mapping[str, Tensor] | None = None,
    ) -> ProtocolOutput:
        """Simulate the client's local computation on ``model``.

        Args:
            model: Module whose forward the client runs (in the mode the caller set).
            objective: The training loss.
            model_input: The client batch in model-input space, ``[E, N, ...]`` when ``has_epoch_prefix``.
            target: Supervision matching ``model_input``.
            create_graph: Keep the graph so the payload is differentiable w.r.t. ``model_input``.
            has_epoch_prefix: Whether ``model_input`` carries a leading epoch axis.
            base_params: Starting weights; default ``model.named_parameters()``.
            base_buffers: Starting buffers; default ``model.named_buffers()``. Buffers the forward updates
                (BN running statistics) start from these values and evolve over the client's local steps.

        """

    def local_steps(self, num_samples: int) -> int | None:  # noqa: ARG002 -- interface signature
        """Number of local optimization steps (forward passes that update the buffers) on ``num_samples`` samples.

        Components that invert a buffer update (``BNStatsLoss(strategy="inferred")``) need it. The default,
        ``None``, means the protocol does not say; such components then refuse to run.
        """
        return None


class TargetSource(abc.ABC):
    """Provides the supervision signal each iteration; may expose optimizable parameters."""

    def bind(self, state: State, ctx: Context) -> TargetSource:  # noqa: ARG002 -- interface signature
        """Return a bound copy initialized from runtime context (device, batch size); ``self`` is unchanged.

        The default returns ``self``, for sources that hold no run state.
        """
        return self

    @abc.abstractmethod
    def current(self) -> TargetInfo:
        """Returns the TargetInfo used in the loss this step."""

    def parameters(self) -> list[dict]:
        """Param-groups for jointly-optimized targets (empty for fixed targets)."""
        return []

    def reduce(self, target: TargetInfo, dim: int, method: str) -> TargetInfo:  # noqa: ARG002 -- interface signature
        """Collapse a parallel-axis target tensor to canonical shape after aggregation.

        The base implementation is a no-op for targets that carry no axis prefix
        (``target.axis_ndim == 0``) — those are shared across the axis and need no
        collapsing.  A target that *does* span the axis cannot be collapsed
        generically (the correct reduction is source-specific: averaging soft
        labels, re-deriving argmax, matching epoch 0, …), so this raises rather
        than silently returning a target whose shape no longer matches the
        aggregated reconstruction.

        Override in subclasses that carry axis-prefixed tensors (e.g. JointTarget).
        """
        if target is not None and target.axis_ndim > 0:
            raise NotImplementedError(
                f"{self.__class__.__name__}.reduce() received a target carrying "
                f"{target.axis_ndim} axis dimension(s) but does not implement a "
                f"reduction for method '{method}'."
            )
        return target

    @abc.abstractmethod
    def slice(self, target: TargetInfo, dim: int, indices: int | Tensor) -> TargetInfo:
        """Extract a target slice along the given axis dimension.

        indices may be:
          int    — scalar seed/epoch index; the selected dimension is removed (.select).
          Tensor — 1D index tensor for minibatch selection; the dimension is kept.

        FixedTarget returns the target unchanged (no axis prefix is ever added).
        JointTarget dispatches on the type of indices.
        """


class LossTerm(abc.ABC):
    """Single, unweighted loss contribution. Weight lives in the stage config.

    Subclasses declare their evaluation scope via the ``scope`` ClassVar:

    - ``LossScope.PER_SLICE`` (default): evaluated once per seed slice inside the
      driver's per-seed loop.  ``inputs.dummy_obs`` is a real ``ClientObservation``
      from ``simulate_dummy_signal``.  The activation graph is freed per slice to
      keep VRAM bounded.

    - ``LossScope.GROUP``: evaluated once on the full live reconstruction after the
      per-slice loop.  ``inputs.dummy_obs`` is ``None`` (no model forward — data-space
      only).  A single backward call allows gradients to flow across seeds.

    A term that reads an observation field (e.g. ``observation.bn_stats``, which the threat model
    may strip) or a run input declares it in ``requirements()``; ``run_attack`` checks every
    declaration before the first stage runs.
    """

    scope: ClassVar[LossScope] = LossScope.PER_SLICE

    def requirements(self) -> frozenset[Requirement]:
        """The observation fields and run inputs this term reads. Default: none."""
        return frozenset()

    def bind(self, ctx: Context) -> LossTerm:  # noqa: ARG002 -- interface signature
        """The term to evaluate in this stage, given the run context.

        A term that needs run resources (e.g. hooks on ``ctx.session.execution_model``) returns a
        copy holding them, so the configured term never carries run state. The default returns
        ``self``.
        """
        return self

    def release(self) -> None:  # noqa: B027 -- optional hook, no-op by default
        """Free the resources a bound copy holds; called once after the stage's loop. Default: no-op."""

    @abc.abstractmethod
    def compute(self, inputs: LossInputs) -> Tensor:
        """The term's unweighted scalar value on ``inputs``; the stage multiplies it by the term's weight."""


class Driver(abc.ABC):
    """Owns the iteration loop for a stage (Gradient / Diffusion / CMA-ES)."""

    @abc.abstractmethod
    def run(
        self,
        stage: OptimStage,
        ctx: Context,
        *,
        callbacks: Sequence[Callback] = (),
        stage_idx: int = 0,
    ) -> DriverResult:
        """Run the loop on the bound ``stage`` (its ``rep``, ``target_source`` and ``losses``).

        The returned input, target and artifacts come from the same iterate(s).
        """


class TargetInference(abc.ABC):
    """Analytically or statistically infers the target (labels) from the intercepted gradients."""

    def requirements(self) -> frozenset[Requirement]:
        """The observation fields and run inputs this inference reads. Default: none."""
        return frozenset()

    @abc.abstractmethod
    def infer(self, ctx: Context) -> TargetInfo:
        """Infers the target information (for example labels) from the observation."""


class Aggregator(abc.ABC):
    """Collapses a multi-seed (or other parallel-axis) driver result to canonical shape.

    Called once after driver.run() completes if OptimStage.aggregator is not None. Receives the
    driver's ``DriverResult`` (input shaped [*axes.shape, N, C, H, W], a target that may carry
    axis prefixes for jointly-optimized targets, artifacts with the axis prefix) and returns the
    collapsed result.
    """

    def collapsed_axes(self) -> tuple[str, ...]:
        """Names of the axes this aggregator removes; a ``ChainAggregator`` needs them to track the remaining axes."""
        raise NotImplementedError(f"{type(self).__name__} does not declare the axes it collapses, so it cannot be chained.")

    @abc.abstractmethod
    def aggregate(
        self,
        result: DriverResult,
        ctx: Context,
        stage: OptimStage,
    ) -> DriverResult:
        """Collapses the parallel axes of the driver result."""


class Initializer(abc.ABC):
    """Draws the starting iterate of a representation.

    ``space`` names the space the samples live in, so a representation optimizing in another space
    can map the draw before using it.
    """

    space: Space

    @abc.abstractmethod
    def sample(self, shape: tuple[int, ...], device: torch.device) -> Tensor:
        """A fresh starting tensor of ``shape`` on ``device``."""


class Schedule(abc.ABC):
    """Builds the learning-rate scheduler of one driver run from that run's optimizer and length."""

    @abc.abstractmethod
    def build(self, optimizer: torch.optim.Optimizer, max_iter: int) -> torch.optim.lr_scheduler.LRScheduler:
        """A scheduler over ``optimizer`` for a run of ``max_iter`` steps, stepped once per step."""


class GradTransform(abc.ABC):
    """Rewrites the gradients after the backward pass and before the optimizer uses them."""

    def bind(self, device: torch.device) -> GradTransform:  # noqa: ARG002 -- interface signature
        """The transform to use for one run on ``device``; stateful transforms return a fresh per-run instance."""
        return self

    @abc.abstractmethod
    def __call__(self, params: Sequence[Tensor]) -> None:
        """Modify ``p.grad`` in place for every parameter that has one."""


class Callback:
    """Base class for optimization loop hooks. Override only the methods you need."""

    def on_stage_start(self, stage_idx: int, state: State, ctx: Context) -> None:
        """Called when a stage begins."""
        pass

    def on_step_end(self, event: StepEvent) -> None:
        """Called after each optimization step."""
        pass

    def on_stage_end(self, stage_idx: int, state: State, ctx: Context) -> None:
        """Called when a stage ends."""
        pass
