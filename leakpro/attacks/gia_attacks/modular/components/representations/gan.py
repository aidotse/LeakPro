#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""GAN-based representations.

Latent-space search and generator fine-tuning for GIAS/GIFD/GGL, mapping
between an optimizable latent (or generator weight) space and pixel data
space via a frozen (or unfrozen) pre-trained generator.
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from leakpro.attacks.gia_attacks.modular.components.constraints.spherical import project_to_radius
from leakpro.attacks.gia_attacks.modular.core.interfaces import Representation

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.state import Context, State


GENERATOR_STATE = "generator_state"


def _check_range(output_range: tuple[float, float]) -> tuple[float, float]:
    lo, hi = (float(v) for v in output_range)
    if not hi > lo:
        raise ValueError(f"output_range must be (lo, hi) with hi > lo, got {output_range!r}")
    return lo, hi


def to_unit_range(raw: Tensor, output_range: tuple[float, float]) -> Tensor:
    """Map a generator output spanning ``output_range`` linearly onto data space [0, 1]."""
    lo, hi = output_range
    return (raw - lo) / (hi - lo)


def _frozen_run_copy(generator: torch.nn.Module, ctx: Context) -> torch.nn.Module:
    """The run's shared private copy of ``generator``, on the run device, in eval mode, with no parameter gradients.

    For representations that only decode through the generator and so never change the copy.
    """
    gen = ctx.session.owned_copy(generator)
    gen.eval()
    gen.requires_grad_(False)
    return gen.to(ctx.device)


def _run_generator(generator: torch.nn.Module, state: State, ctx: Context) -> torch.nn.Module:
    """A private copy of ``generator``: the carried fine-tuned weights if any, in eval mode, on the run device.

    Every binding gets its own copy, so representations of different stages never share (or
    change) one another's generator.
    """
    gen = copy.deepcopy(generator)
    carried = state.artifacts.get(GENERATOR_STATE)
    if carried is not None:
        gen.load_state_dict(carried)
    return gen.eval().to(ctx.device)


def _weights_view(generator: torch.nn.Module) -> dict[str, Tensor]:
    """The generator's weights as detached views (no copy); a driver clones them when it keeps an iterate."""
    return {k: v.detach() for k, v in generator.state_dict().items()}


def _generator_labels(state: State, ctx: Context, num_classes: int) -> Tensor:
    """The run's class labels, flattened in the order of the flat latent batch ``[*ctx.axes.shape, N]``.

    Reads ``state.target.value``: integer class indices ``[N]`` shared across the axis prefix (oracle or
    inferred labels), or ``[*ctx.axes.shape, N]`` when the target carries the prefix.

    Raises:
        ValueError: If there are no labels, they are not integer class indices, their shape does not match
            the batch, or a label lies outside ``[0, num_classes)``.

    """
    labels = state.target.value if state.target is not None else None
    if labels is None:
        raise ValueError(
            "A conditional GAN representation needs class labels: pass target= to run_attack, or give the attack "
            "a target_inference, so that state.target.value holds them when the stage binds."
        )
    if labels.dtype.is_floating_point or labels.dtype == torch.bool:
        raise ValueError(f"A conditional GAN representation needs integer class labels, got dtype {labels.dtype}.")
    prefix, n = tuple(ctx.axes.shape), ctx.input_shape[0]
    if tuple(labels.shape) == (n,):
        labels = labels.expand(*prefix, n)
    elif tuple(labels.shape) != (*prefix, n):
        raise ValueError(
            f"A conditional GAN representation needs one label per sample: shape ({n},) or {(*prefix, n)}, "
            f"got {tuple(labels.shape)}."
        )
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= num_classes):
        raise ValueError(f"Class labels must lie in [0, {num_classes}), got values in "
                         f"[{int(labels.min())}, {int(labels.max())}].")
    return labels.reshape(-1).to(ctx.device)


class GANRepresentation(Representation):
    """Maps an optimizable latent code to data space via a frozen generator.

    ``decode()`` returns data space ([0, 1] for images): the generator's raw output, which spans
    ``output_range``, is mapped linearly onto [0, 1]. The modality then maps data space to the
    model input, as for every other representation.

    Args:
        generator: Pre-trained generator (e.g. loaded via
            ``leakpro.fl_utils.gan_handler.load_pretrained_gan``). Each run decodes through a
            private frozen copy (eval mode, no parameter gradients); the caller's module is never
            modified, and its parameters are never included in ``parameters()``.
        latent_dim: Dimensionality of the generator's latent input.
        output_range: ``(lo, hi)`` range of the generator's raw output; ``(-1, 1)`` for a Tanh head.
        conditional: If True, decodes with ``generator(latent, labels)``: ``labels`` are the run's integer class
            labels (``state.target.value`` when the stage binds, repeated across the axis prefix), one per
            latent vector. ``bind`` raises when the state carries no labels.
        spherical: If True, ``project()`` rescales the latent back onto its
            initial L2-norm radius after every optimizer step (GIFD stage 0).
        num_classes: Required when ``conditional=True``: the number of classes the generator is conditioned on.
            ``bind`` checks that every label lies in ``[0, num_classes)``. Unused otherwise.

    Raises:
        ValueError: If ``conditional=True`` without ``num_classes``.

    """

    def __init__(
        self,
        generator: torch.nn.Module,
        latent_dim: int,
        *,
        output_range: tuple[float, float] = (-1.0, 1.0),
        conditional: bool = False,
        spherical: bool = False,
        num_classes: int | None = None,
    ) -> None:
        if conditional and num_classes is None:
            raise ValueError("A conditional GAN representation needs num_classes, to check the labels it passes on.")
        self.generator = generator
        self.latent_dim = latent_dim
        self.output_range = _check_range(output_range)
        self.conditional = conditional
        self.spherical = spherical
        self.num_classes = num_classes
        self._latent: Tensor | None = None
        self._labels: Tensor | None = None
        self._initial_radius: Tensor | None = None
        self._carried_generator = False

    def bind(self, state: State, ctx: Context) -> GANRepresentation:
        """Copy with a latent resumed from the "latent" artifact (else Gaussian) and a private frozen generator."""
        bound = copy.copy(self)
        # The latent is [*ctx.axes.shape, N, latent_dim] (N = ctx.input_shape[0]), so artifacts and
        # the decoded batch carry the axis prefix the driver slices along. A latent artifact from a
        # prior stage is authoritative.
        if state.artifacts.get("latent") is not None:
            bound._latent = state.artifacts["latent"].detach().clone().to(ctx.device).requires_grad_(True)
        else:
            shape = (*ctx.axes.shape, ctx.input_shape[0], bound.latent_dim)
            bound._latent = torch.randn(shape, device=ctx.device, requires_grad=True)
        bound.generator = _run_generator(self.generator, state, ctx).requires_grad_(False)
        bound._labels = _generator_labels(state, ctx, bound.num_classes) if bound.conditional else None
        bound._carried_generator = state.artifacts.get(GENERATOR_STATE) is not None
        if bound.spherical:
            bound._initial_radius = bound._latent.detach().norm(dim=-1, keepdim=True).clone()
        return bound

    def parameters(self) -> list[dict]:
        """The single "latent" parameter group."""
        return [{"name": "latent", "params": [self._latent], "lr_scale": 1.0}]

    def decode(self) -> Tensor:
        # The generator consumes a flat batch; the axis prefix is folded in and restored.
        """Run the latent through the generator and map its output range onto [0, 1]."""
        prefix = self._latent.shape[:-1]
        flat = self._latent.reshape(-1, self._latent.shape[-1])
        raw = self.generator(flat, self._labels) if self.conditional else self.generator(flat)
        out = to_unit_range(raw, self.output_range)
        return out.reshape(*prefix, *out.shape[1:])

    def encode(self, data: Tensor) -> Tensor:  # noqa: ARG002 -- Representation interface
        """Not supported: inverting the generator would need its own optimisation."""
        raise NotImplementedError("a GAN latent cannot be encoded from data; start from stage artifacts")

    def project(self, _ctx: Context | None = None) -> None:
        """When spherical, rescale each latent vector back to its initial L2 radius; otherwise a no-op."""
        if not self.spherical:
            return
        with torch.no_grad():
            dim = self._latent.shape[-1]
            flat = project_to_radius(self._latent.data.reshape(-1, dim), self._initial_radius.reshape(-1, 1))
            self._latent.data.copy_(flat.view_as(self._latent))

    def to_observable(self) -> Tensor:
        """The decoded batch without gradient tracking."""
        with torch.no_grad():
            return self.decode().detach()

    def artifacts(self) -> dict[str, Any]:
        """The latent, plus the generator weights when a fine-tuned generator was carried in."""
        if self._latent is None:
            return {}
        out: dict[str, Any] = {"latent": self._latent.detach()}
        if self._carried_generator:
            out[GENERATOR_STATE] = _weights_view(self.generator)
        return out


class UnfrozenGANRepresentation(GANRepresentation):
    """GIAS stage 2 representation.

    Fixes the latent (read from ``state.artifacts["latent"]``) and instead
    exposes the generator's own weights as the optimizable group, fine-tuning
    the generator itself to match the intercepted gradients. ``artifacts()`` carries the
    fine-tuned weights, so the driver's best-iterate snapshot holds the weights of the same
    iterate as the returned image.
    """

    def __init__(
        self,
        generator: torch.nn.Module,
        latent_dim: int,
        *,
        output_range: tuple[float, float] = (-1.0, 1.0),
        conditional: bool = False,
        num_classes: int | None = None,
    ) -> None:
        super().__init__(
            generator, latent_dim, output_range=output_range,
            conditional=conditional, spherical=False, num_classes=num_classes,
        )

    def bind(self, state: State, ctx: Context) -> UnfrozenGANRepresentation:
        """Copy with the fixed latent of the "latent" artifact and a private, trainable copy of the generator."""
        bound = copy.copy(self)
        if "latent" not in state.artifacts:
            raise ValueError(
                "UnfrozenGANRepresentation.bind() requires state.artifacts['latent'] "
                "from a prior GANRepresentation stage — it fine-tunes the generator "
                "around a fixed latent, it does not search the latent itself."
            )
        bound._latent = state.artifacts["latent"].detach().clone().to(ctx.device)  # fixed, no grad
        # This binding's private copy of the generator is fine-tuned; the caller's generator never is.
        bound.generator = _run_generator(self.generator, state, ctx)
        bound._labels = _generator_labels(state, ctx, bound.num_classes) if bound.conditional else None
        for p in bound.generator.parameters():
            p.requires_grad_(True)
        return bound

    def parameters(self) -> list[dict]:
        """The single "generator_weights" parameter group: every weight of the private generator copy."""
        return [{"name": "generator_weights", "params": list(self.generator.parameters()), "lr_scale": 1.0}]

    def artifacts(self) -> dict[str, Any]:
        """The fixed latent and the generator weights (views; a driver that keeps them clones them), restored by a later stage."""
        if self._latent is None:
            return {}
        return {"latent": self._latent.detach(), GENERATOR_STATE: _weights_view(self.generator)}
