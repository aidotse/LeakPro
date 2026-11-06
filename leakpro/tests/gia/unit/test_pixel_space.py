#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import torch

from leakpro.attacks.gia_attacks.modular.components.initializers import FromState, FromTensor, Gaussian
from leakpro.attacks.gia_attacks.modular.components.representations.pixel import PixelRepresentation
from leakpro.attacks.gia_attacks.modular.core.state import Context, State
from leakpro.tests.gia import helpers

SHAPE = (2, *helpers.IMAGE_SHAPE)


def _ctx() -> Context:
    return Context(model=helpers.tiny_bn_cnn(), observation=None, modality=helpers.image_modality(), task=None,
                   input_shape=SHAPE, device=helpers.DEVICE)


def _bound(rep: PixelRepresentation, state: State | None = None, ctx: Context | None = None) -> PixelRepresentation:
    return rep.bind(state or State(), ctx or _ctx())


def test_model_input_space_optimizes_z_exactly() -> None:
    z0 = torch.randn(SHAPE, generator=torch.Generator().manual_seed(0))
    rep = _bound(PixelRepresentation(space="model_input", init=FromTensor(z0), clip=False))
    (param,) = rep.parameters()[0]["params"]
    assert torch.equal(param.detach(), z0)
    assert rep.model_input(helpers.image_modality()) is param
    torch.testing.assert_close(rep.decode(), helpers.image_modality().strip_model_input(z0))


def test_model_input_box_is_the_image_of_unit_interval() -> None:
    rep = _bound(PixelRepresentation(space="model_input", init=Gaussian(std=10.0, seed=0)))
    rep.project()
    x = rep.decode()
    assert x.min() >= -1e-6 and x.max() <= 1 + 1e-6


def test_clip_false_leaves_values_outside_the_box() -> None:
    z0 = torch.full(SHAPE, 50.0)
    rep = _bound(PixelRepresentation(space="model_input", init=FromTensor(z0), clip=False))
    rep.project()
    assert rep.to_observable().max() > 1.0


def test_data_space_resume_from_state_input_is_encoded_for_model_input_space() -> None:
    x = torch.rand(SHAPE)
    rep = _bound(PixelRepresentation(space="model_input"), State(input=x))
    torch.testing.assert_close(rep.decode(), x, rtol=1e-6, atol=1e-6)


def test_from_state_without_state_raises() -> None:
    with pytest.raises(ValueError, match="state"):
        _bound(PixelRepresentation(init=FromState()))


def test_default_data_space_is_unchanged() -> None:
    ctx = _ctx()                                # built first: model construction draws from the global RNG
    torch.manual_seed(1)
    expected = torch.empty(SHAPE).uniform_(0.0, 1.0)
    torch.manual_seed(1)
    rep = _bound(PixelRepresentation(), ctx=ctx)
    torch.testing.assert_close(rep.decode().detach(), expected, rtol=0, atol=0)
