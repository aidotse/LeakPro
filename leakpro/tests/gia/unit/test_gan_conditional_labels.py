#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""A conditional GAN representation decodes with the run's class labels, and checks them against ``num_classes``."""
from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from leakpro.attacks.gia_attacks.modular.components.representations.gan import (
    GANRepresentation,
    UnfrozenGANRepresentation,
)
from leakpro.attacks.gia_attacks.modular.core.axes import Axis, AxisManager
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, Context, State, TargetInfo

NUM_CLASSES = 3


class _ConditionalGenerator(nn.Module):
    """Output depends on the class label through an embedding; records the labels of its last call."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 16)
        self.emb = nn.Embedding(NUM_CLASSES, 16)
        self.seen: Tensor | None = None

    def forward(self, z: Tensor, labels: Tensor) -> Tensor:
        self.seen = labels
        return torch.tanh(self.lin(z) + self.emb(labels)).view(-1, 1, 4, 4)


def _ctx(batch_size: int, axes: AxisManager | None = None) -> Context:
    return Context(model=nn.Module(), observation=ClientObservation(), modality=None, task=None,
                   input_shape=(batch_size, 1, 4, 4), device=torch.device("cpu"), axes=axes or AxisManager())


def test_decode_passes_the_state_labels_to_the_generator() -> None:
    labels = torch.tensor([2, 0])
    rep = GANRepresentation(_ConditionalGenerator(), 4, conditional=True, num_classes=NUM_CLASSES)
    bound = rep.bind(State(target=TargetInfo(value=labels)), _ctx(2))
    out = bound.decode()
    assert torch.equal(bound.generator.seen, labels)
    flat = bound.artifacts()["latent"]
    torch.testing.assert_close(out, (bound.generator(flat, labels) + 1.0) / 2.0)


def test_shared_labels_are_repeated_across_the_seed_axis() -> None:
    labels = torch.tensor([1, 2])
    rep = GANRepresentation(_ConditionalGenerator(), 4, conditional=True, num_classes=NUM_CLASSES)
    bound = rep.bind(State(target=TargetInfo(value=labels)), _ctx(2, AxisManager(Axis("seed", 3))))
    assert bound.decode().shape == (3, 2, 1, 4, 4)
    assert torch.equal(bound.generator.seen, labels.repeat(3))


def test_unfrozen_representation_decodes_with_the_labels_too() -> None:
    labels = torch.tensor([0, 1])
    state = State(target=TargetInfo(value=labels), artifacts={"latent": torch.randn(2, 4)})
    rep = UnfrozenGANRepresentation(_ConditionalGenerator(), 4, conditional=True, num_classes=NUM_CLASSES)
    bound = rep.bind(state, _ctx(2))
    bound.decode()
    assert torch.equal(bound.generator.seen, labels)


def test_conditional_requires_num_classes() -> None:
    with pytest.raises(ValueError, match="num_classes"):
        GANRepresentation(_ConditionalGenerator(), 4, conditional=True)


def test_conditional_bind_requires_labels() -> None:
    rep = GANRepresentation(_ConditionalGenerator(), 4, conditional=True, num_classes=NUM_CLASSES)
    with pytest.raises(ValueError, match="labels"):
        rep.bind(State(), _ctx(2))


@pytest.mark.parametrize(
    "labels",
    [torch.tensor([0, NUM_CLASSES]), torch.tensor([0, 1, 2]), torch.tensor([0.0, 1.0])],
    ids=["out-of-range", "wrong-count", "not-integer"],
)
def test_bind_rejects_labels_that_do_not_fit(labels: Tensor) -> None:
    rep = GANRepresentation(_ConditionalGenerator(), 4, conditional=True, num_classes=NUM_CLASSES)
    with pytest.raises(ValueError, match="label"):
        rep.bind(State(target=TargetInfo(value=labels)), _ctx(2))
