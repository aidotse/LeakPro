#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
import torch

from leakpro.attacks.gia_attacks.modular.components.constraints.spherical import project_to_radius
from leakpro.attacks.gia_attacks.modular.components.constraints.l1_ball import project_l1_ball


class TestProjectToRadius:
    def test_rescales_to_initial_radius(self):
        z = torch.tensor([[3.0, 4.0], [1.0, 0.0]])  # norms 5.0, 1.0
        initial_radius = torch.tensor([[1.0], [1.0]])
        out = project_to_radius(z, initial_radius)
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.tensor([1.0, 1.0]), atol=1e-5)

    def test_preserves_direction(self):
        z = torch.tensor([[3.0, 4.0]])
        initial_radius = torch.tensor([[2.0]])
        out = project_to_radius(z, initial_radius)
        assert torch.allclose(out / out.norm(), z / z.norm(), atol=1e-5)


class TestProjectL1Ball:
    def test_leaves_points_inside_ball_unchanged(self):
        x = torch.tensor([[0.1, 0.1, 0.1]])
        out = project_l1_ball(x, radius=1.0)
        assert torch.allclose(out, x)

    def test_projects_points_outside_ball_onto_boundary(self):
        x = torch.tensor([[3.0, 4.0]])
        out = project_l1_ball(x, radius=1.0)
        assert out.abs().sum(dim=1).item() == pytest_approx(1.0)


def pytest_approx(value, tol=1e-5):
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) < tol
    return _Approx()
