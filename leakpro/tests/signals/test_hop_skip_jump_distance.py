#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for leakpro.signals.utils.HopSkipJumpDistance.

Covers:
- geometric_progression_for_stepsize() returns a tensor on self.device, not
  wherever the CPU-only bookkeeping loop happens to leave it
"""
from unittest.mock import MagicMock

import numpy as np
import torch

from leakpro.signals.utils.HopSkipJumpDistance import HopSkipJumpDistance


class TestGeometricProgressionForStepsize:
    def test_returns_tensor_on_self_device(self):
        """Regression test: the returned epsilon must live on self.device.

        The function deliberately keeps its internal bookkeeping (batch_epsilon) on
        CPU throughout the loop -- a Habana graph-compiler workaround -- so it's easy
        to forget to move the final return value back before handing it to a caller
        that multiplies it against tensors on self.device. A CPU-only CI run can't
        catch that by accident (self.device would already be cpu there), so this uses
        torch's device-agnostic "meta" device as a stand-in for "some real
        accelerator" and asserts the device identity explicitly instead.

        decision_function is stubbed to accept every sample on the first pass, so the
        while loop runs exactly once; compute_distance is stubbed to a real (non-meta)
        CPU tensor, matching how only self.device -- never the bookkeeping itself --
        represents the target accelerator in this test.
        """
        obj = MagicMock()
        obj.device = torch.device("meta")
        obj.epsilon_threshold = 1e-6
        obj.compute_distance = MagicMock(return_value=torch.tensor([2.0, 2.0]))
        obj.decision_function = MagicMock(return_value=([True, True], None))
        obj.clamping = MagicMock(side_effect=lambda x: x)

        samples = torch.empty(2, 1, 1, 1, device="meta")
        updates = torch.empty(2, 1, 1, 1, device="meta")
        perturbed = torch.empty(2, 1, 1, 1, device="meta")

        epsilon = HopSkipJumpDistance.geometric_progression_for_stepsize(
            obj, samples, updates, perturbed,
            current_iteration=1, active_indices=np.arange(2), b_i=0,
        )

        assert epsilon.device == obj.device
