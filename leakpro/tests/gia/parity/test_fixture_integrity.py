#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""The recorded fixtures are complete and internally consistent."""
import pytest
import torch

from leakpro.tests.gia import helpers
from leakpro.tests.gia.parity import fixtures

EXPECTED_KEYS = {
    "signal": {"gradients"},
    "labels": {"gradients", "inferred"},
    "loss_points": {"observation", "target_labels", "points", "total", "terms", "grad_wrt_data"},
    "trajectory": {"observation", "target_labels", "z0", "steps", "iterates", "losses", "terms"},
}


def test_manifest_records_provenance() -> None:
    m = fixtures.manifest()
    assert m["v1_source"] and m["torch"] and m["python"]
    assert m["mean"] == helpers.CIFAR_MEAN and m["std"] == helpers.CIFAR_STD
    assert {meta["kind"] for meta in m["cases"].values()} == set(EXPECTED_KEYS)


@pytest.mark.parametrize("case", sorted(fixtures.manifest()["cases"]))
def test_case_has_required_fields(case: str) -> None:
    payload = fixtures.load(case)
    assert EXPECTED_KEYS[payload["kind"]] <= set(payload)
    assert payload["images"].min() >= 0 and payload["images"].max() <= 1
    with torch.no_grad():
        assert payload["model"](payload["images"]).shape[-1] == helpers.NUM_CLASSES


@pytest.mark.parametrize("case", fixtures.cases("loss_points"))
def test_loss_point_totals_are_the_sum_of_their_terms(case: str) -> None:
    payload = fixtures.load(case)
    sums = torch.tensor([sum(terms.values()) for terms in payload["terms"]])
    torch.testing.assert_close(sums, payload["total"].to(sums.dtype))


@pytest.mark.parametrize("case", fixtures.cases("loss_points"))
def test_gradient_matching_vanishes_at_the_client_batch(case: str) -> None:
    """Point 0 is the client's own batch: with the labels v1 used there, matching is exact."""
    payload = fixtures.load(case)
    if not torch.equal(payload["target_labels"].flatten(), payload["labels"]):
        pytest.skip("v1 inferred a permutation of the labels, so the client batch is not a zero of the objective")
    torch.testing.assert_close(payload["points"][0], payload["images"])
    matching = [v for name, v in payload["terms"][0].items() if name.startswith("GradientMatchingLoss")]
    assert len(matching) == 1 and matching[0] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("case", fixtures.cases("trajectory"))
def test_trajectory_starts_at_z0_and_moves(case: str) -> None:
    payload = fixtures.load(case)
    assert payload["steps"][0] == 0
    torch.testing.assert_close(payload["iterates"][0], payload["z0"])
    assert not torch.equal(payload["iterates"][0], payload["iterates"][-1])
    assert len(payload["losses"]) == len(payload["terms"]) > payload["steps"][-1]
