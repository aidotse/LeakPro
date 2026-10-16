#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Contract: an attack's ``MIAResult.audit_indices`` names the population point behind each score.

The shared fixture uses ``train_indices = 0..n`` so audit position and population
index coincide, which hides ordering bugs. These tests reverse both index lists
first, so a mapping that merely returns positions fails.
"""

import numpy as np
import pytest
from dotmap import DotMap

from leakpro.attacks.mia_attacks.base import AttackBASE
from leakpro.attacks.mia_attacks.lira import AttackLiRA
from leakpro.attacks.mia_attacks.multi_signal_lira import AttackMSLiRA
from leakpro.attacks.mia_attacks.rmia import AttackRMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.tests.constants import get_audit_config, get_shadow_model_config
from leakpro.tests.input_handler.image_input_handler import ImageInputHandler


def _reverse_audit_order(handler: ImageInputHandler) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(handler.train_indices)[::-1].copy()
    test = np.asarray(handler.test_indices)[::-1].copy()
    handler.train_indices = train
    handler.test_indices = test
    handler.configs.shadow_model = get_shadow_model_config()
    if ShadowModelHandler.is_created() is False:
        ShadowModelHandler(handler)
    return train, test


def _lira_params() -> DotMap:
    params = DotMap({k: v for k, v in get_audit_config().attack_list[0].items() if k != "attack"})
    params.online = True
    return params


def _rmia_params() -> DotMap:
    params = DotMap({k: v for k, v in get_audit_config().attack_list[1].items() if k != "attack"})
    params.online = True
    return params


ATTACKS = {
    "rmia": lambda h: AttackRMIA(h, _rmia_params()),
    "lira": lambda h: AttackLiRA(h, _lira_params()),
    "base": lambda h: AttackBASE(h, {"num_shadow_models": 2, "online": True, "training_data_fraction": 0.5}),
    "multi_signal_lira": lambda h: AttackMSLiRA(h, {"signals": ["ModelRescaledLogits"], "num_shadow_models": 4,
                                                   "training_data_fraction": 0.5, "online": True}),
}


@pytest.mark.parametrize("attack_name", sorted(ATTACKS))
def test_scores_map_to_the_population_points_they_belong_to(image_handler: ImageInputHandler,
                                                            attack_name: str) -> None:
    """Each score's index is the right population point, with the right membership, in audit order."""
    train, test = _reverse_audit_order(image_handler)
    attack = ATTACKS[attack_name](image_handler)
    attack.prepare_attack()
    result = attack.run_attack()

    indices = result.audit_indices
    assert indices is not None
    assert len(indices) == len(result.signal_values) == len(result.true)

    # Membership of every scored point is consistent with the target's split.
    train_set = set(train.tolist())
    assert all((int(i) in train_set) == bool(m) for i, m in zip(indices, result.true))

    # The scored points are the audit set in audit order (train then test), minus any the
    # attack dropped, so the reported indices must be an ordered subsequence of it.
    audit_order = np.concatenate([train, test]).tolist()
    pos = 0
    for idx in indices.tolist():
        pos = audit_order.index(idx, pos) + 1  # raises ValueError if the order is violated

    # And it is not just the audit-set position in disguise.
    assert indices.tolist() != list(range(len(indices)))
