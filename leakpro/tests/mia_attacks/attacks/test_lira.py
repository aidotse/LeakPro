#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np

from leakpro.attacks.mia_attacks.lira import AttackLiRA
from leakpro.reporting.mia_result import MIAResult
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.tests.constants import get_audit_config, get_shadow_model_config
from leakpro.tests.input_handler.image_input_handler import ImageInputHandler


def test_lira_setup(image_handler:ImageInputHandler) -> None:
    """Test the initialization of LiRA."""
    audit_config = get_audit_config()
    lira_params = audit_config.attack_list[0]
    lira_obj = AttackLiRA(image_handler, lira_params)

    assert lira_obj is not None
    assert lira_obj.target_model is not None
    assert lira_obj.online == lira_params.online
    assert lira_obj.num_shadow_models == lira_params.num_shadow_models
    assert lira_obj.training_data_fraction == lira_params.training_data_fraction

    description = lira_obj.description()
    assert len(description) == 4

def test_lira_prepare_online_attack(image_handler:ImageInputHandler) -> None:
    audit_config = get_audit_config()
    lira_params = audit_config.attack_list[0]
    lira_params.online = True

    image_handler.configs.shadow_model = get_shadow_model_config()
    lira_obj = AttackLiRA(image_handler, lira_params)

    if ShadowModelHandler.is_created() == False:
        ShadowModelHandler(image_handler)

    lira_obj.prepare_attack()

    # ensure correct number of shadow models are read
    assert len(lira_obj.shadow_models) == lira_params.num_shadow_models
    # ensure the attack data indices correspond to the correct pool
    assert sorted(lira_obj.attack_data_indices) == list(range(image_handler.population_size))

    # Check that the filtering of the attack data is correct (this is done after shadow models are created)
    n_attack_points = len(lira_obj.train_indices) + len(lira_obj.test_indices)
    assert n_attack_points > 0
    assert lira_obj.shadow_models_logits.shape == (lira_params.num_shadow_models, n_attack_points)
    assert lira_obj.target_logits.shape == (n_attack_points, )

def test_lira_prepare_offline_attack(image_handler:ImageInputHandler) -> None:
    audit_config = get_audit_config()
    lira_params = audit_config.attack_list[0]
    lira_params.online = False

    image_handler.configs.shadow_model = get_shadow_model_config()
    lira_obj = AttackLiRA(image_handler, lira_params)

    if ShadowModelHandler.is_created() == False:
        ShadowModelHandler(image_handler)

    lira_obj.prepare_attack()

    # ensure correct number of shadow models are read
    assert len(lira_obj.shadow_models) == lira_params.num_shadow_models
    # ensure the attack data indices correspond to the correct pool (all of the data)
    assert sorted(lira_obj.attack_data_indices) == list(range(image_handler.population_size))

    # Check that the filtering of the attack data is correct (this is done after shadow models are created)
    n_attack_points = len(lira_obj.train_indices) + len(lira_obj.test_indices)
    assert n_attack_points > 0
    assert lira_obj.shadow_models_logits.shape == (lira_params.num_shadow_models, n_attack_points)
    assert lira_obj.target_logits.shape == (n_attack_points, )


def test_lira_online_attack(image_handler:ImageInputHandler):
    # Set up for testing
    audit_config = get_audit_config()
    lira_params = audit_config.attack_list[0]
    lira_params.online = True
    image_handler.configs.shadow_model = get_shadow_model_config()
    lira_obj = AttackLiRA(image_handler, lira_params)
    if ShadowModelHandler.is_created() == False:
        ShadowModelHandler(image_handler)
    lira_obj.prepare_attack()

    # Test standard deviation calculation
    std_fixed = lira_obj.get_std(lira_obj.shadow_models_logits.flatten(),
                           ~lira_obj.out_indices.flatten(),
                           True,
                           "fixed")

    lira_obj.fixed_in_std = std_fixed
    lira_obj.fixed_out_std = std_fixed

    std_carlini = lira_obj.get_std(lira_obj.shadow_models_logits.flatten(),
                           ~lira_obj.out_indices.flatten(),
                           True,
                           "carlini")

    std_individual = lira_obj.get_std(lira_obj.shadow_models_logits.flatten(),
                           ~lira_obj.out_indices.flatten(),
                           True,
                           "individual_carlini")
    assert std_fixed == std_carlini
    assert std_fixed == std_individual

    # Test attack
    lira_obj.run_attack()
    assert lira_obj.fixed_in_std != lira_obj.fixed_out_std
    n_attack_points = len(lira_obj.train_indices) + len(lira_obj.test_indices)
    assert len(lira_obj.in_member_signals)+len(lira_obj.out_member_signals) == n_attack_points
    assert not np.any(np.isnan(lira_obj.in_member_signals))
    assert not np.any(np.isnan(lira_obj.out_member_signals))

def test_lira_offline_attack(image_handler:ImageInputHandler):
    # Set up for testing
    audit_config = get_audit_config()
    lira_params = audit_config.attack_list[0]
    lira_params.online = False
    image_handler.configs.shadow_model = get_shadow_model_config()
    lira_obj = AttackLiRA(image_handler, lira_params)
    if ShadowModelHandler.is_created() == False:
        ShadowModelHandler(image_handler)
    lira_obj.prepare_attack()
    lira_obj.fix_var_threshold = 0.0

    # Test attack
    lira_result = lira_obj.run_attack()
    assert lira_obj.fixed_in_std != lira_obj.fixed_out_std
    n_attack_points = len(lira_obj.train_indices) + len(lira_obj.test_indices)
    assert len(lira_obj.in_member_signals)+len(lira_obj.out_member_signals) == n_attack_points
    assert not np.any(np.isnan(lira_obj.in_member_signals))
    assert not np.any(np.isnan(lira_obj.out_member_signals))

    assert lira_result is not None
    assert isinstance(lira_result, MIAResult)

def test_rescale_logits_should_accept_float_labels() -> None:
    """Datasets using BCEWithLogitsLoss carry float labels, which cannot index an array.

    Regression test: float labels used to raise
    "arrays used as indices must be of integer (or boolean) type".
    """
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(32, 1)).astype(np.float32)
    labels_int = rng.integers(0, 2, 32)

    from_float = AttackLiRA.rescale_logits(None, logits, labels_int.astype(np.float32))
    from_int = AttackLiRA.rescale_logits(None, logits, labels_int)

    assert np.allclose(from_float, from_int)


def test_rescale_logits_should_match_the_torch_implementation_for_binary_models() -> None:
    """The numpy and torch rescaling paths must agree, otherwise cached and live logits diverge."""
    import torch
    from torch import nn

    from leakpro.signals.signal_extractor import PytorchModel

    rng = np.random.default_rng(2)
    logits = rng.normal(size=(64, 1)).astype(np.float32)
    labels = rng.integers(0, 2, 64).astype(np.float32)

    class _ConstantModel(nn.Module):
        """Returns fixed logits so both paths score exactly the same values."""

        def __init__(self, values: np.ndarray) -> None:
            super().__init__()
            self.register_buffer("values", torch.tensor(values))

        def forward(self, _: torch.Tensor) -> torch.Tensor:
            return self.values

    torch_scores = PytorchModel(_ConstantModel(logits), nn.BCEWithLogitsLoss()).get_rescaled_logits(
        torch.zeros(64, 3), torch.from_numpy(labels)
    )
    numpy_scores = AttackLiRA.rescale_logits(None, logits, labels)

    assert np.allclose(numpy_scores, torch_scores, atol=1e-5)
