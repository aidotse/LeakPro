import numpy as np
import pytest

from leakpro.attacks.mia_attacks.oslo import AttackOSLO
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.tests.constants import get_shadow_model_config
from leakpro.tests.input_handler.image_input_handler import ImageInputHandler

FAST_OSLO_PARAMS = {
    "n_audits": 3,
    "num_source_models": 1,
    "num_validation_models": 1,
    "num_sub_procedures": 1,
    "num_iterations": 1,
    "n_thresholds": 2,
    "online": False,
    "training_data_fraction": 0.5,
}


def test_oslo_setup(image_handler: ImageInputHandler) -> None:
    """Test that AttackOSLO initializes correctly with default and custom configs."""
    # Default params
    oslo_default = AttackOSLO(image_handler, None)
    assert oslo_default is not None
    assert oslo_default.target_model is not None
    assert oslo_default.num_source_models == 9
    assert oslo_default.num_validation_models == 3
    assert oslo_default.n_audits == 500
    assert oslo_default.online == False

    # Custom fast params
    oslo_custom = AttackOSLO(image_handler, FAST_OSLO_PARAMS)
    assert oslo_custom is not None
    for key, value in FAST_OSLO_PARAMS.items():
        assert getattr(oslo_custom, key) == value

    # Description
    desc = oslo_custom.description()
    assert len(desc) == 4

    # min_threshold must be > 0 (gt=0.0)
    with pytest.raises(Exception):
        AttackOSLO(image_handler, {**FAST_OSLO_PARAMS, "min_threshold": 0.0})


def test_oslo_prepare_attack(image_handler: ImageInputHandler) -> None:
    """Test that prepare_attack sets up shadow models and audit indices correctly."""
    image_handler.configs.shadow_model = get_shadow_model_config()
    oslo_obj = AttackOSLO(image_handler, FAST_OSLO_PARAMS)

    if ShadowModelHandler.is_created() == False:
        ShadowModelHandler(image_handler)

    oslo_obj.prepare_attack()

    assert len(oslo_obj.source_models) == FAST_OSLO_PARAMS["num_source_models"]
    assert len(oslo_obj.validation_models) == FAST_OSLO_PARAMS["num_validation_models"]
    assert len(oslo_obj.audit_data_indices) == FAST_OSLO_PARAMS["n_audits"]
    # audit indices must not overlap with remaining attack data
    assert len(np.intersect1d(oslo_obj.audit_data_indices, oslo_obj.attack_data_indices)) == 0
    # thresholds array has correct length
    assert len(oslo_obj.thresholds) == FAST_OSLO_PARAMS["n_thresholds"]


def test_oslo_n_audits_guard(image_handler: ImageInputHandler) -> None:
    """Test that a ValueError is raised when n_audits exceeds available data."""
    image_handler.configs.shadow_model = get_shadow_model_config()
    # Use a huge n_audits value that exceeds the dataset size
    huge_params = {**FAST_OSLO_PARAMS, "n_audits": 100000}
    oslo_obj = AttackOSLO(image_handler, huge_params)

    if ShadowModelHandler.is_created() == False:
        ShadowModelHandler(image_handler)

    with pytest.raises(ValueError, match="n_audits"):
        oslo_obj.prepare_attack()


def test_oslo_run_attack(image_handler: ImageInputHandler) -> None:
    """Test that run_attack returns a valid MIAResult with no NaN signal values."""
    image_handler.configs.shadow_model = get_shadow_model_config()
    oslo_obj = AttackOSLO(image_handler, FAST_OSLO_PARAMS)

    if ShadowModelHandler.is_created() == False:
        ShadowModelHandler(image_handler)

    oslo_obj.prepare_attack()
    result = oslo_obj.run_attack()

    assert result is not None
    assert isinstance(result, MIAResult)
    assert result.signal_values is not None
    assert result.signal_values.shape == (FAST_OSLO_PARAMS["n_audits"],)
    assert not np.any(np.isnan(result.signal_values))
