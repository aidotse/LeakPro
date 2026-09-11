#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for AttackFactoryMIA's auxiliary-handler gating.

Tests cover:
- every registered attack still declares that it requires both shadow and distillation
  handlers (so the gate is a no-op for the existing attack set)
- the factory builds both handlers for such an attack
- the factory builds neither handler for an attack that opts out via the class flags
- AbstractMIA._wrap_target_model returns a PytorchModel by default
- an unknown attack name raises ValueError before any handler is constructed
"""

import pytest
from pydantic import BaseModel

import leakpro.attacks.mia_attacks.attack_factory_mia as factory_module
from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.mia_attacks.attack_factory_mia import AttackFactoryMIA
from leakpro.attacks.mia_attacks.llm.abstract_llm_mia import AbstractLLMMIA
from leakpro.attacks.mia_attacks.rmia import AttackRMIA
from leakpro.signals.signal_extractor import PytorchModel
from leakpro.tests.input_handler.image_input_handler import ImageInputHandler


class _NoAuxConfig(BaseModel):
    """Empty config for the opt-out test attack."""


class _NoAuxAttack(AbstractMIA):
    """Minimal attack that trains nothing and opts out of both auxiliary handlers."""

    requires_shadow_models = False
    requires_distillation_models = False
    AttackConfig = _NoAuxConfig

    def __init__(self, handler: ImageInputHandler, configs: dict) -> None:
        self.configs = _NoAuxConfig(**(configs or {}))
        super().__init__(handler)

    def description(self) -> dict:
        """Return the 4-key description."""
        return {"title_str": "noaux", "reference": "", "summary": "", "detailed": ""}

    def prepare_attack(self) -> None:
        """Nothing to prepare."""

    def run_attack(self) -> None:
        """Nothing to run."""


@pytest.fixture
def _reset_factory_singletons(image_handler: ImageInputHandler) -> None:  # noqa: ARG001
    """Start each test with fresh factory state and restore it afterwards.

    Depends on ``image_handler`` so it runs *after* it: building that fixture runs
    ``LeakPro(...)`` → ``AttackScheduler``, which itself populates the factory singletons.
    """
    saved = (AttackFactoryMIA.shadow_model_handler, AttackFactoryMIA.distillation_model_handler)
    AttackFactoryMIA.shadow_model_handler = None
    AttackFactoryMIA.distillation_model_handler = None
    yield
    AttackFactoryMIA.shadow_model_handler, AttackFactoryMIA.distillation_model_handler = saved


def test_all_registered_attacks_require_both_handlers() -> None:
    """The gate must be a no-op for every attack that existed before it (LLM attacks opt out by design)."""
    for name, attack_cls in AttackFactoryMIA.attack_classes.items():
        if issubclass(attack_cls, (_NoAuxAttack, AbstractLLMMIA)):
            continue
        assert attack_cls.requires_shadow_models is True, name
        assert attack_cls.requires_distillation_models is True, name


@pytest.mark.usefixtures("_reset_factory_singletons")
def test_factory_builds_both_handlers_for_default_attack(image_handler: ImageInputHandler) -> None:
    """An attack with default flags gets both handlers, exactly as before the gate.

    Uses RMIA rather than the population attack: the image fixture's audit set is the whole
    population, which AttackP rejects by design.
    """
    attack = AttackFactoryMIA.create_attack("rmia", {}, image_handler)
    assert isinstance(attack, AttackRMIA)
    assert AttackFactoryMIA.shadow_model_handler is not None
    assert AttackFactoryMIA.distillation_model_handler is not None


@pytest.mark.usefixtures("_reset_factory_singletons")
def test_factory_skips_handlers_for_opted_out_attack(
    image_handler: ImageInputHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An attack that sets both flags to False must not trigger handler construction at all."""

    def _boom(*_: object, **__: object) -> None:
        raise AssertionError("auxiliary handler constructed for an attack that opted out")

    monkeypatch.setattr(factory_module, "ShadowModelHandler", _boom)
    monkeypatch.setattr(factory_module, "DistillationModelHandler", _boom)
    monkeypatch.setitem(AttackFactoryMIA.attack_classes, "_noaux", _NoAuxAttack)

    attack = AttackFactoryMIA.create_attack("_noaux", {}, image_handler)
    assert isinstance(attack, _NoAuxAttack)
    assert AttackFactoryMIA.shadow_model_handler is None
    assert AttackFactoryMIA.distillation_model_handler is None


@pytest.mark.usefixtures("_reset_factory_singletons")
def test_unknown_attack_raises_before_building_handlers(
    image_handler: ImageInputHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A typo in the attack name should not cost a target forward pass."""

    def _boom(*_: object, **__: object) -> None:
        raise AssertionError("handler constructed for an unknown attack name")

    monkeypatch.setattr(factory_module, "ShadowModelHandler", _boom)
    monkeypatch.setattr(factory_module, "DistillationModelHandler", _boom)
    with pytest.raises(ValueError, match="Unknown attack type"):
        AttackFactoryMIA.create_attack("does_not_exist", {}, image_handler)


def test_wrap_target_model_default_is_pytorch_model(image_handler: ImageInputHandler) -> None:
    """The base hook preserves the original PytorchModel wrapping."""
    wrapped = AbstractMIA._wrap_target_model(image_handler)
    assert isinstance(wrapped, PytorchModel)
    assert wrapped.model_obj is image_handler.target_model
