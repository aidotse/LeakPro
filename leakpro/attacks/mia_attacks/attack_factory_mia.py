#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Module that contains the AttackFactory class which is responsible for creating the attack objects."""

from contextlib import ExitStack, contextmanager
from typing import Iterator

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.mia_attacks.attack_p import AttackP
from leakpro.attacks.mia_attacks.base import AttackBASE
from leakpro.attacks.mia_attacks.dts import AttackDTS
from leakpro.attacks.mia_attacks.HSJ import AttackHopSkipJump
from leakpro.attacks.mia_attacks.lira import AttackLiRA
from leakpro.attacks.mia_attacks.llm.ez_mia import AttackEZMIA
from leakpro.attacks.mia_attacks.llm.wbc import AttackWBC
from leakpro.attacks.mia_attacks.loss_trajectory import AttackLossTrajectory
from leakpro.attacks.mia_attacks.multi_signal_lira import AttackMSLiRA
from leakpro.attacks.mia_attacks.oslo import AttackOSLO
from leakpro.attacks.mia_attacks.qmia import AttackQMIA
from leakpro.attacks.mia_attacks.ramia import AttackRaMIA
from leakpro.attacks.mia_attacks.rapid import AttackRAPID
from leakpro.attacks.mia_attacks.rmia import AttackRMIA
from leakpro.attacks.mia_attacks.seq_mia import AttackSeqMIA
from leakpro.attacks.mia_attacks.yoqo import AttackYOQO
from leakpro.attacks.utils.distillation_model_handler import DistillationModelHandler
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.input_handler.mia_handler import MIAHandler


class AttackFactoryMIA:
    """Class responsible for creating the attack objects."""

    attack_classes = {
        "population": AttackP,
        "rmia": AttackRMIA,
        "qmia": AttackQMIA,
        "loss_traj":AttackLossTrajectory,
        "seqmia":AttackSeqMIA,
        "lira": AttackLiRA,
        "HSJ" : AttackHopSkipJump,
        "yoqo": AttackYOQO,
        "base": AttackBASE,
        "ramia": AttackRaMIA,
        "multi_signal_lira": AttackMSLiRA,
        "dts": AttackDTS,
        "oslo": AttackOSLO,
        "ez_mia": AttackEZMIA,
        "wbc": AttackWBC,
        "rapid": AttackRAPID,
    }

    # Shared variables for all attacks
    shadow_model_handler = None
    distillation_model_handler = None

    @classmethod
    def create_attack(cls, name: str, attack_config: dict, handler: MIAHandler) -> AbstractMIA:  # noqa: ANN102
        """Create the attack object.

        Args:
        ----
            name (str): The name of the attack.
            attack_config (dict): The configuration for the attack.
            handler (MIAHandler): The input handler object.

        Returns:
        -------
            AttackBase: An instance of the attack object.

        Raises:
        ------
            ValueError: If the attack type is unknown.

        """

        attack_cls = cls.attack_classes.get(name)
        if attack_cls is None:
            raise ValueError(f"Unknown attack type: {name}")

        with cls._model_handlers(attack_cls, handler) as (shadow, distillation):
            if shadow is not None:
                cls.shadow_model_handler = shadow
            if distillation is not None:
                cls.distillation_model_handler = distillation
            attack_object = attack_cls(handler, attack_config)
        attack_object.set_effective_optuna_metadata(attack_config) # remove optuna metadata if params not will be optimized
        return attack_object

    @staticmethod
    @contextmanager
    def _model_handlers(attack_cls: type[AbstractMIA], handler: MIAHandler, *, isolated: bool = False) -> Iterator[tuple]:
        """Set up the requested model handlers within the audit's scope."""
        with ExitStack() as scope:
            shadow, distillation = None, None
            if getattr(attack_cls, "requires_shadow_models", True):
                if isolated:
                    scope.enter_context(ShadowModelHandler.isolated_instance())
                shadow = ShadowModelHandler(handler, precompute_logits=getattr(attack_cls, "precompute_shadow_logits", True))
            if getattr(attack_cls, "requires_distillation_models", True):
                if isolated:
                    scope.enter_context(DistillationModelHandler.isolated_instance())
                distillation = DistillationModelHandler(handler)
            yield shadow, distillation
