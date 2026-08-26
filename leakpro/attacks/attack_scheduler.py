#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Module that contains the AttackScheduler class, which is responsible for creating and executing attacks."""

from pathlib import Path

from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.utils.import_helper import Any, Dict, Self
from leakpro.utils.logger import logger


class AttackScheduler:
    """Class responsible for creating and executing attacks."""

    attack_type_to_factory = {}

    def __init__(
        self:Self,
        handler: AbstractInputHandler,
        output_dir: str
    ) -> None:
        """Initialize the AttackScheduler class.

        Args:
        ----
            handler (AbstractInputHandler): The handler object that contains the user inputs.
            output_dir (str): The directory where the results will be saved.

        """
        configs = handler.configs

        # Create factory
        attack_type = configs.audit.attack_type
        self.attack_type = attack_type
        self._initialize_factory(attack_type)

        self.attack_names = [entry["attack"] for entry in configs.audit.attack_list]
        self.attack_configs = [{k: v for k, v in entry.items() if k != "attack"} for entry in configs.audit.attack_list]

        self.attacks = []
        for attack_name, attack_config in zip(self.attack_names, self.attack_configs):
            try:
                attack = self.attack_factory.create_attack(attack_name, attack_config, handler)
                self.add_attack(attack)
                logger.info(f"Added attack: {attack_name}")
            except ValueError as e:
                logger.info(e)
                logger.info(f"Failed to create attack: {attack_name}, supported attacks: {self.attack_factory.attack_classes.keys()}")  # noqa: E501
                if configs.audit.attack_type == "extraction":
                    raise ValueError(f"Failed to create extraction attack {attack_name!r}: {e}") from e

        # Read all previous hashed attack objects from the report directory
        self.output_dir = output_dir
        self.report_dir = Path(output_dir) / "results"

        self.data_object_dir = Path(output_dir) / "data_objects"

    def _read_attack_hashes(self:Self) -> None:
        """Read all previous hashed attack objects from the report directory."""
        if self.data_object_dir.exists() and self.data_object_dir.is_dir():
            self.attack_hashes = [file.stem for file in self.data_object_dir.glob("*.json")]
        else:
            self.data_object_dir.mkdir(parents=True, exist_ok=True)
            self.attack_hashes = []

    def _initialize_factory(self:Self, attack_type:str) -> None:
        """Conditionally import attack factories based on attack."""
        if attack_type == "mia":
            try:
                from leakpro.attacks.mia_attacks.attack_factory_mia import AttackFactoryMIA  # noqa: I001, PLC0415
                self.attack_factory = AttackFactoryMIA
                logger.info("MIA attack factory loaded.")
            except ImportError as e:
                logger.error("Failed to import MIA attack module.")
                raise ImportError("MIA attack module is not available.") from e

        elif attack_type == "gia":
            try:
                from leakpro.attacks.gia_attacks.attack_factory_gia import AttackFactoryGIA # noqa: I001, PLC0415
                self.attack_factory = AttackFactoryGIA
                logger.info("GIA attack factory loaded.")
            except ImportError as e:
                logger.error("Failed to import GIA attack module.")
                raise ImportError("GIA attack module is not available.") from e

        elif attack_type == "minv":
            try:
                from leakpro.attacks.minv_attacks.attack_factory_minv import AttackFactoryMINV # noqa: I001, PLC0415
                self.attack_factory = AttackFactoryMINV
                logger.info("MINV attack factory loaded.")
            except ImportError as e:
                logger.error("Failed to import MINV attack module.")
                raise ImportError("MINV attack module is not available.") from e

        elif attack_type == "extraction":
            try:
                from leakpro.attacks.extraction_attacks.attack_factory_extraction import AttackFactoryExtraction  # noqa: I001, PLC0415, E501
                self.attack_factory = AttackFactoryExtraction
                logger.info("Extraction attack factory loaded.")
            except ImportError as e:
                logger.error("Failed to import extraction attack module.")
                raise ImportError("Extraction attack module is not available.") from e

        else:
            logger.error(f"Unsupported attack type: {attack_type}")
            raise ValueError(f"Unsupported attack type: {attack_type}.")

    def add_attack(self:Self, attack: Any) -> None:
        """Add an attack to the list of attacks."""
        self.attacks.append(attack)

    def run_attacks(self: Self, use_optuna:bool=False) -> Dict[str, Any]:
        """Run the attacks and return the results."""
        if self.attack_type == "extraction":
            self._preflight_extraction_results()
        results = []
        for attack_obj, attack_type in zip(self.attacks, self.attack_names):
            run_with_optuna = use_optuna and attack_obj.optuna_params > 0

            # If Optuna is used, the attack should not be loaded even if it already exists
            if run_with_optuna:
                logger.info(f"Preparing attack: {attack_type}")
                attack_obj.bayesian_optimization = True
                attack_obj.prepare_attack()
                logger.info(f"Running attack with Optuna: {attack_type}")
                study = attack_obj.run_with_optuna()
                best_config = attack_obj.configs.model_copy(update=study.best_params)
                attack_obj.reset_attack(best_config)
                attack_obj._hash_attack() # update hash with new config

            if not run_with_optuna:
                logger.info(f"Preparing attack: {attack_type}")
                attack_obj.bayesian_optimization = False
                attack_obj.prepare_attack()
            logger.info(f"Running attack: {attack_type}")
            result = attack_obj.run_attack()
            logger.info(f"Saving results for attack: {attack_type} to {self.report_dir}")
            result.save(attack_obj = attack_obj, output_dir = self.output_dir)
            results.append({"attack_type": attack_type, "attack_object": attack_obj, "result_object": result})

        return results

    def _preflight_extraction_results(self: Self) -> None:
        """Reject known output collisions before preparing or running extraction attacks."""
        result_ids = [attack.result_id for attack in self.attacks]
        duplicate_ids = sorted({result_id for result_id in result_ids if result_ids.count(result_id) > 1})
        if duplicate_ids:
            raise ValueError(f"Duplicate extraction result IDs in one audit: {duplicate_ids}")
        for attack in self.attacks:
            if attack.config.overwrite_results:
                continue
            result_dir = self.report_dir / attack.result_id
            data_path = self.data_object_dir / f"{attack.result_id}.json"
            if result_dir.exists() or data_path.exists():
                raise FileExistsError(
                    f"Extraction result {attack.result_id!r} already exists. "
                    "Set overwrite_results=true to replace it."
                )

    def map_setting_to_attacks(self:Self) -> None:
        """Identify relevant attacks based on adversary setting."""
        # TODO: Implement this mapping and remove attack list from configs
        pass
