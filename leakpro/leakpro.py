"""Main class for LeakPro."""

from pathlib import Path

import yaml

from leakpro.attacks.attack_scheduler import AttackScheduler
from leakpro.import_helper import Self
from leakpro.user_inputs.abstract_input_handler import AbstractInputHandler
from leakpro.utils.handler_logger import setup_log


class LeakPro:
    """Main class for LeakPro."""

    def __init__(self:Self, handler:AbstractInputHandler, configs_path:str) -> None:

        assert issubclass(handler, AbstractInputHandler), "handler must be an instance of AbstractInputHandler"

        # Read configs
        try:
            with open(configs_path, "rb") as f:
                configs = yaml.safe_load(f)
                assert isinstance(configs, dict), "configs must be a dictionary"
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File {configs_path} not found") from e

        # Create report directory
        report_dir = f"{configs['audit']['output_dir']}/results"
        Path(report_dir).mkdir(parents=True, exist_ok=True)


        # Set up logger
        logger = setup_log("LeakPro", save_file=True)

        # Initialize handler and attack scheduler
        self.handler = handler(configs, logger)
        self.attack_scheduler = AttackScheduler(self.handler)

    def run_audit(self:Self) -> None:
        """Run the audit."""
        audit_results = self.attack_scheduler.run_attacks()

        for attack_name in audit_results:
            self.handler.logger.info(f"Preparing results for attack: {attack_name}")

            self.handler.prepare_privacy_risk_report(
                audit_results[attack_name]["result_object"],
                self.handler.configs["audit"],
                save_path=f"{self.handler.configs['audit']['report_log']}/{attack_name}",
            )

