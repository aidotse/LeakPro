"""Main class for LeakPro."""

from pathlib import Path

import yaml

from leakpro.attacks.attack_scheduler import AttackScheduler
from leakpro.import_helper import Self
from leakpro.user_inputs.abstract_input_handler import AbstractInputHandler
from leakpro.utils.logger import add_file_handler, logger
from leakpro.reporting.utils import prepare_privacy_risk_report

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
        self.report_dir = f"{configs['audit']['output_dir']}/results"
        Path(self.report_dir).mkdir(parents=True, exist_ok=True)

        # Set folder for logger
        log_path = f"{configs['audit']['output_dir']}/{logger.name}.log"
        add_file_handler(logger, log_path)

        # Initialize handler and attack scheduler
        self.handler = handler(configs)
        self.attack_scheduler = AttackScheduler(self.handler)

    def run_audit(self:Self) -> None:
        """Run the audit."""
        audit_results = self.attack_scheduler.run_attacks()

        for attack_name in audit_results:
            logger.info(f"Preparing results for attack: {attack_name}")

            prepare_privacy_risk_report(
                audit_results[attack_name]["result_object"],
                self.handler.configs["audit"],
                save_path=f"{self.report_dir}/{attack_name}",
            )

        logger.info("Auditing completed")