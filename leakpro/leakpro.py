"""Main class for LeakPro."""

import inspect
import types
from pathlib import Path

import yaml

from leakpro.attacks.attack_scheduler import AttackScheduler
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.input_handler.minv_handler import MINVHandler
from leakpro.input_handler.modality_extensions.image_extension import ImageExtension
from leakpro.input_handler.modality_extensions.tabular_extension import TabularExtension
from leakpro.schemas import LeakProConfig
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import add_file_handler, logger

modality_extensions = {"tabular": TabularExtension,
                       "image":ImageExtension,
                       "text":None,
                       "graph":None,
                       "timeseries":None}

class LeakPro:
    """Main class for LeakPro."""

    def __init__(self:Self, user_input_handler:AbstractInputHandler, configs_path:str) -> None:

        assert issubclass(user_input_handler, AbstractInputHandler), "handler must be an instance of AbstractInputHandler"

        # Read configs from path and ensure it adheres to the schema
        try:
            with open(configs_path, "rb") as f:
                configs = yaml.safe_load(f)
                configs = LeakProConfig(**configs)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File {configs_path} not found") from e

        # Create report directory
        self.report_dir = f"{configs.audit.output_dir}/results"
        Path(self.report_dir).mkdir(parents=True, exist_ok=True)

        # Set folder for logger
        log_path = f"{configs.audit.output_dir}/{logger.name}.log"
        add_file_handler(logger, log_path)

        # Initialize handler and attack scheduler
        self.handler = self.setup_handler(user_input_handler, configs)
        self.attack_scheduler = AttackScheduler(self.handler)

    def setup_handler(self:Self, user_input_handler:AbstractInputHandler, configs:dict) -> None:
        """Prepare the handler using dynamic composition to merge the user-input handler and modality extension."""

        if configs.audit.attack_type == "mia":
            handler = MIAHandler(configs)
        elif configs.audit.attack_type == "minv":
            handler = MINVHandler(configs)

            # Attach train_gan method explicitly if it exists in user_input_handler
            if hasattr(user_input_handler, "train_gan"):
                handler.train_gan = types.MethodType(user_input_handler.train_gan, handler)

        else:
            raise ValueError(f"Unknown attack type: {configs.audit.attack_type}")

        # Attach methods to Handler explicitly defined in AbstractInputHandler from user_input_handler
        for name, _ in inspect.getmembers(AbstractInputHandler, predicate=inspect.isfunction):
            if hasattr(user_input_handler, name) and not name.startswith("__"):
                attr = getattr(user_input_handler, name)
                if callable(attr):
                    attr = types.MethodType(attr, handler) # ensure to properly bind methods to handler
                setattr(handler, name, attr)



        # Load extension class and initiate it using the handler (allows for two-way communication)
        modality_extension_instance = modality_extensions[configs.audit.data_modality]
        if modality_extension_instance is not None:
            handler.modality_extension = modality_extension_instance(handler)
        else:
            handler.modality_extension = None
        return handler

    def run_audit(self:Self, return_results: bool = False, use_optuna: bool = False) -> None:
        """Run the audit."""

        audit_results = self.attack_scheduler.run_attacks(use_optuna=use_optuna)
        results = [] if return_results else None

        for attack_name in audit_results:
            logger.info(f"Preparing results for attack: {attack_name}")

            # Return the result object, dont save it
            if return_results:

                result = audit_results[attack_name]["result_object"]
                result.attack_name = attack_name
                result.configs = self.handler.configs.audit

                # Append
                results.append(result)
            else:
                result = audit_results[attack_name]["result_object"]
                result.save(name=attack_name, path=self.report_dir, config=self.handler.configs.audit)

        logger.info("Auditing completed")
        return results
