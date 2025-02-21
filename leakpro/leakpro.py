"""Main class for LeakPro."""

import types
from pathlib import Path

import yaml

from leakpro.attacks.attack_scheduler import AttackScheduler
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.input_handler.handler_setup import (
    _load_model_class,
    _load_population,
    _load_target_metadata,
    _load_trained_target_model,
    _validate_indices,
    _validate_target_metadata,
    get_dataloader,
    get_dataset,
    get_labels,
    get_population_size,
    get_target_model,
    get_target_model_blueprint,
    get_target_model_metadata,
    get_target_replica,
    get_test_indices,
    get_train_indices,
    set_target_model,
    set_target_model_blueprint,
    set_target_model_metadata,
    set_test_indices,
    set_train_indices,
    setup,
)
from leakpro.input_handler.modality_extensions.tabular_extension import TabularExtension
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import add_file_handler, logger

modality_extensions = {"tabular": TabularExtension,
                       "image":None,
                       "text":None,
                       "graph":None}

class LeakPro:
    """Main class for LeakPro."""

    def __init__(self:Self, handler_class:AbstractInputHandler, configs_path:str) -> None:

        assert issubclass(handler_class, AbstractInputHandler), "handler must be an instance of AbstractInputHandler"

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
        self.handler = self.setup_handler(handler_class, configs)

        self.attack_scheduler = AttackScheduler(self.handler)

    def setup_handler(self:Self, handler_class:AbstractInputHandler, configs:dict) -> None:
        """Prepare the handler."""

        # Get the modality extension class
        if "modality" in configs["audit"] and configs["audit"]["modality"].lower() in modality_extensions:
            extension_class = modality_extensions[configs["audit"]["modality"].lower()]
        else:
            extension_class = None

        if extension_class is not None:
            # Initialize handler from both user input and extension classes
            ExtendedHandlerClass = type("ExtendedHandler", (handler_class, extension_class), {})  # noqa: N806
            handler = ExtendedHandlerClass.__new__(ExtendedHandlerClass)
        else:
            handler = handler_class.__new__(handler_class)

        handler_class.__init__(handler, configs)

        # Attach properties to handler
        handler.target_model_blueprint = property(types.MethodType(get_target_model_blueprint, handler),
                                                       types.MethodType(set_target_model_blueprint, handler))
        handler.target_model = property(types.MethodType(get_target_model, handler),
                                             types.MethodType(set_target_model, handler))
        handler.target_model_metadata = property(types.MethodType(get_target_model_metadata, handler),
                                                      types.MethodType(set_target_model_metadata, handler))
        handler.population_size = property(types.MethodType(get_population_size, handler))
        handler.test_indices = property(types.MethodType(get_test_indices, handler),
                                           types.MethodType(set_test_indices, handler))
        handler.train_indices = property(types.MethodType(get_train_indices, handler),
                                            types.MethodType(set_train_indices, handler))

        # attach provided methods to handler as read-only properties
        handler.criterion = property(types.MethodType(handler.get_criterion, handler))
        self.optimizer = property(types.MethodType(handler.get_optimizer, handler))

        # Attach functionality and properties to handler
        handler.get_target_replica = types.MethodType(get_target_replica, handler)
        handler.get_dataset = types.MethodType(get_dataset, handler)
        handler.get_dataloader = types.MethodType(get_dataloader, handler)
        handler.get_labels = types.MethodType(get_labels, handler)

        # Attach setup methods to handler
        handler.setup = types.MethodType(setup, handler)
        handler._load_model_class = types.MethodType(_load_model_class, handler)
        handler._load_population = types.MethodType(_load_population, handler)
        handler._load_target_metadata = types.MethodType(_load_target_metadata, handler)
        handler._load_trained_target_model = types.MethodType(_load_trained_target_model, handler)
        handler._validate_indices = types.MethodType(_validate_indices, handler)
        handler._validate_target_metadata = types.MethodType(_validate_target_metadata, handler)

        # Load population data, target model, and target model metadata
        handler.setup()
        if extension_class is not None:
            extension_class.__init__(handler)

        return handler

    def run_audit(self:Self, return_results: bool = False) -> None:
        """Run the audit."""
        audit_results = self.attack_scheduler.run_attacks()
        results = [] if return_results else None

        for attack_name in audit_results:
            logger.info(f"Preparing results for attack: {attack_name}")

            if return_results:

                result = audit_results[attack_name]["result_object"]
                result.attack_name = attack_name
                result.configs = self.handler.configs["audit"]

                # Append
                results.append(result)
            else:
                result = audit_results[attack_name]["result_object"]
                result.save(name=attack_name, path=self.report_dir, config=self.handler.configs["audit"])

        logger.info("Auditing completed")
        return results
