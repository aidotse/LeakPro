"""Main class for LeakPro."""

import types
from pathlib import Path

import yaml

from leakpro.attacks.attack_scheduler import AttackScheduler
from leakpro.reporting.utils import prepare_privacy_risk_report
from leakpro.user_inputs.abstract_input_handler import AbstractInputHandler
from leakpro.user_inputs.handler_setup import (
    _load_model_class,
    _load_population,
    _load_target_metadata,
    _load_trained_target_model,
    _validate_indices,
    _validate_target_metadata,
    get_dataloader,
    get_dataset,
    get_labels,
    get_optimizer,
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
from leakpro.user_inputs.modality_extensions.image_extension import ImageExtension
from leakpro.user_inputs.modality_extensions.tabular_extension import TabularExtension
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import add_file_handler, logger

modality_extensions = {"tabular": TabularExtension,
                       "image":ImageExtension,
                       "text":None,
                       "graph":None}

attack_types = ["mia", "gia"]

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

        attack_type = handler.configs["audit"]["attack_type"]
        assert attack_type in attack_types, f"attack_type must be one of {attack_types}"

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

        # if FL: Enable the handler to map to meta-optimizer from target metadata
        if attack_type == "gia":
            handler.get_optimizer = types.MethodType(get_optimizer, handler)

        # gia: Load local training data (as population), global model (as target model and metadata)
        # Note: we need to train the client model ourselves if we want to use batch statistics etc.
        # mia: Load population data, target model, and target model metadata
        handler.setup()

        if extension_class is not None:
            extension_class.__init__(handler)

        return handler

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
