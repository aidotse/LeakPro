"""Main class for LeakPro."""

import inspect
import types
from pathlib import Path

import yaml
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from leakpro.attacks.attack_scheduler import AttackScheduler
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.input_handler.minv_handler import MINVHandler
from leakpro.input_handler.modality_extensions.image_extension import ImageExtension
from leakpro.input_handler.modality_extensions.tabular_extension import TabularExtension
from leakpro.schemas import EvalOutput, LeakProConfig, MIAMetaDataSchema, TrainingOutput
from leakpro.utils.conversion import _dataloader_to_config, _get_model_init_params, _loss_to_config, _optimizer_to_config
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
        """Initialize LeakPro.

        Args:
        ----
            user_input_handler (AbstractInputHandler): The user-defined input handler
            configs_path (str): The path to the configuration file

        """

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
        """Prepare the handler using dynamic composition to merge the user-input handler and modality extension.

        Args:
        ----
            user_input_handler (AbstractInputHandler): The user-defined input handler
            configs (dict): The configuration object

        """

        if configs.audit.attack_type == "mia":
            handler = MIAHandler(configs, user_input_handler)

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

    @staticmethod
    def make_mia_metadata(train_result: TrainingOutput,
                          optimizer: Optimizer,
                          loss_fn: Module,
                          dataloader: DataLoader,
                          test_result: EvalOutput,
                          epochs: int,
                          train_indices:list,
                          test_indices:list,
                          dataset_name:str) -> MIAMetaDataSchema:
        """Create metadata for the MIA attack.

        Args:
        ----
            train_result (TrainingOutput): The result of the model evaluation on the training set
            optimizer (Optimizer): The optimizer used to train "model"
            loss_fn (Module): The loss function used to train "model"
            epochs (int): The number of epochs used to train the model
            dataloader (DataLoader): The dataloader used to train "model"
            test_result (EvalOutput): The result of the model evaluation on the test set
            train_indices (list[int]): The indices of the training set
            test_indices (list[int]): The indices of the test set
            dataset_name (str): The name of the dataset

        Returns:
        -------
            MIAMetaDataSchema: The metadata for the MIA attack

        """

        return MIAMetaDataSchema(
            init_params=_get_model_init_params(train_result.model),
            optimizer=_optimizer_to_config(optimizer),
            loss=_loss_to_config(loss_fn),
            data_loader=_dataloader_to_config(dataloader),
            epochs=epochs,
            train_indices=train_indices,
            test_indices=test_indices,
            num_train=len(train_indices),
            dataset=dataset_name,
            train_result=train_result.metrics,
            test_result=test_result
        )

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
