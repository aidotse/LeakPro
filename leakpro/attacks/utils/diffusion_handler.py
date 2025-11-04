"""Module for handling Diffusion models from Diff-MI."""
import os

import torch
from torch.nn import Module

from leakpro.attacks.utils.diff_mi.resample import create_named_schedule_sampler
from leakpro.attacks.utils.diff_mi.script_util import create_gaussian_diffusion, create_model, diffusion_defaults, model_defaults
from leakpro.attacks.utils.diff_mi.setup import DiffMiConfig, args_to_dict
from leakpro.attacks.utils.diff_mi.train_util import FineTune, PreTrain
from leakpro.input_handler.minv_handler import MINVHandler
from leakpro.input_handler.user_imports import get_class_from_module, import_module_from_file
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class DiffMiHandler():
    def __init__(self: Self,
                 handler: MINVHandler = None,
                 configs: DiffMiConfig = None
        ) -> None:
        """Initialize the DiffMiHandler class.

        Args:
        ----
            handler (MINVHandler): The MINVHandler object.
            configs (DiffMiConfig): The DiffMiConfig object.

        Returns:
        -------
            None

        """
        logger.info("Initializing DiffMiHandler...")
        self.handler = handler
        self.trained_bool = False
        self.configs = configs
        self._setup_diffusion_configs(self.configs)

        logger.info("Initialized DiffMiHandler")

    def _setup_diffusion_configs(self: Self, configs: DiffMiConfig) -> None:
        """Load diffusion-specific configurations (e.g., diffusion path, params).

        Args:
        ----
            self: Self
            configs: DiffMiConfig

        Returns:
        -------
            None

        """
        logger.info("Setting up diffusion configurations")
        self.diffusion_path = configs.diffusion.module_path
        self.diffusion_model_class = configs.diffusion.model_class
        self.gaussian_diffusion_class = configs.diffusion.diffusion_class

        logger.info(f"Diffusion path: {self.diffusion_path}, Diffusion class: {self.diffusion_model_class}")

    def _init_model(self: Self, _configs_: DiffMiConfig) -> None:
        """Initialize the diffusion model.

        Args:
        ----
            self: Self
            _configs_: DiffMiConfig

        Returns:
        -------
            None

        """
        logger.info("Initializing diffusion model")

        diffusion_model_params = args_to_dict(_configs_, model_defaults().keys())
        gaussian_diffusion_params = args_to_dict(_configs_, diffusion_defaults().keys())

        if self.diffusion_path and self.diffusion_model_class and self.gaussian_diffusion_class:
            self.diffusion_model_blueprint = self._import_model_from_path(self.diffusion_path, self.diffusion_model_class)
            self.gaussian_diffusion_blueprint = self._import_model_from_path(self.diffusion_path, self.gaussian_diffusion_class)
            logger.info(f"Imported Diffusion model blueprint and Gaussian diffusion from {self.diffusion_model_class} and {self.gaussian_diffusion_class} in {self.diffusion_path}")

            self.diffusion_model = self.diffusion_model_blueprint(**diffusion_model_params)
            self.diffusion = self.gaussian_diffusion_blueprint(**gaussian_diffusion_params)

        elif not self.diffusion_path and self.diffusion_model_class == "UNetModel":
            logger.info("Using default Diffusion UNet model from Diff-MI.")

            self.diffusion_model = create_model(
                **diffusion_model_params
            )
            self.diffusion = create_gaussian_diffusion(
                **gaussian_diffusion_params
            )
        else:
            logger.warning("Diffusion path or class is not set or is invalid.")
            logger.info(f"Diffusion path: {self.diffusion_path}")
            logger.info(f"Diffusion class: {self.diffusion_model_class}")
            logger.info(f"Gaussian diffusion class: {self.gaussian_diffusion_class}")
            logger.warning("Using defaults from Diff-MI instead, UNet and SpacedDiffusion model.")
            self.diffusion_model = create_model(
                **diffusion_model_params
            )
            self.diffusion = create_gaussian_diffusion(
                **gaussian_diffusion_params
            )

    def get_finetuned(self) -> Module:
        """Return the fine-tuned diffusion model.

        Args:
        ----
            self: Self

        Returns:
        -------
            Module: The fine-tuned diffusion model.

        """
        # Re-initialize model with fine-tuning configs
        self._init_model(_configs_ = self.configs.finetune)

        fine_tune_path = f"{self.configs.save_path}/{self.configs.finetune.save_name}.pt"

        if os.path.exists(fine_tune_path):

            logger.info(f"Loading fine-tuned diffusion model from {fine_tune_path}")
            self.diffusion_model.load_state_dict(torch.load(fine_tune_path))
        else:
            logger.warning("Fine-tuned path not found or doesn't exist")
            logger.info("Loading Pre-trained model to fine-tune...")
            self.get_pretrained()
            self.fine_tune()
            self.diffusion_model.load_state_dict(torch.load(fine_tune_path))
        return self.diffusion_model


    def get_pretrained(self) -> Module:
        """Return the pre-trained diffusion model.

        Args:
        ----
            self: Self

        Returns:
        -------
            Module: The pre-trained diffusion model.

        """
        # Init pretrain model
        self._init_model(_configs_ = self.configs.pretrain)

        pre_trained_path = f"{self.configs.save_path}/{self.configs.pretrain.save_name}.pt"

        if os.path.exists(pre_trained_path):
            logger.info(f"Loading pre-trained diffusion model from {pre_trained_path}")
            self.diffusion_model.load_state_dict(torch.load(pre_trained_path))
        else:
            logger.warning("Pre-trained path not found or doesn't exist")
            self.pre_train()
            self.diffusion_model.load_state_dict(torch.load(pre_trained_path))
        return self.diffusion_model

    def pre_train(self) -> None:
        """Pre-train the diffusion model."""
        logger.info("Pre-training diffusion model...")
        schedule_sampler = create_named_schedule_sampler(self.configs.pretrain.schedule_sampler, self.diffusion)
        PreTrain(
            model=self.diffusion_model,
            diffusion=self.diffusion,
            data=self.pseudo_dataloader,
            args=self.configs.pretrain,
            save_path=self.configs.save_path,
            schedule_sampler=schedule_sampler,
        ).run()


    def fine_tune(self) -> None:
        """Fine-tune the diffusion model.

        Args:
        ----
            self: Self

        Returns:
        -------
            None

        """
        logger.info("Fine-tuning diffusion model...")
        # Diffusion-specific fine-tuning logic would be implemented here.
        schedule_sampler = create_named_schedule_sampler(self.configs.finetune.schedule_sampler, self.diffusion)
        FineTune(
            target=self.target_model,
            model=self.diffusion_model,
            diffusion=self.diffusion,
            args=self.configs.finetune,
            p_reg=self.p_reg,
            save_path=self.configs.save_path,
            schedule_sampler=schedule_sampler,
        ).run()

    def _import_model_from_path(self:Self, module_path:str, model_class:str)->None:
        """Import the model from the given path.

        Args:
        ----
            module_path (str): The path to the module.
            model_class (str): The name of the model class.

        """
        try:
            module = import_module_from_file(module_path)
            return get_class_from_module(module, model_class)
        except Exception as e:
            logger.error(f"Failed to create model blueprint from {model_class} in {module_path}")
            logger.error(f"{e}")

    def sample_from_diffusion_model(self,
                                batch_size: int = None,
                                label: int = None,
                                ) -> tuple:
        """Samples data from a given diffusion model.

        Args:
            batch_size (int): The number of samples to generate.
            label (int): The optional class label to generate samples for, otherwise random.

        Returns:
            tuple: A tuple containing the generated samples, the class labels.

        """
        pass

    def save_diffusion_model(self, diffusion_model: Module, path: str) -> None:
        """Save the diffusion model.

        Args:
        ----
            diffusion_model (Module): The diffusion model to save.
            path (str): The path to save the model.

        Returns:
        -------
            None

        """
        torch.save(diffusion_model.state_dict(), path)
