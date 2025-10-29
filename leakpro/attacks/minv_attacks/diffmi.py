"""Implementation of the Diff-Mi attack."""
from typing import Any, Dict, Optional

import os
import math
import numpy as np
import torch
import kornia.augmentation as K
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader

from leakpro.attacks.minv_attacks.abstract_minv import AbstractMINV
from leakpro.attacks.utils.diffusion_handler import DiffMiHandler
from leakpro.input_handler.minv_handler import MINVHandler
from leakpro.input_handler.modality_extensions.image_metrics import ImageMetrics
from leakpro.reporting.minva_result import MinvResult
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger

from leakpro.utils.save_load import hash_config
from leakpro.attacks.utils.diff_mi.setup import DiffMiConfig, get_p_reg, top_n_pseudo_label_dataset
from leakpro.attacks.utils.diff_mi.attack_utils import calc_acc, calc_acc_std, get_PGD, Iterative_Image_Reconstruction, save_tensor_to_image

class AttackDiffMi(AbstractMINV):
    """Class that implements the DiffMi attack."""

    def __init__(self: Self, handler: MINVHandler = None, configs: dict = {}) -> None:
        """Initialize the Diff-Mi attack.

        Args:
        ----
            handler (MINVHandler): The input handler object.
            configs (dict): Configuration parameters for the attack.

        """
        logger.info("Configuring Diff-Mi attack")

        self.output_dir = handler.configs.audit.output_dir if handler.configs.audit.output_dir else "./leakpro_output/"
        self._configs_ = ["preprocessing", "pretrain", "finetune", "diffusion"]
        self._setup_paths_()

        configs = self._validate_configs(configs)
        self.config = DiffMiConfig(**configs)

        # Get hashes of the configurations. Make sure to do this before setting up paths and other.
        self.hashes = self._hash_configs()
        
        # Alter save_names to include hashes
        if self.config.hash_identifiable:
            logger.info(f"Using pretrain save_name: {self.hashes['pretrain']}")
            self.config.pretrain.save_name += f"_{self.hashes['pretrain']}"
            logger.info(f"Using finetune save_name: {self.hashes['finetune']}")
            self.config.finetune.save_name += f"_{self.hashes['finetune']}"

        # Set save path for diffusion models
        self.config.save_path = self.storage_path

        # Call the parent class constructor
        super().__init__(handler)

        self.diff_handler = DiffMiHandler(self.handler, configs=self.config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _setup_(self:Self) -> None:
        """Setup the attack."""
        logger.info("Setting up AttackDiffMi")

    def _hash_configs(self:Self) -> None:
        """Hash the configuration parameters."""
        hashes: Dict[str, Any] = {}
        configs = self.config.model_dump()

        hashes['attackhash'] = hash_config(configs)
        logger.info(f"Attack configuration hashed")
        for k in configs.keys():
            if isinstance(configs[k], dict):
                if k in self._configs_:
                    config_hash = hash_config(configs[k])[:8]
                    hashes[f"{k}"] = config_hash
                    logger.info(f"{k} configuration hashed")
        if self.config.hash_identifiable:
            logger.info(f"Attack is hash identifiable.")

        return hashes

    def _validate_configs(self:Self, configs) -> None:
        """Validate the configuration parameters."""
        for config in self._configs_:
            if config not in configs.keys():
                configs[config] = {}
            else:
                if not isinstance(configs[config], dict):
                    configs[config] = {}
        return configs

    def _setup_paths_(self:Self) -> None:
        """Setup storage paths for attack objects."""
        storage_path = self.output_dir
        if storage_path is not None:
            self.storage_path = f"{storage_path}/attack_objects/diffusion_models"
            if not os.path.exists(self.storage_path):
                # Create the folder
                os.makedirs(self.storage_path)
                logger.info(f"Created storage path at {self.storage_path} for diffusion models")
        else:
            self.storage_path = f"./leakpro_output/attack_objects/diffusion_models"

    def description(self:Self) -> dict:
        """Return the description of the attack."""

        title_str = "Diff-Mi Attack"
        reference_str = "Model Inversion Attacks Through Target-Specific Conditional \
                        Diffusion Models, Ouxiang et al. 2024, https://arxiv.org/abs/2407.11424"
        summary_str = "This attack is a model inversion attack that uses the Diff-Mi algorithm."
        detailed_str = "The Pseudo Label Guided Model Inversion Attack (Diff-Mi) is a white-box attack \
                        that implements pseudo-labels on a public dataset to construct a conditional GAN. \
                            Steps: \
                                1. Top-n selection of pseudo labels. \
                                2. Train the Diffusion model. \
                                3. (Optional) Fine-tune the Diffusion model. \
                                4. Generate samples from the Diffusion model and evaluate them. "
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }

    def prepare_attack(self:Self) -> None:
        """Prepare the attack."""

        # GET TARGET MODEL and EVALUATION MODEL (same as target model for now)
        self.target_model = self.handler.target_model
        self.evaluation_model = self.handler.target_model
        logger.info("Target model set.")

        # PUBLIC DATASET
        public_dataloader = self.handler.get_public_dataloader(batch_size=self.config.pretrain.batch_size)
        logger.info("Loading public dataloader finished.")

        # PSEUDO LABELED DATASET
        pseudo_dataset = top_n_pseudo_label_dataset(public_dataloader, self.target_model, self.device,
                                                    num_classes=self.handler.get_num_classes(),
                                                    top_n=self.config.preprocessing.top_n
                                                    )
        logger.info("Pseudo labeled dataset created.")

        pseudo_dataloader = DataLoader(pseudo_dataset, batch_size=self.config.pretrain.batch_size, shuffle=True)
        logger.info(f"Pseudo labeled dataloader created.")

        self.diff_handler.pseudo_dataloader = pseudo_dataloader
        logger.info("Pseudo labeled dataloader set in DiffMiHandler.")

        self.p_reg = get_p_reg(public_dataloader, self.target_model, self.device, args=self.config.preprocessing)
        logger.info("Done computing p_reg.")

        if self.config.do_fine_tune:
            self.diff_handler.target_model = self.target_model
            self.diff_handler.p_reg = self.p_reg

            self.diffusion_model = self.diff_handler.get_finetuned()
        else:
            self.diffusion_model = self.diff_handler.get_pretrained()

    def run_attack(self:Self) -> MinvResult:
        """Run the attack."""
        logger.info("Running the Diff-Mi attack")
        pgd_model = get_PGD(self.target_model)

        recon_list, success_img_list, success_label_list = [], [], []
        labels = torch.cat([torch.randperm(self.config.attack.label_num) for _ in range(self.config.attack.repeat_N)]).to(self.device)
        label_dataset = DataLoader(labels, batch_size=self.config.attack.batch_size, shuffle=False)
        batch_num = math.ceil(len(labels) / self.config.attack.batch_size) - 1

        self.attack.aug = K.container.ImageSequential(
            K.RandomHorizontalFlip(),
            K.ColorJitter(brightness=0.2, p=0.5),
            K.RandomGaussianBlur((7, 7), (3, 3), p=0.5),
        )

        for i, classes in enumerate(label_dataset):
            classes = classes.to(self.device)
            recon_imgs = Iterative_Image_Reconstruction(self.config.attack, diff_net=self.diffusion_model,  classifier=self.target_model, classes=classes,
                                                        iter=i, batch_num=batch_num, device=self.device).clamp(0,1).to(device=self.device)

            img_translated = pgd_model(recon_imgs, target=classes, **self.config.attack.pgdconfig.__dict__)[-1].clamp(0,1)
            _, _, idx = calc_acc(self.evaluation_model, K.Resize((112, 112))(img_translated), classes, with_success=True)

            recon_list.append(img_translated)
            success_img_list.append(img_translated[idx])
            success_label_list.append(classes[idx])
        recon_list = torch.cat(recon_list)
        success_img_list = torch.cat(success_img_list)
        success_label_list = torch.cat(success_label_list)

        acc1, acc5, var1, var5 = calc_acc_std(recon_list, labels, self.evaluation_model, self.config.attack.label_num)
        logger.info(f"Final Top1: {acc1:.2%} ± {var1:.2%}, Top5: {acc5:.2%} ± {var5:.2%}")

        # Save Reconstructed Images
        recon_paths = save_tensor_to_image(recon_list, labels, f'./{self.output_dir}/results/Diff-Mi/all_imgs')
        success_img_paths = save_tensor_to_image(success_img_list, success_label_list, f'./{self.output_dir}/results/Diff-Mi/success_imgs')
        print(f'Saved {self.config.attack.repeat_N}x{self.config.attack.label_num} generated images.')

        if self.config.attack.cal_fid:
            calc_pytorch_fid(recon_paths)

        if self.config.attack.cal_knn:
            calc_knn(success_img_list, success_label_list, E=self.evaluation_model, device=self.device)


    def reconstruction(self:Self):
        pass

    @staticmethod
    def get_attack() -> str:
        """Return the name of the attack."""
        return "Diff-Mi"