"""Implementation of the Diff-Mi attack."""
import math
import os
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakpro.attacks.minv_attacks.abstract_minv import AbstractMINV
from leakpro.attacks.utils.diff_mi.attack_utils import (
    Iterative_Reconstruction,
    calc_acc,
    calc_acc_std,
    calc_knn,
    calc_lpips,
    calc_mse,
    calc_pytorch_fid,
    get_PGD,
    save_tensor,
)
from leakpro.attacks.utils.diff_mi.setup import DiffMiConfig, extract_features, get_p_reg, top_n_pseudo_label_dataset
from leakpro.attacks.utils.diffusion_handler import DiffMiHandler
from leakpro.input_handler.minv_handler import MINVHandler
from leakpro.reporting.minva_result import MinvResult
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger
from leakpro.utils.save_load import hash_config


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

        hashes["attackhash"] = hash_config(configs)
        logger.info("Attack configuration hashed")
        for k in configs.keys():
            if isinstance(configs[k], dict):
                if k in self._configs_:
                    config_hash = hash_config(configs[k])[:8]
                    hashes[f"{k}"] = config_hash
                    logger.info(f"{k} configuration hashed")
        if self.config.hash_identifiable:
            logger.info("Attack is hash identifiable.")

        return hashes

    def _validate_configs(self:Self, configs) -> None:
        """Validate the configuration parameters."""
        for config in self._configs_:
            if config not in configs.keys() or not isinstance(configs[config], dict):
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
            self.storage_path = "./leakpro_output/attack_objects/diffusion_models"

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
        self.target_model = self.handler.target_model.to(device=self.device)
        logger.info("Target model set.")

        # EVALUATION MODEL
        self.evaluation_model = self.handler.target_model.to(device=self.device)
        logger.info("Evaluation model set.")

        # Private DATASET
        self.private_dataloader = self.handler.get_private_dataloader(batch_size=self.config.pretrain.batch_size)
        logger.info("Loading private dataloader finished.")

        self.private_features, self.private_idents = extract_features(self.target_model, self.private_dataloader, self.device, save_dir=self.config.save_path)
        logger.info("Extracted private features from target model.")

        # PUBLIC DATASET
        self.public_dataloader = self.handler.get_public_dataloader(batch_size=self.config.pretrain.batch_size)
        logger.info("Loading public dataloader finished.")

        # PSEUDO LABELED DATASET
        pseudo_dataset = top_n_pseudo_label_dataset(self.public_dataloader, self.target_model, self.device,
                                                    num_classes=self.handler.get_num_classes(),
                                                    top_n=self.config.preprocessing.top_n
                                                    )
        logger.info("Pseudo labeled dataset created.")

        self.pseudo_dataloader = DataLoader(pseudo_dataset, batch_size=self.config.pretrain.batch_size, shuffle=True)
        logger.info("Pseudo labeled dataloader created.")

        self.diff_handler.pseudo_dataloader = self.pseudo_dataloader
        logger.info("Pseudo labeled dataloader set in DiffMiHandler.")

        self.p_reg = get_p_reg(self.public_dataloader, self.target_model, self.device, args=self.config)
        logger.info("Done computing p_reg.")

        if self.config.do_fine_tune:
            self.diff_handler.target_model = self.target_model
            self.diff_handler.p_reg = self.p_reg

            self.diffusion_model = self.diff_handler.get_finetuned().to(device=self.device)
        else:
            self.diffusion_model = self.diff_handler.get_pretrained().to(device=self.device)

    def run_attack(self:Self) -> MinvResult:
        """Run the attack."""
        logger.info("Running the Diff-Mi attack")
        pgd_model = get_PGD(self.target_model)

        recon_list, success_list, success_label_list = [], [], []
        labels = torch.cat([torch.randperm(self.config.diffmiattack.label_num) for _ in range(self.config.diffmiattack.repeat_N)]).to(self.device)
        label_dataset = DataLoader(labels, batch_size=self.config.diffmiattack.batch_size, shuffle=False)
        batch_num = math.ceil(len(labels) / self.config.diffmiattack.batch_size) - 1

        logger.info("Running reconstruction... ")
        for i, classes in tqdm(enumerate(label_dataset), total=len(label_dataset)):
            classes = classes.to(self.device)
            recon = Iterative_Reconstruction(args=self.config.diffmiattack, diff_net=self.diffusion_model,  classifier=self.target_model, classes=classes,
                                                        p_reg=self.p_reg, iter=i, batch_num=batch_num, device=self.device).clamp(0,1).to(device=self.device)

            translated = pgd_model(recon, target=classes, **self.config.diffmiattack.pgdconfig.__dict__)[-1].clamp(0,1)
            _, _, idx = calc_acc(self.evaluation_model, translated, classes, with_success=True)

            recon_list.append(translated)
            success_list.append(translated[idx])
            success_label_list.append(classes[idx])

        recon_list = torch.cat(recon_list)
        success_list = torch.cat(success_list)
        success_label_list = torch.cat(success_label_list)

        acc1, acc5, var1, var5 = calc_acc_std(recon_list, labels, self.evaluation_model, self.config.diffmiattack.label_num)
        logger.info(f"Final Top1: {acc1:.2%} ± {var1:.2%}, Top5: {acc5:.2%} ± {var5:.2%}")

        # Save Reconstructed Images
        logger.info(f"Saved {self.config.diffmiattack.repeat_N}x{self.config.diffmiattack.label_num} generated images.")

        recon_path = f"{self.output_dir}/results/Diff-Mi/all_recreated"
        recon_paths = save_tensor(recon_list, labels, recon_path)
        logger.info(f"All recreated images saved at: {recon_path}")

        success_path = f"{self.output_dir}/results/Diff-Mi/success_recreated"
        success_paths = save_tensor(success_list, success_label_list, success_path)
        logger.info(f"Successful recreated images saved at: {success_path}")

        # Calculate additional metrics
        knn, knn_arr = np.nan, []
        value_a, value_v = np.nan, np.nan
        avg_mse, mse_per_label, mse_arr, mse_min_fake, mse_min_real = np.nan, [], [], [], []
        fid_value = np.nan

        if self.config.diffmiattack.calc_knn:
            logger.info("Calculating KNN Distance...")
            knn, knn_arr = calc_knn(success_list, success_label_list,
                     private_feats=self.private_features,
                     private_idents=self.private_idents,
                     evaluation_model=self.evaluation_model, device=self.device)
            logger.info(f"KNN Dist computed on {success_list.shape[0]} attack samples: {knn:.2f}")

        if self.config.diffmiattack.calc_lpips:
            logger.info("Calculating LPIPS...")
            value_a, value_v = calc_lpips(private_data=self.private_dataloader, fakes=success_list, fake_targets=success_label_list, anno="", device="cuda")
            logger.info(f"LPIPS Alex: {value_a:.4f}, VGG: {value_v:.4f}")

        if self.config.diffmiattack.calc_mse:
            logger.info("Calculating MSE...")
            avg_mse, mse_per_label, mse_arr, mse_min_fake, mse_min_real = calc_mse(
                private_data=self.private_dataloader,
                fakes=success_list,
                fake_labels=success_label_list,
                device=self.device,
            )
            logger.info(f"Average MSE: {avg_mse:.4f}")

        if self.config.diffmiattack.calc_fid:
            logger.info("Calculating FID...")
            fid_value = calc_pytorch_fid(recon_list, self.private_dataloader)
            logger.info(f"FID score: {fid_value:.2f}")

        return MinvResult.from_metrics(
            result_name="Diff-Mi Attack Result",
            result_id=self.hashes["attackhash"][:8],
            config=self.config,
            metrics={
                "top1_accuracy": acc1,
                "top5_accuracy": acc5,
                "knn": knn,
                "knn_arr": knn_arr,
                "lpips_alex": value_a,
                "lpips_vgg": value_v,
                "mse": avg_mse,
                "fid": fid_value,
                },
            )
