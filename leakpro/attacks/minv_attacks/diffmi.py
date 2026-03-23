"""Implementation of the Diff-Mi attack."""
import math
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakpro.attacks.minv_attacks.abstract_minv import AbstractMINV
from leakpro.attacks.utils.diff_mi.attack_utils import (
    calc_acc,
    calc_acc_std,
    calc_knn,
    calc_lpips,
    calc_mse,
    calc_pytorch_fid,
    get_pgd,
    iterative_reconstruction,
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

    def __init__(self: Self, handler: MINVHandler, configs: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the Diff-Mi attack.

        Args:
        ----
            handler (MINVHandler): The input handler object.
            configs (dict): Configuration parameters for the attack.

        """
        logger.info("Configuring Diff-Mi attack")

        if handler is None:
            raise ValueError("handler must be provided")
        self.output_dir = handler.configs.audit.output_dir if handler.configs.audit.output_dir else "./leakpro_output/"
        self._configs_ = ["preprocessing", "pretrain", "finetune", "diffusion"]
        self._setup_paths_()

        configs = self._validate_configs(configs)
        self.config = DiffMiConfig(**configs)

        if self.config.pretrain.save_name is None:
            self.config.pretrain.save_name = "pretrain"
        if self.config.finetune.save_name is None:
            self.config.finetune.save_name = "finetune"

        # Get hashes of the configurations. Make sure to do this before setting up paths and other.
        self.hashes = self._hash_configs()

        # Alter save_names to include hashes
        if self.config.hash_identifiable:
            if self.config.pretrain.save_name is None or self.config.finetune.save_name is None:
                raise ValueError("save_name must be set when hash_identifiable is True")
            logger.info(f"Using pretrain save_name: {self.hashes['pretrain']}")
            self.config.pretrain.save_name += f"_{self.hashes['pretrain']}"
            logger.info(f"Using finetune save_name: {self.hashes['finetune']}")
            self.config.finetune.save_name += f"_{self.hashes['finetune']}"

        # Set save path for diffusion models
        if not self.config.save_path:
            self.config.save_path = self.storage_path
        os.makedirs(self.config.save_path, exist_ok=True)

        # Call the parent class constructor
        super().__init__(handler)

        self.diff_handler = DiffMiHandler(self.handler, configs=self.config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _setup_(self:Self) -> None:
        """Setup the attack."""
        logger.info("Setting up AttackDiffMi")

    def _hash_configs(self:Self) -> Dict[str, Any]:
        """Hash the configuration parameters."""
        hashes: Dict[str, Any] = {}
        configs = self.config.model_dump()

        hashes["attackhash"] = hash_config(configs)
        logger.info("Attack configuration hashed")
        for k, cfg in configs.items():
            if isinstance(cfg, dict) and k in self._configs_:
                config_hash = hash_config(cfg)[:8]
                hashes[k] = config_hash
                logger.info(f"{k} configuration hashed")
        if self.config.hash_identifiable:
            logger.info("Attack is hash identifiable.")

        return hashes


    def _validate_configs(self:Self, configs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate the configuration parameters."""
        configs = {} if configs is None else configs
        for config in self._configs_:
            if config not in configs or not isinstance(configs[config], dict):
                configs[config] = {}
        if "diffmiattack" not in configs or not isinstance(configs["diffmiattack"], dict):
            configs["diffmiattack"] = {}
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

        self.private_features, self.private_idents = extract_features(
            self.target_model,
            self.private_dataloader,
            self.device,
            save_dir=self.config.save_path or self.storage_path,
        )
        logger.info("Extracted private features from target model.")

        # PUBLIC DATASET
        self.public_dataloader = self.handler.get_public_dataloader(batch_size=self.config.pretrain.batch_size)
        logger.info("Loading public dataloader finished.")

        self.num_classes = self.handler.get_num_classes()
        if self.num_classes < 1:
            raise ValueError("Target model must expose at least one class for Diff-MI.")
        model_num_classes = self.num_classes + 1
        self.config.pretrain.num_classes = model_num_classes
        self.config.finetune.num_classes = model_num_classes
        try:
            sample_batch = next(iter(self.private_dataloader))[0]
            self.data_channels = int(sample_batch.shape[1])
            self.data_height = int(sample_batch.shape[2])
            self.data_width = int(sample_batch.shape[3])
        except StopIteration as exc:
            raise ValueError("Private dataloader is empty; cannot run Diff-MI attack.") from exc

        # PSEUDO LABELED DATASET
        pseudo_dataset = top_n_pseudo_label_dataset(self.public_dataloader, self.target_model, self.device,
                                                    num_classes=self.num_classes,
                                                    top_n=self.config.preprocessing.top_n
                                                    )
        logger.info("Pseudo labeled dataset created.")

        self.pseudo_dataloader = DataLoader(pseudo_dataset, batch_size=self.config.pretrain.batch_size, shuffle=True)
        logger.info("Pseudo labeled dataloader created.")

        self.diff_handler.pseudo_dataloader = self.pseudo_dataloader
        logger.info("Pseudo labeled dataloader set in DiffMiHandler.")

        self.p_reg = get_p_reg(self.public_dataloader, self.target_model, self.device, args=self.config)
        logger.info("Done computing p_reg.")

        self.pgd_model = get_pgd(self.target_model)
        logger.info("PGD model for reconstruction set.")

        if self.config.do_fine_tune:
            self.diff_handler.target_model = self.target_model
            self.diff_handler.p_reg = self.p_reg

            self.diffusion_model = self.diff_handler.get_finetuned().to(device=self.device)
        else:
            self.diffusion_model = self.diff_handler.get_pretrained().to(device=self.device)

    def reconstruct(self, label_num: int, repeat_n: int) -> torch.Tensor:
        """Reconstruct samples for randomly sampled labels."""
        recon_list, success_list, success_label_list = [], [], []
        if label_num <= 0:
            raise ValueError("label_num must be a positive integer.")
        if repeat_n <= 0:
            raise ValueError("repeat_n must be a positive integer.")
        effective_label_num = min(label_num, self.num_classes)
        if effective_label_num != label_num:
            logger.warning(
                f"label_num={label_num} exceeds available classes ({self.num_classes}); "
                f"using {effective_label_num}."
            )
        labels = torch.cat([torch.randperm(effective_label_num) for _ in range(repeat_n)]).to(self.device)
        label_dataset = DataLoader(labels, batch_size=self.config.diffmiattack.batch_size, shuffle=False)

        for _, classes in tqdm(enumerate(label_dataset), total=len(label_dataset)):
            classes = classes.to(self.device)

            recon = iterative_reconstruction(
                args=self.config.diffmiattack,
                diff_net=self.diffusion_model,
                classifier=self.target_model,
                classes=classes,
                p_reg=self.p_reg,
                device=self.device,
                diffusion_steps=self.config.pretrain.diffusion_steps,
                data_channels=self.data_channels,
                data_height=self.data_height,
                data_width=self.data_width,
            ).clamp(0, 1).to(device=self.device)

            translated = self.pgd_model(recon, target=classes, **self.config.diffmiattack.pgdconfig.__dict__)[-1].clamp(0,1)
            _, _, idx = calc_acc(self.evaluation_model, translated, classes, with_success=True)

            recon_list.append(translated)
            success_list.append(translated[idx])
            success_label_list.append(classes[idx])

        recon = (
            torch.cat(recon_list)
            if recon_list
            else torch.empty(0, self.data_channels, self.data_height, self.data_width, device=self.device)
        )
        success = (
            torch.cat(success_list)
            if success_list
            else torch.empty(0, self.data_channels, self.data_height, self.data_width, device=self.device)
        )
        success_labels = (
            torch.cat(success_label_list)
            if success_label_list
            else torch.empty(0, dtype=labels.dtype, device=self.device)
        )
        return recon, success, success_labels, labels, effective_label_num

    def run_attack(self:Self) -> MinvResult:
        """Run the attack."""
        logger.info("Running the Diff-Mi attack")

        recon_list, success_list, success_label_list, labels, effective_label_num = self.reconstruct(
                                                            label_num=self.config.diffmiattack.label_num,
                                                            repeat_n=self.config.diffmiattack.repeat_n
                                                            )

        logger.info("Running reconstruction... ")

        acc1, acc5, var1, var5 = calc_acc_std(recon_list, labels, self.evaluation_model, effective_label_num)
        logger.info(f"Final Top1: {acc1:.2%} ± {var1:.2%}, Top5: {acc5:.2%} ± {var5:.2%}")

        # Save Reconstructed Images
        logger.info(f"Saved {self.config.diffmiattack.repeat_n}x{effective_label_num} generated images.")

        recon_path = f"{self.output_dir}/results/Diff-Mi/all_recreated"
        save_tensor(recon_list, labels, recon_path)
        logger.info(f"All recreated images saved at: {recon_path}")

        success_path = f"{self.output_dir}/results/Diff-Mi/success_recreated"
        save_tensor(success_list, success_label_list, success_path)
        logger.info(f"Successful recreated images saved at: {success_path}")

        # Calculate additional metrics
        knn, knn_arr = np.nan, []
        value_a, value_v = np.nan, np.nan
        avg_mse, mse_per_label, mse_arr, mse_min_fake, mse_min_real = np.nan, [], [], [], []
        fid_value = np.nan

        if self.config.diffmiattack.calc_knn and success_list.shape[0] > 0:
            logger.info("Calculating KNN Distance...")
            knn, knn_arr = calc_knn(success_list, success_label_list,
                     private_feats=self.private_features,
                     private_idents=self.private_idents,
                     evaluation_model=self.evaluation_model,
                     device=self.device,
                     dims=(self.data_height, self.data_width)
                     )
            logger.info(f"KNN Dist computed on {success_list.shape[0]} attack samples: {knn:.2f}")
        elif self.config.diffmiattack.calc_knn:
            logger.warning("Skipping KNN calculation because no successful reconstructions were produced.")

        if self.config.diffmiattack.calc_lpips and success_list.shape[0] > 0:
            logger.info("Calculating LPIPS...")
            value_a, value_v = calc_lpips(
                private_data=self.private_dataloader,
                fakes=success_list,
                fake_targets=success_label_list,
                device=self.device,
            )
            logger.info(f"LPIPS Alex: {value_a:.4f}, VGG: {value_v:.4f}")
        elif self.config.diffmiattack.calc_lpips:
            logger.warning("Skipping LPIPS calculation because no successful reconstructions were produced.")

        if self.config.diffmiattack.calc_mse and success_list.shape[0] > 0:
            logger.info("Calculating MSE...")
            avg_mse, mse_per_label, mse_arr, mse_min_fake, mse_min_real = calc_mse(
                private_data=self.private_dataloader,
                fakes=success_list,
                fake_labels=success_label_list,
                device=self.device,
            )
            logger.info(f"Average MSE: {avg_mse:.4f}")
        elif self.config.diffmiattack.calc_mse:
            logger.warning("Skipping MSE calculation because no successful reconstructions were produced.")

        if self.config.diffmiattack.calc_fid:
            logger.info("Calculating FID...")
            fid_value = calc_pytorch_fid(recon_list, self.private_dataloader, device=self.device)
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

    def get_attack_state(self: Self) -> Dict[str, Any]:
        """Return the attack state with all initialized instance variables.

        Returns
        -------
            Dict[str, Any]: Dictionary containing all initialized attack state variables.

        """
        state = {
            "config": self.config,
            "hashes": self.hashes,
            "output_dir": self.output_dir,
            "storage_path": self.storage_path,
            "device": self.device,
            "diff_handler": self.diff_handler,
        }
        optional_attrs = (
            "target_model",
            "evaluation_model",
            "diffusion_model",
            "private_dataloader",
            "public_dataloader",
            "pseudo_dataloader",
            "private_features",
            "private_idents",
            "p_reg",
            "pgd_model",
        )
        for attr_name in optional_attrs:
            if hasattr(self, attr_name):
                state[attr_name] = getattr(self, attr_name)

        return state

    @classmethod
    def get_generator(cls, handler: MINVHandler, configs: dict) -> "AttackDiffMi":
        """Get the attack.

        Args:
        ----
            handler (MINVHandler): The input handler object.
            configs (dict): Configuration parameters for the attack.

        Returns:
        -------
            AttackDiffMi: The Diff-Mi attack object.

        """
        assert isinstance(handler, MINVHandler), "Handler must be an instance of MINVHandler"
        assert isinstance(configs, dict), "Configs must be a dictionary"

        return AttackDiffMi(handler, configs)
