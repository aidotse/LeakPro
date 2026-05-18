#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#

"""Implementation of the Diff-Mi attack."""
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
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
from leakpro.attacks.utils.diff_mi.setup import (
    DiffMiConfig,
    clear_cuda_cache,
    extract_features,
    get_p_reg,
    top_n_pseudo_label_dataset,
)
from leakpro.attacks.utils.diff_mi.train_util import is_finetune_complete, is_pretrain_complete
from leakpro.attacks.utils.diffusion_handler import DiffMiHandler
from leakpro.input_handler.minv_handler import MINVHandler
from leakpro.reporting.minva_result import MinvResult
from leakpro.schemas import ReconstructionConfig
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger
from leakpro.utils.save_load import hash_config


class AttackDiffMi(AbstractMINV):
    """Class that implements the DiffMi attack."""

    SUPPORTED_RECONSTRUCTION_METRICS = {"accuracy", "knn", "lpips", "mse", "fid", "save_images"}

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
        self.handler = handler
        self.output_dir = handler.configs.audit.output_dir if handler.configs.audit.output_dir else "./leakpro_output/"
        self._configs_ = ["preprocessing", "pretrain", "finetune", "diffusion"]
        self._setup_paths_()

        configs = self._validate_configs(configs)
        self.config = DiffMiConfig(**configs)
        self._setup_reconstruction_config()

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
        reconstruction_config = self.handler.configs.audit.reconstruction
        if reconstruction_config is not None:
            configs["reconstruction"] = reconstruction_config.model_dump()

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

    def _setup_reconstruction_config(self: Self) -> None:
        """Initialize Diff-MI reconstruction options from the shared audit config."""
        reconstruction_config = self.handler.configs.audit.reconstruction or ReconstructionConfig()
        self._apply_reconstruction_config(reconstruction_config)

    def _apply_reconstruction_config(self: Self, reconstruction_config: ReconstructionConfig) -> None:
        """Apply shared audit.reconstruction settings to Diff-MI runtime options."""
        self.reconstruction_batch_size = reconstruction_config.batch_size
        self.reconstruction_label_num = reconstruction_config.num_audited_classes
        self.reconstruction_repeat_n = reconstruction_config.num_class_samples

        metrics = reconstruction_config.metrics or {"accuracy": None}
        unknown_metrics = set(metrics) - self.SUPPORTED_RECONSTRUCTION_METRICS
        if unknown_metrics:
            supported = ", ".join(sorted(self.SUPPORTED_RECONSTRUCTION_METRICS))
            unknown = ", ".join(sorted(unknown_metrics))
            raise ValueError(f"Unsupported Diff-MI reconstruction metrics: {unknown}. Supported metrics: {supported}.")

        self.enabled_metrics = set(metrics)
        self.save_reconstruction_images = "save_images" in self.enabled_metrics
        self.save_image_count = None
        self.save_image_dir = None
        save_images_config = metrics.get("save_images")
        if isinstance(save_images_config, dict):
            self.save_image_count = save_images_config.get("n_images")
            self.save_image_dir = save_images_config.get("save_dir")

        logger.info("Applied audit.reconstruction settings to Diff-MI runtime options.")

    def _load_evaluation_model(self: Self) -> torch.nn.Module:
        """Load the configured reconstruction evaluation model, or fall back to the target model."""
        reconstruction_config = self.handler.configs.audit.reconstruction
        eval_config = getattr(reconstruction_config, "eval_model", None) if reconstruction_config else None
        if eval_config is None:
            logger.info("No reconstruction eval_model configured; using target model for Diff-MI evaluation.")
            return self.target_model

        return self.handler.load_model_from_config(
            module_path=eval_config.module_path,
            model_class=eval_config.model_class,
            model_folder=eval_config.eval_folder,
            role="evaluation",
        ).to(device=self.device)

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

        # GET TARGET MODEL and EVALUATION MODEL
        self.target_model = self.handler.target_model.to(device=self.device)
        logger.info("Target model set.")

        # EVALUATION MODEL
        self.evaluation_model = self._load_evaluation_model()
        logger.info("Evaluation model set.")

        # Private DATASET
        self.private_dataloader = self.handler.get_private_dataloader(batch_size=self.config.pretrain.batch_size)
        logger.info("Loading private dataloader finished.")

        self.private_features, self.private_idents = extract_features(
            self.evaluation_model,
            self.private_dataloader,
            self.device,
            save_dir=self.config.save_path or self.storage_path,
        )
        logger.info("Extracted private features from evaluation model.")

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
        clear_cuda_cache(self.device)

        if self.config.do_fine_tune:
            self.diff_handler.target_model = self.target_model
            self.diff_handler.p_reg = self.p_reg

            self.diffusion_model = self.diff_handler.get_finetuned().to(device=self.device)
        else:
            self.diffusion_model = self.diff_handler.get_pretrained().to(device=self.device)

        clear_cuda_cache(self.device)
        self.pgd_model = get_pgd(self.target_model)
        logger.info("PGD model for reconstruction set.")
        self._freeze_model(self.target_model)
        self._freeze_model(self.evaluation_model)
        self._prepare_diffusion_model_for_guidance()
        self._freeze_model(self.pgd_model)

    def _prepare_diffusion_model_for_guidance(self: Self) -> None:
        """Prepare the diffusion model for image-gradient reconstruction."""
        self.diffusion_model.eval()
        for param in self.diffusion_model.parameters():
            param.requires_grad_(True)
        self.diffusion_model.zero_grad(set_to_none=True)

    def _freeze_model(self: Self, model: torch.nn.Module) -> None:
        """Disable parameter gradients for models used only as reconstruction guides."""
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

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
        label_dataset = DataLoader(labels, batch_size=self.reconstruction_batch_size, shuffle=False)

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

            translated = (
                self.pgd_model(recon, target=classes, **self.config.diffmiattack.pgdconfig.__dict__)[-1]
                .clamp(0, 1)
                .detach()
            )
            _, _, idx = calc_acc(self.evaluation_model, translated, classes, with_success=True)

            recon_list.append(translated)
            success_list.append(translated[idx])
            success_label_list.append(classes[idx].detach())
            del recon, translated, idx, classes
            clear_cuda_cache(self.device)

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
                                                            label_num=self.reconstruction_label_num,
                                                            repeat_n=self.reconstruction_repeat_n
                                                            )

        logger.info("Running reconstruction... ")

        result_id = f"diffmi-{self.hashes['attackhash'][:8]}"
        result_path = f"{self.output_dir}/results/{result_id}"

        acc1, acc5, var1, var5 = np.nan, np.nan, np.nan, np.nan
        if "accuracy" in self.enabled_metrics:
            acc1, acc5, var1, var5 = calc_acc_std(recon_list, labels, self.evaluation_model, effective_label_num)
            logger.info(f"Final Top1: {acc1:.2%} ± {var1:.2%}, Top5: {acc5:.2%} ± {var5:.2%}")

        # Save Reconstructed Images
        if self.save_reconstruction_images:
            save_count = self.save_image_count
            recon_to_save = recon_list[:save_count] if save_count else recon_list
            labels_to_save = labels[:save_count] if save_count else labels
            success_to_save = success_list[:save_count] if save_count else success_list
            success_labels_to_save = success_label_list[:save_count] if save_count else success_label_list

            logger.info(f"Saved {recon_to_save.shape[0]} generated images.")

            image_result_path = self.save_image_dir or result_path
            recon_path = f"{image_result_path}/all_recreated"
            save_tensor(recon_to_save, labels_to_save, recon_path, file_extension=".png")
            logger.info(f"All recreated images saved at: {recon_path}")

            success_path = f"{image_result_path}/success_recreated"
            save_tensor(success_to_save, success_labels_to_save, success_path, file_extension=".png")
            logger.info(f"Successful recreated images saved at: {success_path}")

        # Calculate additional metrics
        knn, knn_arr = np.nan, []
        value_a, value_v = np.nan, np.nan
        avg_mse, mse_per_label, mse_arr, mse_min_fake, mse_min_real = np.nan, [], [], [], []
        fid_value = np.nan

        if "knn" in self.enabled_metrics and success_list.shape[0] > 0:
            logger.info("Calculating KNN Distance...")
            knn, knn_arr = calc_knn(success_list, success_label_list,
                     private_feats=self.private_features,
                     private_idents=self.private_idents,
                     evaluation_model=self.evaluation_model,
                     device=self.device,
                     dims=(self.data_height, self.data_width)
                     )
            logger.info(f"KNN Dist computed on {success_list.shape[0]} attack samples: {knn:.2f}")
        elif "knn" in self.enabled_metrics:
            logger.warning("Skipping KNN calculation because no successful reconstructions were produced.")

        if "lpips" in self.enabled_metrics and success_list.shape[0] > 0:
            logger.info("Calculating LPIPS...")
            value_a, value_v = calc_lpips(
                private_data=self.private_dataloader,
                fakes=success_list,
                fake_targets=success_label_list,
                device=self.device,
            )
            logger.info(f"LPIPS Alex: {value_a:.4f}, VGG: {value_v:.4f}")
        elif "lpips" in self.enabled_metrics:
            logger.warning("Skipping LPIPS calculation because no successful reconstructions were produced.")

        if "mse" in self.enabled_metrics and success_list.shape[0] > 0:
            logger.info("Calculating MSE...")
            avg_mse, mse_per_label, mse_arr, mse_min_fake, mse_min_real = calc_mse(
                private_data=self.private_dataloader,
                fakes=success_list,
                fake_labels=success_label_list,
                device=self.device,
            )
            logger.info(f"Average MSE: {avg_mse:.4f}")
        elif "mse" in self.enabled_metrics:
            logger.warning("Skipping MSE calculation because no successful reconstructions were produced.")

        if "fid" in self.enabled_metrics:
            logger.info("Calculating FID...")
            fid_value = calc_pytorch_fid(recon_list, self.private_dataloader, device=self.device)
            logger.info(f"FID score: {fid_value:.2f}")

        return MinvResult.from_metrics(
            result_name="Diff-Mi Attack Result",
            result_id=result_id,
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

    @classmethod
    def get_generator(
        cls,
        handler: Optional[MINVHandler] = None,
        configs: Optional[Dict[str, Any] | DiffMiConfig] = None,
        config_path: Optional[str | os.PathLike[str]] = None,
        diffusion_steps: Optional[int] = None,
    ) -> DiffMiHandler:
        """Return a Diff-MI diffusion handler loaded from a completed checkpoint.

        Args:
        ----
            handler: Optional input handler to attach to the diffusion handler.
            configs: Diff-MI attack config, either as a raw audit entry or a DiffMiConfig.
            config_path: Optional path to an audit.yaml file. Used to resolve config and checkpoint paths.
            diffusion_steps: Optional number of diffusion sampling steps to use after loading the config.

        Returns:
        -------
            DiffMiHandler: Handler that can sample with ``sample_from_diffusion_model``.

        """
        diffmi_config = cls._resolve_generator_config(handler, configs, config_path)
        if diffusion_steps is not None:
            diffmi_config.pretrain.diffusion_steps = diffusion_steps
            diffmi_config.finetune.diffusion_steps = diffusion_steps
        diff_handler = DiffMiHandler(handler=handler, configs=diffmi_config)
        if diffmi_config.do_fine_tune:
            diff_handler.get_finetuned()
        else:
            diff_handler.get_pretrained()
        return diff_handler

    @classmethod
    def _resolve_generator_config(
        cls,
        handler: Optional[MINVHandler],
        configs: Optional[Dict[str, Any] | DiffMiConfig],
        config_path: Optional[str | os.PathLike[str]],
    ) -> DiffMiConfig:
        """Resolve a Diff-MI config suitable for standalone sampling."""
        if configs is None:
            if config_path is None:
                raise ValueError("Either configs or config_path must be provided to load a Diff-MI generator.")
            return cls._resolve_generator_config_from_audit(Path(config_path))

        if isinstance(configs, DiffMiConfig):
            diffmi_cfg = deepcopy(configs.model_dump())
        else:
            diffmi_cfg = deepcopy(configs)
            diffmi_cfg.pop("attack", None)

        if not diffmi_cfg.get("save_path"):
            if handler is None:
                raise ValueError("save_path must be configured when loading a generator without a handler.")
            output_dir = handler.configs.audit.output_dir or "./leakpro_output"
            diffmi_cfg["save_path"] = str(Path(output_dir) / "attack_objects" / "diffusion_models")

        save_path = Path(diffmi_cfg["save_path"])
        cls._select_completed_generator_checkpoint(diffmi_cfg, save_path)
        return DiffMiConfig(**diffmi_cfg)

    @classmethod
    def _resolve_generator_config_from_audit(cls, config_path: Path) -> DiffMiConfig:
        """Resolve Diff-MI generator config from an audit.yaml file."""
        with config_path.open("r") as f:
            full_config = yaml.safe_load(f)

        diffmi_cfg = next(
            (entry for entry in full_config["audit"]["attack_list"] if entry.get("attack") == "diffmi"),
            None,
        )
        if diffmi_cfg is None:
            raise ValueError(f"No Diff-MI attack entry found in {config_path}.")

        diffmi_cfg = deepcopy(diffmi_cfg)
        diffmi_cfg.pop("attack", None)

        output_dir = Path(full_config["audit"].get("output_dir", "./leakpro_output"))
        if not output_dir.is_absolute():
            output_dir = (config_path.parent / output_dir).resolve()

        save_path = Path(diffmi_cfg.get("save_path") or output_dir / "attack_objects" / "diffusion_models")
        if not save_path.is_absolute():
            save_path = (config_path.parent / save_path).resolve()
        diffmi_cfg["save_path"] = str(save_path)

        cls._select_completed_generator_checkpoint(diffmi_cfg, save_path)
        return DiffMiConfig(**diffmi_cfg)

    @staticmethod
    def _select_completed_generator_checkpoint(diffmi_cfg: Dict[str, Any], save_path: Path) -> None:
        """Select the newest completed fine-tune or pretrain checkpoint for sampling."""
        if not save_path.exists():
            raise FileNotFoundError(f"No Diff-MI checkpoint directory found at {save_path}.")

        config = DiffMiConfig(**diffmi_cfg)
        pretrain_save_name = config.pretrain.save_name or "pretrain"
        finetune_save_name = config.finetune.save_name or "finetune"

        finetune_matches = sorted(
            (
                path
                for path in save_path.glob(f"{finetune_save_name}*.pt")
                if is_finetune_complete(
                    str(save_path),
                    path.stem,
                    config.finetune.epochs,
                    config.finetune.threshold,
                )
            ),
            key=lambda path: path.stat().st_mtime,
        )
        pretrain_matches = sorted(
            (
                path
                for path in save_path.glob(f"{pretrain_save_name}*.pt")
                if is_pretrain_complete(str(save_path), path.stem, config.pretrain.max_steps)
            ),
            key=lambda path: path.stat().st_mtime,
        )

        if finetune_matches:
            diffmi_cfg["do_fine_tune"] = True
            diffmi_cfg.setdefault("finetune", {})["save_name"] = finetune_matches[-1].stem
        elif pretrain_matches:
            diffmi_cfg["do_fine_tune"] = False
            diffmi_cfg.setdefault("pretrain", {})["save_name"] = pretrain_matches[-1].stem
        else:
            raise FileNotFoundError(f"No completed Diff-MI checkpoints found in {save_path}.")
