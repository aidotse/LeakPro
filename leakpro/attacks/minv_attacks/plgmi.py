"""Implementation of the PLGMI attack."""
import os
from typing import Any, Dict, Optional

import cudf
import joblib
import numpy as np
import optuna
import pandas as pd
import torch
from kornia import augmentation
from pydantic import BaseModel, Field
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader

from leakpro.attacks.minv_attacks.abstract_minv import AbstractMINV
from leakpro.attacks.utils import gan_losses
from leakpro.attacks.utils.gan_handler import GANHandler
from leakpro.input_handler.minv_handler import MINVHandler
from leakpro.input_handler.modality_extensions.image_metrics import ImageMetrics
from leakpro.input_handler.modality_extensions.tabular_metrics import TabularMetrics
from leakpro.metrics.attack_result import MinvResult
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


# TODO: Move this to a separate file (GanHandler?)  # noqa: ERA001
class GANConfig(BaseModel):
    """Configuration for GAN."""

    module_path: str = Field(..., description="Path to the model script.")
    model_class: str = Field(..., description="Class name of the model.")
    checkpoint_path: Optional[str] = Field(None, description="Path to the saved model checkpoint.")
    init_params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Initialization parameters.")

class AttackPLGMI(AbstractMINV):
    """Class that implements the PLGMI attack."""

    class Config(BaseModel):
        """Configuration for the PLGMI attack."""

        # General parameters
        batch_size: int = Field(32, ge=1, description="Batch size for training/evaluation")

        # PLG-MI parameters
        top_n : int = Field(10, ge=1, description="Number of pseudo-labels to select")
        alpha: float = Field(0.1, ge=0.0, description="Regularization parameter for inversion optimization")
        n_iter: int = Field(1000, ge=1, description="Number of iterations for optimization")
        log_interval: int = Field(10, ge=1, description="Log interval")

        # Generator parameters
        gen_lr: float = Field(0.0002, ge=0.0, description="Learning rate for the generator")
        gen_beta1: float = Field(0.0, ge=0.0, le=1.0, description="Beta1 parameter for the generator")
        gen_beta2: float = Field(0.9, ge=0.0, le=1.0, description="Beta2 parameter for the generator")

        # Discriminator parameters
        n_dis: int = Field(2, ge=1, description="Number of discriminator updates per generator update")
        dis_lr: float = Field(0.0002, ge=0.0, description="Learning rate for the discriminator")
        dis_beta1: float = Field(0.0, ge=0.0, le=1.0, description="Beta1 parameter for the discriminator")
        dis_beta2: float = Field(0.9, ge=0.0, le=1.0, description="Beta2 parameter for the discriminator")

        # Model parameters
        generator: GANConfig = Field(..., description="Configuration for the generator")
        discriminator: Optional[GANConfig] = Field(..., description="Configuration for the discriminator")

        # Latent space parameters
        dim_z: int = Field(128, ge=1, description="Dimension of the latent space")
        z_optimization_iter: int = Field(1000, ge=1, description="Number of iterations for optimizing z")
        z_optimization_lr: float = Field(0.0002, ge=0.0, description="Learning rate for optimizing z")

        # dataloader or dataframe
        data_format: str = Field("dataloader", description="Data format for the pseudo labels")


        # TODO: Most of these are not necessary if models are pre-trained


    def __init__(self: Self, handler: MINVHandler, configs: dict) -> None:
        """Initialize the PLG-MI attack.

        Args:
        ----
            handler (MINVHandler): The input handler object.
            configs (dict): Configuration parameters for the attack.

        """
        logger.info("Configuring PLG-MI attack")
        self.configs = self.Config() if configs is None else self.Config(**configs)
        self.attack_id = 1 # Workaround for now - required by attack scheduler
        self.use_reference_model = True

        # Call the parent class constructor
        super().__init__(handler)

        # Assign the configuration parameters to the object
        for key, value in self.configs.model_dump().items():
            setattr(self, key, value)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_classes = self.handler.get_num_classes()

        if self.configs.generator is not None:
            self.configs.generator.init_params["num_classes"] = self.num_classes
        if self.configs.discriminator is not None:
            self.configs.discriminator.init_params["num_classes"] = self.num_classes



    def description(self:Self) -> dict:
        """Return the description of the attack."""
        title_str = "PLG-MI Attack"
        reference_str = "Pseudo Label-Guided Model Inversion Attack via Conditional Generative \
                            Adversarial Network, Yuan et al. 2023, https://arxiv.org/abs/2302.09814"
        summary_str = "This attack is a model inversion attack that uses the PLG-MI algorithm."
        detailed_str = "The Pseudo Label Guided Model Inversion Attack (PLG-MI) is a white-box attack \
                        that implements pseudo-labels on a public dataset to construct a conditional GAN. \
                            Steps: \
                                1. Top-n selection of pseudo labels. \
                                2. Train the GAN. \
                                3. Generate samples from the GAN. \
                                4. Compute image metrics. "
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }


    def top_n_selection(self:Self) -> DataLoader:  # noqa: C901, PLR0912
        """"Top n selection of pseudo labels."""
        # TODO: This does not scale well. Consider creating a class for the dataloader and implementing the __getitem__ method.  # noqa: E501, ERA001
        logger.info("Performing top-n selection for pseudo labels")
        self.target_model.eval()
        self.target_model.to(self.device)
        all_confidences = []
        self.target_model.to(self.device)

        # TODO: Maybe this is handler/modality functions
        if self.data_format == "dataloader":
            for entry, _ in self.public_dataloader:
                with torch.no_grad():
                    outputs = self.target_model(entry.to(self.device))
                    confidences = F.softmax(outputs, dim=1)
                    all_confidences.append(confidences)
            # Concatenate all confidences

        elif self.data_format == "dataframe":
            # Remove "identity" column from dataset
            public_data = self.public_dataloader.dataset.drop(columns=["identity"])
            with torch.no_grad():
                outputs = self.target_model(public_data)
                confidences = F.softmax(outputs, dim=1)
                all_confidences.append(confidences)

        else:
            raise ValueError("Data format not supported")

        self.confidences = torch.cat(all_confidences)

        logger.info("Retrieved confidences from the target model")
        # Get the pseudo label confidences
        label_confidences = torch.max(self.confidences, dim=1)

        # Empty array of size num_classes to store the entries for each pseudo label
        pseudo_map = [[] for _ in range(self.num_classes)]

        for i, (conf, label) in enumerate(zip(label_confidences[0], label_confidences[1])):
            # Append the image index i and confidence to the corresponding pseudo label
            pseudo_map[label.item()].append((i, conf.item()))

        # Sort pseudo_map by confidence descending
        for i in range(self.num_classes):
            pseudo_map[i] = sorted(pseudo_map[i], key=lambda x: x[1], reverse=True)

        # Keep only top_n entries in each element of pseudo_map
        top_n_pseudo_map = [pseudo_map[i][:self.top_n] for i in range(self.num_classes)]

        # Create pseudo dataloader from top-n pseudo_map
        pseudo_data = []

        if self.data_format == "dataloader":
            for i in range(self.num_classes):
                for index, _ in top_n_pseudo_map[i]:
                    # Append the image and pseudo label (index i) to the pseudo data  # noqa: ERA001
                    pseudo_data.append((self.public_dataloader.dataset[index][0], i))
        elif self.data_format == "dataframe":
            for i in range(self.num_classes):
                for index, _ in top_n_pseudo_map[i]:
                    # Append the image and pseudo label (index i) to the pseudo data  # noqa: ERA001
                    # Name the column with i is "pseudo_label"
                    pseudo_entry = public_data.iloc[index].copy()
                    pseudo_entry["pseudo_label"] = i
                    pseudo_data.append(pseudo_entry)
            if torch.cuda.is_available() and isinstance(public_data, cudf.DataFrame):
                pseudo_data = cudf.DataFrame(pseudo_data)
            else:
                pseudo_data = pd.DataFrame(pseudo_data)

            logger.info(f"Number of unique classes in pseudo data: {pseudo_data['pseudo_label'].nunique()}")
        logger.info("Created pseudo dataloader")

        # pseudo_data is now a list of tuples (entry, pseudo_label)  # noqa: ERA001
        # We want to set the default device to the sampler in the returned dataloader
        # to be on device, does not apply when using CTGAN  # noqa: ERA001
        return DataLoader(pseudo_data, batch_size=self.batch_size, shuffle=True, generator=torch.Generator(device=self.device))


    def prepare_attack(self:Self) -> None:
        """Prepare the attack."""
        logger.info("Preparing attack")

        # Get the target model from the handler
        self.target_model = self.handler.target_model
        if self.data_format == "dataframe":
            self.gan_handler = GANHandler(self.handler, configs=self.configs, use_discriminator=False)
            self.generator = self.gan_handler.get_generator()

            self.discriminator = None
            self.gen_optimizer = None
            self.dis_optimizer = None
        elif self.data_format == "dataloader":
            self.gan_handler = GANHandler(self.handler, configs=self.configs)
            self.generator = self.gan_handler.get_generator()

            # Get the discriminator
            self.discriminator = self.gan_handler.get_discriminator()
            # Set Adam optimizer for both generator and discriminator
            self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.gen_lr,
                                                betas=(self.gen_beta1, self.gen_beta2))
            self.dis_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.dis_lr,
                                                betas=(self.dis_beta1, self.dis_beta2))
        else:
            raise ValueError("Data format not supported")

        # TODO: Change structure of how we load data, handler or model_handler should do this, not gan_handler  # noqa: ERA001
        # Get public dataloader
        self.public_dataloader = self.handler.get_public_dataloader(self.configs.batch_size)

        # Train the GAN
        # self.gan_handler.trained_bool = True  # noqa: ERA001
        if not self.gan_handler.trained_bool:
            logger.info("GAN not trained, getting psuedo labels")
            # Top-n-selection to get pseudo labels
            self.pseudo_loader = self.top_n_selection()

            # Print number of unique classes in pseudo_loader


            logger.info("Training the GAN")
            # TODO: Change this input structure to just pass the attack class
            self.handler.train_gan(pseudo_loader = self.pseudo_loader,
                                    gen = self.generator,
                                    dis = self.discriminator,
                                    gen_criterion = gan_losses.GenLoss(loss_type="hinge", is_relativistic=False),
                                    dis_criterion = gan_losses.DisLoss(loss_type="hinge", is_relativistic=False),
                                    inv_criterion = gan_losses.max_margin_loss, # TODO: WAS MAX_MARGIN, CHANGE BACK
                                    target_model = self.target_model,
                                    opt_gen = self.gen_optimizer,
                                    opt_dis = self.dis_optimizer,
                                    n_iter = self.n_iter,
                                    n_dis  = self.n_dis,
                                    device = self.device,
                                    alpha = self.alpha,
                                    log_interval = self.log_interval,
                                    sample_from_generator = lambda: \
                                        self.gan_handler.sample_from_generator(batch_size=self.batch_size))

            self.gan_handler.trained_bool = True
        else:
            logger.info("GAN already trained, loading from file")
            from ctgan import CTGAN  # or your CustomCTGAN if you're using that  # noqa: PLC0415
            ctgan = CTGAN.load("ctgan.pkl")
            ctgan.eval()
            self.generator = ctgan  # Replace uninitialized generator

        if self.generator.loss_values is not None:
            self.generator.loss_values.to_pickle("GAN_losses.pkl")
        else:
            logger.warning("No generator loss values found — skipping save.")


    def run_attack(self:Self) -> MinvResult:
        """Run the attack."""
        logger.info("Running the PLG-MI attack")
        # Define image metrics class
        # if getattr(self, "reference_model", None) is None:  # noqa: ERA001
        #     self.load_reference_model(model_folder="./reference")  # noqa: ERA001

        self.evaluation_model = self.target_model # TODO: Change to evaluation model
        reconstruction_configs = self.handler.configs.audit.reconstruction

        num_audited_classes = reconstruction_configs.num_audited_classes


        # Number of unique categories seen by the generator during training
        # If the pseudo_labeling did not find all classes, we get num_unique_categories < num_classes  # noqa: ERA001
        num_unique_categories = (self.generator._data_sampler._n_categories
                                 - self.generator._data_sampler._discrete_column_cond_st[-1])

        if self.num_classes > num_unique_categories:
            logger.info(
                "Auditing %d classes out of %d classes instead,"
                " due to partial class separation in psuedo-labeling step.",
                num_unique_categories,
                self.num_classes,
            )
            labels = torch.arange(num_unique_categories).to(self.device)

        else:
            labels = torch.arange(num_audited_classes).to(self.device)

        # Get range of labels from 0 to num_audited_classes
        #labels = torch.arange(num_audited_classes).to(self.device)  # noqa: ERA001

        # random_z = torch.randn(num_audited_classes, self.generator.dim_z, device=self.device)  # noqa: ERA001

        # Optimize z, TODO: Optimize in batches  # noqa: ERA001

        if self.data_format == "dataloader":

            opt_z = self.optimize_z_grad(y=labels,
                                iter_times=self.configs.z_optimization_iter).to(self.device)

            # Compute image metrics for the optimized z and labels
            metrics = ImageMetrics(self.handler, self.gan_handler,
                                        reconstruction_configs,
                                        labels=labels,
                                        z=opt_z)
            # TODO: Implement a class with a .save function.  # noqa: ERA001

        elif self.data_format == "dataframe":
            # Accuracy of the target model on the random labels
            # initial_metrics = TabularMetrics(self.handler, self.gan_handler,  # noqa: ERA001
            #                             reconstruction_configs,  # noqa: ERA001
            #                             labels=labels,  # noqa: ERA001
            #                             z=random_z)  # noqa: ERA001
            # logger.info("INITIAL ACCURACY:", initial_metrics.results)  # noqa: ERA001

            # generate samples from the generator
            if self.handler.configs.target.model_type == "xgboost":
                opt_z = self.optimize_z_no_grad(y=labels,
                                iter_times=self.configs.z_optimization_iter)
            elif self.handler.configs.target.model_type == "pytorch_tabular":
                opt_z, a = self.optimize_z_grad_per_sample(y=labels,
                                iter_times=self.configs.z_optimization_iter, augment=False)

            metrics = TabularMetrics(self.handler, self.gan_handler,
                                        reconstruction_configs,
                                        labels=labels,
                                        z=opt_z)

        return metrics
        # return metrics  # noqa: ERA001

    def optimize_z_grad_original(self:Self,
                   y: torch.tensor,
                   iter_times: int = 10,
                   augment: bool = True) -> torch.tensor:
        """Find the optimal latent vectors z for labels y.

        Args:
        ----
            y (torch.tensor): The class labels.
            lr (float): The learning rate for optimization.
            iter_times (int): The number of iterations for optimization.
            augment (bool): Whether to apply data augmentation.

        Returns:
        -------
            torch.tensor: Optimized latent vectors.

        """
        bs = y.shape[0] # Number of samples
        y = y.view(-1).long().to(self.device)

        self.generator.eval()
        self.generator.to(self.device)
        self.target_model.eval()
        self.target_model.to(self.device)
        self.evaluation_model.eval()
        self.evaluation_model.to(self.device)

        if augment:
            aug_list = augmentation.container.ImageSequential(
                augmentation.RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
                augmentation.ColorJitter(brightness=0.2, contrast=0.2),
                augmentation.RandomHorizontalFlip(),
                augmentation.RandomRotation(5),
            ).to(self.device) # TODO: Move this to a image modality extension and have it as an input
        else:
            def aug_list(x):  # noqa: ANN001, ANN202
                return x

        logger.info("Optimizing z for the PLG-MI attack")

        z = torch.randn(bs, self.generator.dim_z, device=self.device, requires_grad=True)
        z.requires_grad = True

        optimizer = torch.optim.Adam([z], lr=self.configs.z_optimization_lr)

        min_loss = 1e6
        for i in range(iter_times):
            # Generate fake images
            fake = self.generator(z, y)

            if self.handler.configs.target.model_type == "pytorch_tabular":
                #y = fake["pseudo_label"].values  # noqa: ERA001
                #y = torch.tensor(y, dtype=torch.long).to(self.device)  # noqa: ERA001
                fake = fake.drop(columns=["pseudo_label"])

            out1 = self.target_model(aug_list(fake))
            out2 = self.target_model(aug_list(fake))
            # compute the loss

            inv_loss = F.cross_entropy(out1, y) + F.cross_entropy(out2, y)

            if inv_loss < min_loss:
                min_loss = inv_loss
                # Save the best z
                best_z = z.clone()

            if z.grad is not None:
                z.grad.data.zero_()

            # Update the latent vector z
            optimizer.zero_grad()
            inv_loss.backward()
            optimizer.step()

            inv_loss_val = inv_loss.item()

            if (i + 1) % self.log_interval == 0:
                with torch.no_grad():
                    eval_prob = self.evaluation_model(fake)
                    eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                    acc = y.eq(eval_iden.long()).sum().item() * 1.0 / bs
                    logger.info("Iteration:{}\tInv Loss:{:.2f}\tAttack Acc:{:.2f}".format(i + 1, inv_loss_val, acc))

        return best_z

    def optimize_z_grad_per_sample(self, y, iter_times=30, augment: bool = True):  # noqa: ANN001, ANN201, ARG002, C901, PLR0915
        """For each label y[i], run 20 restarts and keep the best 10 (by final loss).
        SIDE EFFECTS (no change to signature or returns):
        - saves per-class loss-vs-iteration plots (mean ± std) of the kept top-k restarts
        - saves per-class CSVs of those curves
        - saves per-class mean/variance of final top-k losses.

        RETURNS (unchanged):
        z_topk   : [num_samples, 10, dim_z]
        loss_topk: [num_samples, 10]
        """  # noqa: D205
        import os  # noqa: PLC0415

        import matplotlib.pyplot as plt  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415
        import torch  # noqa: PLC0415

        # --- hyperparams for the restart strategy ---
        restarts  = 1
        keep_topk = 1
        lr        = 2e-4

        assert restarts >= keep_topk, "restarts must be >= keep_topk"

        y_all = y.view(-1).long().to(self.device)

        # Target model
        mdl = (self.target_model.model if hasattr(self.target_model, "model") else self.target_model).eval().to(self.device)
        for p in mdl.parameters():
            p.requires_grad_(False)  # we only optimize z

        # Generator
        gen = self.generator
        gen.eval()
        gen.to(self.device)

        # Helper: forward to logits for a single sample (batch size 1)  # noqa: ERA001
        def _forward_logits(z1, yi1):  # noqa: ANN001, ANN202
            if self.handler.configs.target.model_type == "pytorch_tabular":
                # differentiable tabular path
                fakeact, _ = gen.forward_fakeact_with_labels(z1, yi1.view(1))
                x_cont, x_cat, _ = gen._pack_for_gandalf(fakeact)

                batch = {}
                if x_cont is not None and x_cont.numel() > 0:
                    batch["continuous"] = x_cont.float()
                if x_cat is not None and x_cat.numel() > 0:
                    batch["categorical"] = x_cat.long()

                assert batch, "No continuous or categorical features were produced for this batch."
                out = mdl(batch)
                return out["logits"] if isinstance(out, dict) else out
            imgs = gen(z1, yi1.view(1))
            return mdl(imgs)

        z_topk_list    = []
        loss_topk_list = []
        per_class_traces = {}  # cls -> list of np arrays (length iter_times), from kept restarts

        for yi in y_all:
            cls = int(yi.item())
            candidates = []  # (final_loss_after_last_step, best_z_of_restart, trace_post_step[np(iter_times)])

            for _ in range(restarts):
                z   = torch.randn(1, gen.dim_z, device=self.device, requires_grad=True)
                opt = torch.optim.Adam([z], lr=lr)

                best_loss_r = float("inf")
                best_z_r    = z.detach().clone()
                trace = np.zeros(iter_times, dtype=np.float32)

                for it in range(iter_times):
                    # forward (with grad) -> loss_before  # noqa: ERA001
                    logits = _forward_logits(z, yi)
                    # inv_loss = gan_losses.max_margin_loss(logits, yi.view(1))  # noqa: ERA001
                    temp = 2
                    inv_loss = torch.nn.functional.cross_entropy(logits / temp, yi.view(1))

                    # step
                    opt.zero_grad(set_to_none=True)
                    inv_loss.backward()
                    z.grad.norm().item() if z.grad is not None else 0.0
                    opt.step()

                    # recompute loss AFTER the update (no grad) to plot meaningful progress  # noqa: ERA001
                    with torch.no_grad():
                        logits_after = _forward_logits(z, yi)
                        # loss_after = gan_losses.max_margin_loss(logits_after, yi.view(1)).item()  # noqa: ERA001
                        loss_after = torch.nn.functional.cross_entropy(logits_after/ temp, yi.view(1)).item()

                    # record post-step loss
                    trace[it] = float(loss_after)

                    # track the best within this restart by current (post-step) loss  # noqa: ERA001
                    if loss_after < best_loss_r:
                        best_loss_r = loss_after
                        best_z_r    = z.detach().clone()

                    # optional lightweight logging
                    if (it % 20) == 0:
                        pass

                candidates.append((best_loss_r, best_z_r, trace))

            # keep top-k restarts for THIS sample
            candidates.sort(key=lambda t: t[0])
            top = candidates[:keep_topk]

            # store returns
            loss_topk_list.append(torch.tensor([t[0] for t in top], device=self.device))
            z_topk_list.append(torch.stack([t[1].squeeze(0) for t in top], dim=0))  # [keep_topk, dim_z]

            # accumulate traces for per-class plotting
            per_class_traces.setdefault(cls, []).extend([t[2] for t in top])

        z_topk   = torch.stack(z_topk_list, dim=0)    # [N, keep_topk, dim_z]
        loss_topk = torch.stack(loss_topk_list, dim=0)  # [N, keep_topk]

        # ---------- SAVE per-class plots & stats (side effects only) ----------  # noqa: ERA001
        out_dir = os.path.join(os.getcwd(), "z_opt_plots")
        os.makedirs(out_dir, exist_ok=True)

        # 1) Loss-vs-iteration curves: mean ± std over kept traces  # noqa: ERA001
        rows_curves = []
        for cls, traces in per_class_traces.items():
            T = np.stack(traces, axis=0)   # [n_traces, iter_times]  # noqa: N806
            mean = T.mean(axis=0)
            std  = T.std(axis=0, ddof=0)
            iters = np.arange(1, mean.size + 1)

            # plot
            plt.figure(figsize=(6, 4))
            plt.plot(iters, mean, label=f"class {cls} mean")
            plt.fill_between(iters, mean - std, mean + std, alpha=0.3)
            plt.xlabel("iteration"); plt.ylabel("loss")  # noqa: E702
            plt.title(f"Loss vs. iteration (top-{keep_topk} restarts kept, class {cls}, n_traces={T.shape[0]})")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"loss_curve_class_{cls}.png"), dpi=200)
            plt.close()

            # csv of curve data
            pd.DataFrame({
                "iteration": iters,
                "mean": mean,
                "std": std,
                "n_traces": T.shape[0]
            }).to_csv(os.path.join(out_dir, f"loss_curve_class_{cls}.csv"), index=False)

            rows_curves.append({
                "class": int(cls),
                "n_traces": int(T.shape[0]),
                "final_mean": float(mean[-1]),
                "final_std":  float(std[-1]),
            })

        pd.DataFrame(rows_curves).sort_values("class").to_csv(
            os.path.join(out_dir, "loss_curve_class_summary.csv"), index=False
        )

        # 2) Final top-k loss stats per class (mean/var of kept final losses)  # noqa: ERA001
        y_cpu  = y_all.detach().cpu().numpy()
        L_cpu  = loss_topk.detach().cpu().numpy()   # [N, keep_topk]  # noqa: N806
        classes = np.unique(y_cpu)
        rows_stats = []
        for cls in classes:
            mask = (y_cpu == cls)
            if not mask.any():
                continue
            L_cls = L_cpu[mask].reshape(-1)  # flatten all kept losses for this class  # noqa: N806
            rows_stats.append({
                "class": int(cls),
                "n_samples_in_class": int(mask.sum()),
                "k_kept_per_sample": int(keep_topk),
                "final_loss_mean": float(L_cls.mean()),
                "final_loss_var":  float(L_cls.var(ddof=0)),
            })

        pd.DataFrame(rows_stats).sort_values("class").to_csv(
            os.path.join(out_dir, "topk_final_loss_stats_by_class.csv"), index=False
        )

        # ---------- RETURN (unchanged) ----------  # noqa: ERA001
        return z_topk, loss_topk



    def optimize_z_grad(self:Self,
                   y: torch.tensor,
                   iter_times: int = 10,
                   augment: bool = True) -> torch.tensor:
        """Find the optimal latent vectors z for labels y.

        Args:
        ----
            y (torch.tensor): The class labels.
            lr (float): The learning rate for optimization.
            iter_times (int): The number of iterations for optimization.
            augment (bool): Whether to apply data augmentation.

        Returns:
        -------
            torch.tensor: Optimized latent vectors.

        """
        bs = y.shape[0] # Number of samples
        y = y.view(-1).long().to(self.device)

        self.generator.eval()
        self.generator.to(self.device)
        self.target_model.eval()
        self.target_model.to(self.device)
        self.evaluation_model.eval()
        self.evaluation_model.to(self.device)

        if augment:
            aug_list = augmentation.container.ImageSequential(
                augmentation.RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
                augmentation.ColorJitter(brightness=0.2, contrast=0.2),
                augmentation.RandomHorizontalFlip(),
                augmentation.RandomRotation(5),
            ).to(self.device) # TODO: Move this to a image modality extension and have it as an input
        else:
            def aug_list(x):  # noqa: ANN001, ANN202
                return x

        logger.info("Optimizing z for the PLG-MI attack")
        # inside optimize_z_grad(...)  # noqa: ERA001
        z = torch.randn(bs, self.generator.dim_z, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([z], lr=self.configs.z_optimization_lr)
        min_loss = float("inf")
        best_z = z.detach().clone()

        mdl = self.target_model.model if hasattr(self.target_model, "model") else self.target_model
        mdl = mdl.to(self.device).eval()

        for _i in range(iter_times):
            if self.handler.configs.target.model_type == "pytorch_tabular":
                # ---- differentiable tabular path ----
                fakeact, _ = self.generator.forward_fakeact_with_labels(z, y)     # <<< NEW
                x_cont, x_cat, _ = self.generator._pack_for_gandalf(fakeact)      # uses your torch invertor
                batch = {}
                if x_cont is not None: batch["continuous"]  = x_cont.float()  # noqa: E701
                if x_cat  is not None: batch["categorical"] = x_cat.long()  # noqa: E701
                out1 = mdl(batch)["logits"]
            else:
                # ---- image (original) path ----  # noqa: ERA001
                fake = self.generator(z, y)                   # image tensor path only
                out1 = self.target_model(aug_list(fake))

            inv_loss = gan_losses.max_margin_loss(out1, y)

            if inv_loss.item() < min_loss:
                min_loss = inv_loss.item()
                best_z = z.detach().clone()

            optimizer.zero_grad()
            inv_loss.backward()
            optimizer.step()

        return best_z

    def optimize_z_grad2(self:Self,
                   y: torch.tensor,
                   iter_times: int = 10,
                   augment: bool = True) -> torch.tensor:
        """Find the optimal latent vectors z for labels y.

        Args:
        ----
            y (torch.tensor): The class labels.
            lr (float): The learning rate for optimization.
            iter_times (int): The number of iterations for optimization.
            augment (bool): Whether to apply data augmentation.

        Returns:
        -------
            torch.tensor: Optimized latent vectors.

        """
        bs = y.shape[0] # Number of samples
        y = y.view(-1).long().to(self.device)

        self.generator.eval()
        self.generator.to(self.device)
        self.target_model.eval()
        self.target_model.to(self.device)
        self.evaluation_model.eval()
        self.evaluation_model.to(self.device)

        if augment:
            aug_list = augmentation.container.ImageSequential(
                augmentation.RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
                augmentation.ColorJitter(brightness=0.2, contrast=0.2),
                augmentation.RandomHorizontalFlip(),
                augmentation.RandomRotation(5),
            ).to(self.device) # TODO: Move this to a image modality extension and have it as an input
        else:
            def aug_list(x):  # noqa: ANN001, ANN202
                return x

        logger.info("Optimizing z for the PLG-MI attack")
        z = torch.randn(bs, self.generator.dim_z, device=self.device)
        z = torch.nn.Parameter(z)
        optimizer = torch.optim.Adam([z], lr=self.configs.z_optimization_lr)

        for _i in range(iter_times):
            # z = torch.randn(bs, self.generator.dim_z, device=self.device)  # noqa: ERA001
            # z.requires_grad = True  # noqa: ERA001
            # Generate fake images
            fake = self.generator(z, y)

            if self.handler.configs.target.model_type == "pytorch_tabular":
                #y = fake["pseudo_label"].values  # noqa: ERA001
                #y = torch.tensor(y, dtype=torch.long).to(self.device)  # noqa: ERA001
                fake = fake.drop(columns=["pseudo_label"])

            out1 = self.target_model(aug_list(fake))
            #out2 = self.target_model(aug_list(fake))  # noqa: ERA001

            # if z.grad is not None:  # noqa: ERA001
            #     z.grad.data.zero_()  # noqa: ERA001

            # TODO: Change it latter
            # Load a refrence model
            # if True:  # noqa: ERA001
                # Compute inversion loss
                # inv_loss = self.compute_total_loss(fake,  # noqa: ERA001
                                                # y,  # noqa: ERA001
                                                # out1)  # noqa: ERA001
                # logger.info(f"Using Ref model loss")  # noqa: ERA001
            # else:  # noqa: ERA001
                # inv_loss = F.cross_entropy(out1, y) #+ F.cross_entropy(out2, y)  # noqa: ERA001
            inv_loss = gan_losses.max_margin_loss(out1, y)



            # if inv_loss < min_loss:  # noqa: ERA001
            #     min_loss = inv_loss  # noqa: ERA001
            #     # Save the best z
            #     best_z = z.clone()  # noqa: ERA001

            # Update the latent vector z
            optimizer.zero_grad()
            inv_loss.backward()
            optimizer.step()

            # print("After backward:", z.grad)  # noqa: ERA001
            inv_loss.item()

            # if (i + 1) % self.log_interval == 0:  # noqa: ERA001
            #     with torch.no_grad():  # noqa: ERA001
            #         eval_prob = self.evaluation_model(fake)  # noqa: ERA001
            #         eval_iden = torch.argmax(eval_prob, dim=1).view(-1)  # noqa: ERA001
            #         acc = y.eq(eval_iden.long()).sum().item() * 1.0 / bs  # noqa: ERA001
            #         logger.info("Iteration:{}\tInv Loss:{:.2f}\tAttack Acc:{:.2f}".format(i + 1, inv_loss_val, acc))  # noqa: E501, ERA001

        return z

    def optimize_z_no_grad(self, y: torch.tensor, iter_times: int = 10) -> torch.tensor:
        """Find the optimal latent vectors z for labels y.

        Args:
        ----
            y (torch.tensor): The class labels.
            lr (float): The learning rate for optimization.
            iter_times (int): The number of iterations for optimization.

        Returns:
        -------
            torch.tensor: Optimized latent vectors.

        """
        bs = y.shape[0]  # Batch size
        y = y.view(-1).long().to(self.device)
        self.generator.eval()
        self.generator.to(self.device)


        # Use Optuna for Bayesian optimization
        def objective(trial: optuna.trial.Trial) -> float:
            # Suggest values for each dimension of z
            z_numpy = np.array([trial.suggest_float(f"z_{i}", -3, 3) for i in range(bs * self.generator.dim_z)])
            z_numpy = z_numpy.reshape(bs, self.generator.dim_z)  # Reshape to match the required shape

            z_tensor = torch.tensor(z_numpy, dtype=torch.float32, device=self.device)

            fake = self.generator(z_tensor, y)
            fake = fake.drop(columns=["pseudo_label"])  # Remove pseudo_label

            out1 = self.target_model.predict_proba(fake)  # Get class probabilities from XGBoost
            y_one_hot = torch.nn.functional.one_hot(y, num_classes=out1.shape[1]).float().cpu().numpy()

            # Compute cross-entropy loss manually
            eps = 1e-8  # Prevent log(0)
            return -np.sum(y_one_hot * np.log(out1 + eps)) / bs

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")

        logger.info("Optimizing z using Optuna")
        study.optimize(objective, n_trials=iter_times, show_progress_bar=True, n_jobs=-1)

        # Convert the dictionary values to a NumPy array
        z_numpy = np.array(list(study.best_params.values()))

        # Reshape the array to match the required shape (bs, self.generator.dim_z)  # noqa: ERA001
        z_numpy = z_numpy.reshape(bs, self.generator.dim_z)

        # Convert the NumPy array to a PyTorch tensor
        return torch.tensor(z_numpy, dtype=torch.float32, device=self.device)



    def load_reference_model(self, model_folder: str = "./reference") -> None:  # noqa: C901
        """Loads a TreeGrad (TGDClassifier) + preprocessing saved as:
        model_folder/
            ├─ model.ckpt            (joblib dict with {"model": tgd_model, ...})
            ├─ custom_params.sav     (joblib: {"ohe","num_cols","cat_cols","feature_order","target_col"})
            └─ config.yml            (optional)
        Sets: self.reference_model  (callable like a torch model returning torch logits).
        """  # noqa: D205
        if model_folder is None or not os.path.isdir(model_folder):
            raise FileNotFoundError(f"Model folder not found: {model_folder}")

        bundle = joblib.load(os.path.join(model_folder, "model.ckpt"))   # {"model": tgd_model, ...}
        meta   = joblib.load(os.path.join(model_folder, "custom_params.sav"))
        # cfg = yaml.safe_load(open(os.path.join(model_folder, "config.yml"))) if os.path.exists(os.path.join(model_folder,"config.yml")) else {}  # noqa: E501, ERA001

        tgd_model = bundle["model"]
        ohe        = meta["ohe"]
        num_cols   = meta["num_cols"]
        cat_cols   = meta["cat_cols"]
        feat_order = meta["feature_order"]

        class _RefModel(torch.nn.Module):
            def __init__(self, model, ohe, num_cols, cat_cols, feat_order, device) -> None:  # noqa: ANN001
                super().__init__()
                self.model = model
                self.ohe = ohe
                self.num_cols = num_cols
                self.cat_cols = cat_cols
                self.feature_order = feat_order
                self.device = torch.device(device)

            def to(self, device):  # noqa: ANN001, ANN202
                self.device = torch.device(device); return self  # noqa: E702

            def eval(self): return self  # no-op  # noqa: ANN202

            @torch.no_grad()
            def __call__(self, entry) -> torch.Tensor:  # noqa: ANN001
                # Expect a pandas.DataFrame with original feature columns (tabular PLGMI uses this)  # noqa: ERA001
                if isinstance(entry, pd.DataFrame):
                    cols = [c for c in self.feature_order if c in entry.columns]
                    X_df = entry[cols].copy()  # noqa: N806
                    X_cat = self.ohe.transform(X_df[self.cat_cols]) if self.cat_cols else np.empty((len(X_df), 0))  # noqa: N806
                    X_num = X_df[self.num_cols].to_numpy(dtype=float)    if self.num_cols else np.empty((len(X_df), 0))  # noqa: N806
                    X_np  = np.hstack([X_cat, X_num]).astype("float32", copy=False)  # noqa: N806
                elif isinstance(entry, torch.Tensor):
                    # If you ever pass a preprocessed numeric tensor (OHE+num), allow it:  # noqa: ERA001
                    X_np = entry.detach().cpu().numpy()  # noqa: N806
                else:
                    raise TypeError("reference_model expects pandas.DataFrame or preprocessed torch.Tensor")

                if hasattr(self.model, "predict_proba"):
                    prob = self.model.predict_proba(X_np)
                    logits = np.log(np.clip(prob, 1e-12, 1.0))
                    return torch.tensor(logits, dtype=torch.float32, device=self.device)
                if hasattr(self.model, "decision_function"):
                    dec = self.model.decision_function(X_np)
                    if dec.ndim == 1: dec = np.vstack([-dec, dec]).T  # noqa: E701
                    return torch.tensor(dec, dtype=torch.float32, device=self.device)
                raise RuntimeError("Reference model lacks predict_proba/decision_function.")

        self.reference_model = _RefModel(tgd_model, ohe, num_cols, cat_cols, feat_order, self.device).to(self.device).eval()
        logger.info(f"Loaded reference model from {model_folder}")




    def compute_total_loss(  # noqa: ANN201, D102
            self,
            x: torch.Tensor,               # generated samples; if you need R-gradient, x must require_grad=True
            y: torch.Tensor,               # target class indices [B]
            logits_T: torch.Tensor,        # target model logits on x  (T(x))  # noqa: N803
            # beta: float = 1.0,             # weight on reference margin  # noqa: ERA001
            ):
        # 1) Target inversion loss (max-margin on T)  # noqa: ERA001
        l_inv = gan_losses.max_margin_loss(logits_T, y)  # scalar


        ref_logits = self.reference_model(x)
        l_ref_max_margin = gan_losses.max_margin_loss(ref_logits, y)
        # if l is positive, it means the correct class logit is smaller than some other class logit by at least margin  # noqa: E501, ERA001
        return l_inv -  l_ref_max_margin


def fixed_margin_loss(out, y, neg):  # noqa: ANN001, ANN201, D103
    real = out.gather(1, y[:,None]).squeeze(1)
    negv = out.gather(1, torch.tensor([neg], device=out.device).repeat(len(y))[:,None]).squeeze(1)
    return (-real + negv).mean()
