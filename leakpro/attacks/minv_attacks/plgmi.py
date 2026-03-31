"""Implementation of the PLGMI attack."""
from typing import Any, Dict, Optional

import cudf
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
from leakpro.reporting.minva_result import MinvResult
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


# TODO: Move this to a separate file (GanHandler?)
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
        checkpoint_interval: int = Field(10000, ge=1, description="Checkpoint interval for saving models")
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
        # TODO: This does not scale well. Consider creating a class for the dataloader and implementing the __getitem__ method.
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
                    # Append the image and pseudo label (index i) to the pseudo data
                    pseudo_data.append((self.public_dataloader.dataset[index][0], i))
        elif self.data_format == "dataframe":
            for i in range(self.num_classes):
                for index, _ in top_n_pseudo_map[i]:
                    # Append the image and pseudo label (index i) to the pseudo data
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

        # pseudo_data is now a list of tuples (entry, pseudo_label)
        # We want to set the default device to the sampler in the returned dataloader
        # to be on device, does not apply when using CTGAN
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

        # TODO: Change structure of how we load data, handler or model_handler should do this, not gan_handler
        # Get public dataloader
        self.public_dataloader = self.handler.get_public_dataloader(self.configs.batch_size)

        # Train the GAN
        # self.gan_handler.trained_bool = True
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
                                    inv_criterion = gan_losses.max_margin_loss,
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
            from ctgan import CTGAN  # or your CustomCTGAN if you're using that
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

        self.evaluation_model = self.target_model # TODO: Change to evaluation model
        reconstruction_configs = self.handler.configs.audit.reconstruction

        num_audited_classes = reconstruction_configs.num_audited_classes

        # Number of unique categories seen by the generator during training
        # If the pseudo_labeling did not find all classes, we get num_unique_categories < num_classes
        num_unique_categories = (self.generator._data_sampler._n_categories
                                 - self.generator._data_sampler._discrete_column_cond_st[-1])

        if self.num_classes > num_unique_categories:
            logger.info(
                "Auditing %d classes out of %d classes instead,"
                " due to partial class separation in psuedo-labeling step.",
                num_unique_categories,
                self.num_classes,
            )
            # Get random labels
            #labels = torch.randint(0, num_unique_categories, (num_audited_classes,)).to(self.device)
            labels = torch.arange(num_unique_categories).to(self.device)

        else:
            # Get random labels
            #labels = torch.randint(0, self.num_classes, (num_audited_classes,)).to(self.device)
            labels = torch.arange(num_audited_classes).to(self.device)

        # Get range of labels from 0 to num_audited_classes
        #labels = torch.arange(num_audited_classes).to(self.device)

        random_z = torch.randn(num_audited_classes, self.generator.dim_z, device=self.device)

        # Optimize z, TODO: Optimize in batches

        if self.data_format == "dataloader":

            opt_z = self.optimize_z_grad(y=labels,
                                iter_times=self.configs.z_optimization_iter).to(self.device)

            # Compute image metrics for the optimized z and labels
            metrics = ImageMetrics(self.handler, self.gan_handler,
                                        reconstruction_configs,
                                        labels=labels,
                                        z=opt_z)
            # TODO: Implement a class with a .save function.

        elif self.data_format == "dataframe":
            # Accuracy of the target model on the random labels
            # initial_metrics = TabularMetrics(self.handler, self.gan_handler,
            #                             reconstruction_configs,
            #                             labels=labels,
            #                             z=random_z)
            # logger.info("INITIAL ACCURACY:", initial_metrics.results)

            # generate samples from the generator
            if self.handler.configs.target.model_type == "xgboost":
                opt_z = self.optimize_z_no_grad(y=labels,
                                iter_times=self.configs.z_optimization_iter)
            elif self.handler.configs.target.model_type == "pytorch_tabular":
                opt_z = self.optimize_z_grad(y=labels,
                                iter_times=self.configs.z_optimization_iter, augment=False)

            metrics = TabularMetrics(self.handler, self.gan_handler,
                                        reconstruction_configs,
                                        labels=labels,
                                        z=opt_z)

        return metrics
<<<<<<< Updated upstream
=======
        # return metrics
>>>>>>> Stashed changes

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
        z = torch.randn(bs, self.generator.dim_z, device=self.device)
        z.requires_grad = False

        #optimizer = torch.optim.Adam([z], lr=self.configs.z_optimization_lr)
        min_loss = 1e6

        for i in range(iter_times):
            z = torch.randn(bs, self.generator.dim_z, device=self.device)
            # Generate fake images
            fake = self.generator(z, y)

            if self.handler.configs.target.model_type == "pytorch_tabular":
                #y = fake["pseudo_label"].values
                #y = torch.tensor(y, dtype=torch.long).to(self.device)
                fake = fake.drop(columns=["pseudo_label"])

            out1 = self.target_model(aug_list(fake))
            #out2 = self.target_model(aug_list(fake))

            # if z.grad is not None:
            #     z.grad.data.zero_()

            # Compute inversion loss
            inv_loss = F.cross_entropy(out1, y) #+ F.cross_entropy(out2, y)

            if inv_loss < min_loss:
                min_loss = inv_loss
                # Save the best z
                best_z = z.clone()

            # Update the latent vector z
            # print("Before backward:", z.grad)
            # optimizer.zero_grad()
            # inv_loss.backward()
            # optimizer.step()

            # print("After backward:", z.grad)
            inv_loss_val = inv_loss.item()

            if (i + 1) % self.log_interval == 0:
                with torch.no_grad():
                    eval_prob = self.evaluation_model(fake)
                    eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                    acc = y.eq(eval_iden.long()).sum().item() * 1.0 / bs
                    logger.info("Iteration:{}\tInv Loss:{:.2f}\tAttack Acc:{:.2f}".format(i + 1, inv_loss_val, acc))

        return best_z

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
        self.target_model.eval()
        self.target_model.to(self.device)
        self.evaluation_model.eval()
        self.evaluation_model.to(self.device)


        # Use Optuna for Bayesian optimization
        def objective(trial: optuna.trial.Trial) -> float:
            # Suggest values for each dimension of z
            z_numpy = np.array([trial.suggest_float(f"z_{i}", -3, 3) for i in range(bs * self.generator.dim_z)])
            z_numpy = z_numpy.reshape(bs, self.generator.dim_z)  # Reshape to match the required shape

            z_tensor = torch.tensor(z_numpy, dtype=torch.float32, device=self.device)

            fake = self.generator(z_tensor, y)
            fake = fake.drop(columns=["pseudo_label"])  # Remove pseudo_label
            out1 = self.target_model(fake)

            # Cross entropy
            return F.cross_entropy(out1, y)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")

        logger.info("Optimizing z using Optuna")
        study.optimize(objective, n_trials=iter_times, show_progress_bar=True, n_jobs=-1)

        # Convert the dictionary values to a NumPy array
        z_numpy = np.array(list(study.best_params.values()))

        # Reshape the array to match the required shape (bs, self.generator.dim_z)
        z_numpy = z_numpy.reshape(bs, self.generator.dim_z)

        # Convert the NumPy array to a PyTorch tensor
        return torch.tensor(z_numpy, dtype=torch.float32, device=self.device)
