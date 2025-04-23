"""Implementation of the PLGMI attack."""
from typing import Any, Dict, Optional

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
from leakpro.metrics.attack_result import MinvResult
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
        discriminator: GANConfig = Field(..., description="Configuration for the discriminator")

        # Latent space parameters
        dim_z: int = Field(128, ge=1, description="Dimension of the latent space")
        z_optimization_iter: int = Field(1000, ge=1, description="Number of iterations for optimizing z")
        z_optimization_lr: float = Field(0.0002, ge=0.0, description="Learning rate for optimizing z")


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

        # Call the parent class constructor
        super().__init__(handler)

        # Assign the configuration parameters to the object
        for key, value in self.configs.model_dump().items():
            setattr(self, key, value)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_classes = self.handler.get_num_classes()

        self.configs.generator.init_params["num_classes"] = self.num_classes
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


    def top_n_selection(self:Self) -> DataLoader:
        """"Top n selection of pseudo labels."""
        # TODO: This does not scale well. Consider creating a class for the dataloader and implementing the __getitem__ method.
        logger.info("Performing top-n selection for pseudo labels")
        self.target_model.eval()
        self.target_model.to(self.device)
        all_confidences = []
        for entry, _ in self.public_dataloader:
            with torch.no_grad():
                outputs = self.target_model(entry.to(self.device))
                confidences = F.softmax(outputs, dim=1)
                all_confidences.append(confidences)
        # Concatenate all confidences
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
        for i in range(self.num_classes):
            for index, _ in top_n_pseudo_map[i]:
                # Append the image and pseudo label (index i) to the pseudo data
                pseudo_data.append((self.public_dataloader.dataset[index][0], i))

        logger.info("Created pseudo dataloader")
        # pseudo_data is now a list of tuples (image, pseudo_label)
        # We want to set the default device to the sampler in the returned dataloader to be on device
        return DataLoader(pseudo_data, batch_size=self.batch_size, shuffle=True, generator=torch.Generator(device=self.device))


    def prepare_attack(self:Self) -> None:
        """Prepare the attack."""
        logger.info("Preparing attack")

        # Get the target model from the handler
        self.target_model = self.handler.target_model

        self.gan_handler = GANHandler(self.handler, configs=self.configs)
        # TODO: Change structure of how we load data, handler or model_handler should do this, not gan_handler
        # Get public dataloader
        self.public_dataloader = self.handler.get_public_dataloader(self.configs.batch_size)

        # Get discriminator
        self.discriminator = self.gan_handler.get_discriminator()

        # Get generator
        self.generator = self.gan_handler.get_generator()

        # Set Adam optimizer for both generator and discriminator
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.gen_lr,
                                              betas=(self.gen_beta1, self.gen_beta2))
        self.dis_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.dis_lr,
                                               betas=(self.dis_beta1, self.dis_beta2))

        # Train the GAN
        if not self.gan_handler.trained_bool:
            logger.info("GAN not trained, getting psuedo labels")
            # Top-n-selection to get pseudo labels
            self.pseudo_loader = self.top_n_selection()
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
                                        checkpoint_interval = self.checkpoint_interval,
                                        n_dis  = self.n_dis,
                                        device = self.device,
                                        alpha = self.alpha,
                                        log_interval = self.log_interval,
                                        sample_from_generator = lambda: \
                                            self.gan_handler.sample_from_generator(batch_size=self.batch_size))
            # Save generator
            # self.gan_handler.save_generator(self.generator,
            #                                 self.output_dir + "/trained_models/plgmi_generator.pth")  # noqa: ERA001
            self.gan_handler.trained_bool = True
        else:
            logger.info("GAN already trained, skipping training")

        # Save the trained generator


    def run_attack(self:Self) -> MinvResult:
        """Run the attack."""
        logger.info("Running the PLG-MI attack")
        # Define image metrics class

        self.evaluation_model = self.target_model # TODO: Change to evaluation model

        reconstruction_configs = self.handler.configs.audit.reconstruction

        num_audited_classes = reconstruction_configs.num_audited_classes

        # Get random labels
        labels = torch.randint(0, self.num_classes, (num_audited_classes,)).to(self.device)

        # Optimize z, TODO: Optimize in batches
        opt_z = self.optimize_z(y=labels, lr= self.configs.z_optimization_lr,
                                iter_times=self.configs.z_optimization_iter).to(self.device)

        # Compute image metrics for the optimized z and labels
        image_metrics = ImageMetrics(self.handler, self.gan_handler,
                                     reconstruction_configs,
                                     labels=labels,
                                     z=opt_z)
        logger.info(image_metrics.results)
        # TODO: Implement a class with a .save function.
        return image_metrics

    def optimize_z(self:Self,
                   y: torch.tensor,
                   lr: float =2e-2,
                   iter_times: int = 10) -> torch.tensor:
        """Find the optimal latent vectors z for labels y.

        Args:
        ----
            y (torch.tensor): The class labels.
            lr (float): The learning rate for optimization.
            iter_times (int): The number of iterations for optimization.

        """
        bs = y.shape[0] # Number of samples
        y = y.view(-1).long().to(self.device)

        self.generator.eval()
        self.generator.to(self.device)
        self.target_model.eval()
        self.target_model.to(self.device)
        self.evaluation_model.eval()
        self.evaluation_model.to(self.device)

        aug_list = augmentation.container.ImageSequential(
            augmentation.RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            augmentation.ColorJitter(brightness=0.2, contrast=0.2),
            augmentation.RandomHorizontalFlip(),
            augmentation.RandomRotation(5),
        ).to(self.device) # TODO: Move this to a image modality extension and have it as an input

        logger.info("Optimizing z for the PLG-MI attack")

        z = torch.randn(bs, self.generator.dim_z, device=self.device)
        z.requires_grad = True

        optimizer = torch.optim.Adam([z], lr=lr)

        for i in range(iter_times):

            # Generate fake images
            fake = self.generator(z, y)

            # Average the output of the target model
            out1 = self.target_model(aug_list(fake))
            out2 = self.target_model(aug_list(fake))

            if z.grad is not None:
                z.grad.data.zero_()

            # Compute the loss
            inv_loss = F.cross_entropy(out1, y) + F.cross_entropy(out2, y)

            # Update the latent vector z
            optimizer.zero_grad()
            inv_loss.backward()
            optimizer.step()

            inv_loss_val = inv_loss.item()

            if (i + 1) % self.log_interval == 0:
                with torch.no_grad():
                    fake_img = self.generator(z, y)
                    eval_prob = self.evaluation_model(fake_img)
                    eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                    acc = y.eq(eval_iden.long()).sum().item() * 1.0 / bs
                    logger.info("Iteration:{}\tInv Loss:{:.2f}\tAttack Acc:{:.2f}".format(i + 1, inv_loss_val, acc))

        return z
