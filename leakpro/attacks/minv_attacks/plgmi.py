"""Implementation of the PLGMI attack."""
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

from leakpro.attacks.minv_attacks.abstract_minv import AbstractMINV
from leakpro.attacks.utils.gan_handler_2 import GANHandler
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.metrics.attack_result import MinvResult
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class AttackPLGMI(AbstractMINV):
    """Class that implements the PLGMI attack."""

    def __init__(self: Self, handler: AbstractInputHandler, configs: dict) -> None:
        super().__init__(handler)
        """Initialize the PLG-MI attack.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.
            configs (dict): Configuration parameters for the attack.

        """

        logger.info("Configuring PLG-MI attack")
        self._configure_attack(configs)

    def _configure_attack(self: Self, configs: dict) -> None:
        """Configure the attack parameters.

        Args:
        ----
            configs (dict): Configuration parameters for the attack.

        """
        # TODO: There are some optimizer specific parameters that need to be set here.
        self.num_classes = configs.get("num_classes") # TODO: fail check
        self.batch_size = configs.get("batch_size", 32)
        self.top_n = configs.get("top_n", 10)
        # General parameters
        self.alpha = configs.get("alpha", 0.1)
        # Generator parameters
        self.gen_lr = configs.get("gen_lr", 0.0002) # Learning rate of the generator
        # Discriminator parameters
        self.n_dis = configs.get("n_dis", 5) # Number of discriminator updates per generator update
        self.dis_lr = configs.get("dis_lr", 0.0002)
        self.pseudo_label_path = configs.get("pseudo_label_path")

        # Define the validation dictionary as: {parameter_name: (parameter, min_value, max_value)}
        validation_dict = {
            # alpha, 0 to inf
            "alpha": (self.alpha, 0, 1000), # 0 to inf
            "n_dis": (self.n_dis, 1, 1000), # 1 to inf
        }

        # Validate parameters
        for param_name, (param_value, min_val, max_val) in validation_dict.items():
            self._validate_config(param_name, param_value, min_val, max_val)


    def description(self:Self) -> dict:
        """Return the description of the attack."""
        title_str = "PLG-MI Attack"
        reference_str = "https://arxiv.org/abs/2302.09814"
        summary_str = "This attack is a model inversion attack that uses the PLG-MI algorithm."
        detailed_str = ""
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }


    def top_n_selection(self:Self) -> DataLoader:
        """"Top n selection of pseudo labels."""
        self.target_model.eval()
        all_confidences = []
        for images, _ in self.public_dataloader:
            with torch.no_grad():
                outputs = self.target_model(images)
                confidences = F.softmax(outputs, dim=1)
                all_confidences.append(confidences)
        # Concatenate all confidences
        self.confidences = torch.cat(all_confidences)
        # Get the pseudo labels
        pseudo_labels = torch.max(self.confidences, dim=1)

        # Empty array of size num_classes to store the pseudo labels
        pseudo_map = [[] for _ in range(self.num_classes)]

        for i, (conf, label) in enumerate(zip(pseudo_labels[0], pseudo_labels[1])):
            pseudo_map[label.item()].append((i, conf.item()))

        # Sort pseudo_map by confidence descending
        for i in range(self.num_classes):
            pseudo_map[i] = sorted(pseudo_map[i], key=lambda x: x[1], reverse=True)

        # keep only top_n entries in each element of pseudo_map
        pseudo_map = [pseudo_map[i][:self.top_n] for i in range(self.num_classes)]

        # Collect indices and pseudo-labels
        selected_indices = []
        pseudo_labels = []

        for pseudo_label, entries in enumerate(pseudo_map):
            for index, _ in entries:  # Ignore confidence scores
                selected_indices.append(index)
                pseudo_labels.append(pseudo_label)

        # Convert pseudo labels to tensor
        pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long)

        # Fetch data from public dataset using selected indices
        selected_images = [self.public_dataloader.dataset[i][0] for i in selected_indices]
        selected_images = torch.stack(selected_images)

        # Combine images and pseudo labels into a TensorDataset
        pseudo_dataset = TensorDataset(selected_images, pseudo_labels)

        # Create DataLoader #TODO: High risk, check if this is correct
        return DataLoader(
            pseudo_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )


    def prepare_attack(self:Self) -> None:
        """Prepare the attack."""
        logger.info("Preparing attack")
        self.gan_handler = GANHandler(self.handler)
        self.generator = self.gan_handler.get_generator()
        self.discriminator = self.gan_handler.get_discriminator()

        # Get public dataloader
        self.public_dataloader = self.gan_handler.get_public_data(self.batch_size)

        # Get pseudo_loader
        self.pseudo_loader = self.top_n_selection()

        

    def run_attack(self:Self) -> MinvResult:
        """Run the attack."""
        # Use trained generator to generate samples and evaluate
        pass
