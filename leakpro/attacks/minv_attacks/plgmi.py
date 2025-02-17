"""Implementation of the PLGMI attack."""

from leakpro.attacks.minv_attacks.abstract_minv import AbstractMINV
from leakpro.attacks.utils.GANHandler import GANHandler
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
        # General parameters
        self.alpha = configs.get("alpha", 0.1)
        # Generator parameters
        self.gen_lr = configs.get("gen_lr", 0.0002) # Learning rate of the generator
        # Discriminator parameters
        self.n_dis = configs.get("n_dis", 5) # Number of discriminator updates per generator update
        self.dis_lr = configs.get("dis_lr", 0.0002)

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

    def prepare_attack(self:Self) -> None:
        """Prepare the attack."""
        # Load a generator, or train one if not given.
        # load generator
        self.gen , self.dis = GANHandler().create_gan()




        pass

    def run_attack(self:Self) -> MinvResult:
        """Run the attack."""
        # Use trained generator to generate samples and evaluate
        pass
