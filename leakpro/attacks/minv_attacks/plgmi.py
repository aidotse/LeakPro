"""Implementation of the PLGMI attack."""

from leakpro.attacks.minv_attacks.abstract_minv import AbstractMINV
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

        self.n_dis = None
        self.gen_lr = None
        self.dis_lr = None
        
        logger.info("Configuring PLG-MI attack")
        self._configure_attack(configs)

    def _configure_attack(self: Self, configs: dict) -> None:
        """Configure the attack parameters.

        Args:
        ----
            configs (dict): Configuration parameters for the attack.

        """
        self.n_dis = configs.get("n_dis", 5)
        self.gen_lr = configs.get("gen_lr", 0.0002)
        self.dis_lr = configs.get("dis_lr", 0.0002)


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

        # load generator
        self.gen = self.handler.get_generator()

        # load discriminator
        self.dis = self.handler.get_discriminator()

        self.dis_opt = self.dis.get_optimizer(self.dis_lr)
        self.gen_opt = self.gen.get_optimizer(self.gen_lr)


        # check if pre-trained models are available
        # if not, train the models
        if self.gen is None:
            logger.info("Training generator model")


        pass

    def run_attack(self:Self) -> MinvResult:
        """Run the attack."""
        pass