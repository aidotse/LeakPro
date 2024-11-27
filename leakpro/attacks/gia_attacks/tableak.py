"""TabLeak attack on federated learning."""

from torch.nn import Module
from torch.utils.data import DataLoader

from leakpro.attacks.gia_attacks.abstract_gia import AbstractGIA
from leakpro.metrics.attack_result import GIAResults
from leakpro.utils.import_helper import Callable, Self


class TabLeak(AbstractGIA):
    """TabLeak attack class for federated learning."""

    def __init__(self: Self, model: Module, client_loader: DataLoader, train_fn: Callable,) -> None:
        self.model = model
        self.client_loader = client_loader
        self.train_fn = train_fn

    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "TabLeak"
        reference_str = """Vero, Mark, et al. TabLeak - Tabular Data Leakage in Federated Learning? ICML, 2023."""
        summary_str = ""
        detailed_str = ""
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }

    def prepare_attack(self:Self) -> None:
        """Prepare the attack.

        Args:
        ----
            self (Self): The instance of the class.

        Returns:
        -------
            None

        """
        pass

    def run_attack(self:Self) -> GIAResults:
        """Run the attack and return the combined metric result.

        Returns
        -------
            GIAResults: The results of the attack.

        """
        pass

    def _configure_attack(self: Self, configs: dict) -> None:
        pass
