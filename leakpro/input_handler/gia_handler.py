"""GIA handler module."""

from torch.nn import Module
from torch.utils.data import DataLoader

from leakpro.fl_utils.gia_optimizers import MetaAdam, MetaOptimizer, MetaSGD
from leakpro.input_handler.abstract_gia_input_handler import AbstractGIAInputHandler

optimizer_mapping = {
    "sgd": MetaSGD,
    "adam": MetaAdam
}

class GIAHandler:
    """GIA handler class."""

    def __init__(self, user_handler:AbstractGIAInputHandler,
                 global_model:Module,
                 meta_data:dict,
                 client_dataloader:DataLoader) -> None:
        super().__init__()

        # Transfer specified attributes from user_handler to GIAHandler
        for attr in ["train", "get_criterion", "get_optimizer"]:
            if hasattr(user_handler, attr):
                setattr(self, attr, getattr(user_handler, attr))
            else:
                raise AttributeError(f"{attr} is missing in user_handler.")

        # map optimizer to MetaOptimizer
        supported_optimizers = ["sgd", "adam"]
        user_provided_optimizer = type(self.get_optimizer()).__name__.lower()
        if user_provided_optimizer in supported_optimizers:
            self.optimizer = optimizer_mapping[user_provided_optimizer]
        else:
            raise ValueError(f"Optimizer {user_provided_optimizer} is not supported.")

        # load target model and target model metadata
        self.global_model = global_model
        self.meta_data = meta_data
        self.client_dataloader = client_dataloader

