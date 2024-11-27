from leakpro.input_handler.abstract_gia_input_handler import AbstractGIAInputHandler
from leakpro.fl_utils.gia_optimizers import MetaOptimizer
from leakpro.fl_utils.gia_module_to_functional import MetaModule

from torch.nn import Module, BCEWithLogitsLoss
from torch import cuda, device, optim
from torch.utils.data import DataLoader

class adultGiaHandler(AbstractGIAInputHandler):

    def __init__(self, configs: dict, dataloader:DataLoader) -> None:
        super().__init__(configs)
        self.dataloader = dataloader
    
    def get_client_dataloader(self):
         return self.dataloader
    
    def get_criterion(self)->None:
        """Set the CrossEntropyLoss for the model."""
        return BCEWithLogitsLoss()

    def get_optimizer(self, model:Module) -> None:
        """Set the optimizer for the model."""
        learning_rate = 0.1
        momentum = 0.8
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
    def train(
            dataloader: DataLoader,
            model: Module = None,
            criterion: Module = None,
            optimizer: MetaOptimizer = None,
            epochs: int = None,
        ) -> dict:
            """Model training procedure."""

            dev = device("cuda" if cuda.is_available() else "cpu")
            model.to(dev)
            patched_model = MetaModule(model)
            
            for e in range(epochs):
                for data, target in dataloader:
                    target = target.float().unsqueeze(1)
                    data, target = data.to(dev, non_blocking=True), target.to(dev, non_blocking=True)
                    output = patched_model(data, patched_model.parameters)
                    loss = criterion(output, target)                
                    patched_model.parameters = optimizer.step(loss, patched_model.parameters)
            return patched_model
