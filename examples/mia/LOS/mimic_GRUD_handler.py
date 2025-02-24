
from sklearn.metrics import accuracy_score
from torch import cuda, device, nn, optim, sigmoid
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakpro import AbstractInputHandler
from leakpro.schemas import TrainingOutput

class MimicInputHandlerGRU(AbstractInputHandler):
    """Class to handle the user input for the MIMICIII dataset."""

    def __init__(self, configs: dict) -> None:
        super().__init__(configs = configs)

    def get_criterion(self)->BCEWithLogitsLoss:
        """Set the CrossEntropyLoss for the model."""
        return BCEWithLogitsLoss()

    def get_optimizer(self, model:nn.Module) -> optim.Optimizer:
        """Set the optimizer for the model."""
        learning_rate = 0.01
        return optim.Adam(model.parameters(), lr=learning_rate)

    def train(
        self,
        dataloader: DataLoader,
        model: nn.Module = None,
        criterion: nn.Module = None,
        optimizer: optim.Optimizer = None,
        epochs: int = None,
    ) -> TrainingOutput:
        """Model training procedure."""
        device_name = device("cuda" if cuda.is_available() else "cpu")
        model.to(device_name)
        model.train()

        criterion = self.get_criterion()
        optimizer = self.get_optimizer(model)

        for e in tqdm(range(epochs), desc="Training Progress"):
            model.train()
            train_acc, train_loss = 0.0, 0.0
            all_predictions = []
            all_labels = []

            for _, (x, labels) in enumerate(tqdm(dataloader, desc="Training Batches")):
                x = x.to(device_name)
                labels = labels.to(device_name).float()

                optimizer.zero_grad()
                output = model(x)

                loss = criterion(output.squeeze(), labels.squeeze())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # Collect predictions and labels for last epoch
                binary_predictions = sigmoid(output).squeeze().round().cpu().detach().numpy()
                binary_labels = labels.squeeze().cpu().numpy().astype(int)

                all_predictions.extend(binary_predictions)
                all_labels.extend(binary_labels)

            # Ensure labels are integer and 1D
            binary_labels = labels.squeeze().cpu().numpy().astype(int)
            train_acc = accuracy_score(binary_labels, binary_predictions)
            train_loss = train_loss/len(dataloader)

        output_dict = {"model": model, "metrics": {"accuracy": train_acc, "loss": train_loss}}
        output = TrainingOutput(**output_dict)
        
        return output










