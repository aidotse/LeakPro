from typing import Self
from torch import Tensor, cuda, device, optim, no_grad, from_numpy, unique
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch import nn

from leakpro import AbstractInputHandler
from leakpro.schemas import TrainingOutput, EvalOutput

class MIMICInputHandler(AbstractInputHandler):

    def train(self, dataloader, model, criterion, optimizer, epochs) -> TrainingOutput:
        model_name = model.__class__.__name__
        train_fn_map = {
            "LR": self.train_LR,
            "GRUD": self.train_GRUD,
        }

        if model_name not in train_fn_map:
            raise ValueError(f"Unsupported model type: {model_name}")

        return train_fn_map[model_name](dataloader, model, criterion, optimizer, epochs)


    def eval(self, loader, model, criterion):
        """Selects the appropriate evaluation method based on the model type."""
        model_name = model.__class__.__name__

        eval_fn_map = {
            "LR": self.eval_LR,
            "GRUD": self.eval_GRUD,
        }

        if model_name not in eval_fn_map:
            raise ValueError(f"Unsupported model type '{model_name}'. Expected one of {list(eval_fn_map.keys())}.")

        return eval_fn_map[model_name](loader, model, criterion)


    def train_GRUD(self):
        pass

    def eval_GRUD(self):
        pass

    def train_LR(self,
        dataloader: DataLoader,
        model: nn.Module = None,
        criterion: nn.Module = None,
        optimizer: optim.Optimizer = None,
        epochs: int = None,
    ) -> TrainingOutput:
        """Model training procedure."""

        # prepare training
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)

        accuracy_history = []
        loss_history = []

        # training loop
        for epoch in range(epochs):
            train_loss, train_acc, total_samples = 0, 0, 0
            model.train()
            for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                labels = labels.float().unsqueeze(1)
                inputs, labels = inputs.to(gpu_or_cpu, non_blocking=True), labels.to(gpu_or_cpu, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                pred = outputs >= 0.5
                loss.backward()
                optimizer.step()

                # Accumulate performance of shadow model
                train_acc += pred.eq(labels.view_as(pred)).sum().item()
                total_samples += labels.size(0)
                train_loss += loss.item() * labels.size(0)

            avg_train_loss = train_loss / total_samples
            train_accuracy = train_acc / total_samples

            accuracy_history.append(train_accuracy)
            loss_history.append(avg_train_loss)

        results = EvalOutput(accuracy = train_accuracy,
                             loss = avg_train_loss,
                             extra = {"accuracy_history": accuracy_history, "loss_history": loss_history})
        return TrainingOutput(model = model, metrics=results)

    def eval_LR(self, loader, model, criterion):
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)
        model.eval()
        loss, acc = 0, 0
        total_samples = 0

        with no_grad():
            for data, target in loader:
                data, target = data.to(gpu_or_cpu), target.to(gpu_or_cpu)
                target = target.float().unsqueeze(1)
                output = model(data)
                loss += criterion(output, target).item()
                pred = (output) >= 0.5
                acc += pred.eq(target).sum().item()
                total_samples += target.size(0)
            loss /= len(loader)
            acc = float(acc) / total_samples

        output_dict = {"accuracy": acc, "loss": loss}
        return EvalOutput(**output_dict)

    def to_3D_tensor(df):
        idx = pd.IndexSlice
        np_3D = np.dstack([df.loc[idx[:, :, :, i], :].values for i in sorted(set(df.index.get_level_values("hours_in")))])
        return from_numpy(np_3D)



    class UserDataset(AbstractInputHandler.UserDataset):
        """
        A custom dataset class for handling user data.

        Args:
            x (torch.Tensor): The input features as a torch tensor.
            y (torch.Tensor): The target labels as a torch tensor.
            
        Methods:
            __len__(): Returns the length of the dataset.
            __getitem__(idx): Returns the item at the given index.
            subset(indices): Returns a subset of the dataset based on the given indices.
        """

        def __init__(self, data, targets, **kwargs):
            """
            Args:
                data (torch.Tensor): The input features as a torch tensor.
                targets (torch.Tensor): The target labels as a torch tensor.
                mean (Tensor, optional): Precomputed mean for normalization.
                std (Tensor, optional): Precomputed std for normalization.
            """
            assert data.shape[0] == targets.shape[0], "Mismatch between number of samples in data and targets"
            assert set(unique(targets.int()).tolist()).issubset({0, 1}), "Target labels should be either 0 or 1"

            # Ensure both x and y are converted to tensors (float32 type)
            self.data = Tensor(data).float()
            self.targets = Tensor(targets).float()

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, idx):
            x = self.data[idx]
            y = self.targets[idx]
            return x, y.squeeze(0)
        
from leakpro.input_handler.mia_handler import MIAHandler

MIAHandler.train_LR = MIMICInputHandler.train_LR
MIAHandler.eval_LR = MIMICInputHandler.eval_LR
MIAHandler.train_GRUD = MIMICInputHandler.train_GRUD
MIAHandler.eval_GRUD = MIMICInputHandler.eval_GRUD

        



