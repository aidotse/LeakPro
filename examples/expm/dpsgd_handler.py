
from torch import cuda, device, nn, optim, squeeze
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from leakpro import AbstractInputHandler
import os
import pickle
from opacus.accountants.utils import get_noise_multiplier
from opacus import PrivacyEngine, GradSampleModule


class MimicInputHandlerGRU(AbstractInputHandler):
    """Class to handle the user input for the MIMICIII dataset."""

    def __init__(self, configs: dict) -> None:
        super().__init__(configs = configs)

    def get_criterion(self)->CrossEntropyLoss:
        """Set the CrossEntropyLoss for the model."""
        return CrossEntropyLoss()

    def get_optimizer(self, model:nn.Module) -> optim.Optimizer:
        """Set the optimizer for the model."""
        learning_rate = 0.01
        return optim.Adam(model.parameters(), lr=learning_rate)

    def convert_to_device(self, x):
        device_name = device("cuda" if cuda.is_available() else "cpu")
        return x.to(device_name)

    def to_numpy(self, tensor) :
        return tensor.detach().cpu().numpy() if tensor.is_cuda else tensor.detach().numpy()

    def train(
        self,
        dataloader: DataLoader,
        model: nn.Module = None,
        criterion: nn.Module = None,
        optimizer: optim.Optimizer = None,
        epochs: int = None,
    ) -> dict:

        if self.configs['audit']['dpsgd']:
            print("Training shadow models with DP-SGD")
            dpsgd_path = self.configs['audit']['dpsgd']['dpsgd_path']

            sample_rate = 1/len(dataloader) 
            # Check if the file exists
            if os.path.exists(dpsgd_path):
                # Open and read the pickle file
                with open(dpsgd_path, "rb") as file:
                    data = pickle.load(file)
                print("Pickle file loaded successfully!")
                print("Data:", data)
            else:
                print(f"File not found at: {dpsgd_path}")
            try:
                noise_multiplier = get_noise_multiplier(target_epsilon = 2,
                                                    target_delta = data['delta'],
                                                    sample_rate = data['sample_rate'],
                                                    epochs = 21,
                                                    epsilon_tolerance = data['epsilon_tolerance'],
                                                    accountant = 'prv',
                                                    eps_error = data['eps_error'])
            except:
                # the prv accountant is not robust to large epsilon (even epsilon = 10)
                # so we will use rdp when it fails, so the actual epsilon may be slightly off
                # see https://github.com/pytorch/opacus/issues/604
                noise_multiplier = get_noise_multiplier(target_epsilon = 2,
                                                        target_delta = data['delta'],
                                                        sample_rate = sample_rate,
                                                        epochs = 21,
                                                        epsilon_tolerance = 0.01,
                                                        accountant = 'rdp')
            # make the model private
            privacy_engine = PrivacyEngine(accountant = 'prv')
            model, optimizer, dataloader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=dataloader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=1,
            )




        device_name = device("cuda" if cuda.is_available() else "cpu")
        model.to(device_name)
        model.train()

        criterion = self.get_criterion()
        optimizer = self.get_optimizer(model)

        for e in tqdm(range(epochs), desc="Training Progress"):
            model.train()
            train_acc, train_loss = 0.0, 0.0

            for _, (x, labels) in enumerate(tqdm(dataloader, desc="Training Batches")):
                x = self.convert_to_device(x)
                labels = self.convert_to_device(labels)
                labels = labels.long()

                optimizer.zero_grad()
                output = model(x)

                loss = criterion(squeeze(output), squeeze(labels).long())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss = train_loss/len(dataloader)
            binary_predictions = self.to_numpy(output).argmax(axis=1)

            # Ensure labels are integer and 1D
            binary_labels = self.to_numpy(labels).astype(int)
            # Compute accuracy
            train_acc = accuracy_score(binary_labels, binary_predictions)

        return {"model": model, "metrics": {"accuracy": train_acc, "loss": train_loss}}