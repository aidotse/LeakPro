
from torch import cuda, device, nn, optim, squeeze
from torch.nn import BCEWithLogitsLoss
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

    def get_criterion(self)->BCEWithLogitsLoss:
        """Set the CrossEntropyLoss for the model."""
        return BCEWithLogitsLoss()

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


        print("Training shadow models with DP-SGD")
        dpsgd_path = self.configs['audit']['dpsgd']['dpsgd_path']

        sample_rate = 1/len(dataloader) 
        # Check if the file exists
        if os.path.exists(dpsgd_path):
            # Open and read the pickle file
            with open(dpsgd_path, "rb") as file:
                privacy_engine_dict = pickle.load(file)
            print("Pickle file loaded successfully!")
            print("Data:", privacy_engine_dict)
        else:
            raise Exception(f"File not found at: {dpsgd_path}")
                
        try:
            noise_multiplier = get_noise_multiplier(target_epsilon = privacy_engine_dict["target_epsilon"],
                                            target_delta = privacy_engine_dict["target_delta"],
                                            sample_rate = sample_rate ,
                                            epochs = privacy_engine_dict["epochs"],
                                            epsilon_tolerance = privacy_engine_dict["epsilon_tolerance"],
                                            accountant = 'prv',
                                            eps_error = privacy_engine_dict["eps_error"],)
        except:
            # the prv accountant is not robust to large epsilon (even epsilon = 10)
            # so we will use rdp when it fails, so the actual epsilon may be slightly off
            # see https://github.com/pytorch/opacus/issues/604
            noise_multiplier = get_noise_multiplier(target_epsilon = 2,
                                                    target_delta = privacy_engine_dict["target_delta"],
                                                    sample_rate = sample_rate,
                                                    epochs = privacy_engine_dict["epochs"],
                                                    epsilon_tolerance = privacy_engine_dict["epsilon_tolerance"],
                                                    accountant = 'rdp')

        # make the model private
        privacy_engine = PrivacyEngine(accountant = 'prv')
        model, optimizer, dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm= privacy_engine_dict["max_grad_norm"],
        )

        device_name = device("cuda" if cuda.is_available() else "cpu")
        model.to(device_name)
        model.train()

        criterion = self.get_criterion()

        for e in tqdm(range(epochs), desc="Training Progress"):
            model.train()
            train_acc, train_loss = 0.0, 0.0

            for _, (x, labels) in enumerate(tqdm(dataloader, desc="Training Batches")):
                if x.numel() == 0:  # Skip empty batches
                    continue

                x = self.convert_to_device(x)
                labels = self.convert_to_device(labels)
                labels = labels.float()

                optimizer.zero_grad()
                output = model(x).squeeze(dim=1) 

                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()

            train_loss = train_loss/len(dataloader)
            binary_predictions = (output > 0).float().cpu().numpy()

            binary_labels = labels.cpu().numpy()
            # Compute accuracy
            train_acc = accuracy_score(binary_labels, binary_predictions)

        return {"model": model, "metrics": {"accuracy": train_acc, "loss": train_loss}}