# TODO: add abstract parent class, allow loading from outside of the package with importlib.util.spec_from_file_location

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from typing import Type, Optional, Dict, Literal, Union, Any
from leakpro.utils.input_handler import get_class_from_module, import_module_from_file
from leakpro.dataset import GeneralDataset
import logging
from leakpro.user_code.parent_template import CodeHandler
from torch import cuda, device
from tqdm import tqdm

class Cifar10CodeHandler(CodeHandler):

    def __init__(self, configs: dict, logger:logging.Logger):
        super().__init__(configs = configs, logger = logger)


    def train_shadow_model(self, dataset_indices: np.ndarray) -> Dict[Literal["model", "metrics", "configuration"], Union[torch.nn.Module, Dict[str, Any]]]:

        # define hyperparams for training (dataloader ones are in get dataloader defined!):
        epochs = self.configs["shadow_model"]["epochs"]
        lr = self.configs["shadow_model"]["lr"]
        weight_decay = 0

        # create and initialize shadow model 
        shadow_train_loader = self.get_dataloader(dataset_indices)
        shadow_model_class = self.get_shadow_model_class()
        shadow_model = shadow_model_class(**self.get_shadow_model_init_params())

        # prepare training
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        shadow_model.to(gpu_or_cpu)
        shadow_model.train()

        # create optimizer and loss function
        optimizer = optim.SGD(shadow_model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
        loss_func = self.loss

        # training loop
        for epoch in range(epochs):
            train_loss, train_acc = 0, 0
            shadow_model.train()
            for inputs, labels in tqdm(shadow_train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                labels = labels.long()  # noqa: PLW2901
                inputs, labels = inputs.to(gpu_or_cpu, non_blocking=True), labels.to(gpu_or_cpu, non_blocking=True)  # noqa: PLW2901
                optimizer.zero_grad()
                outputs = shadow_model(inputs)
                loss = loss_func(outputs, labels)
                pred = outputs.data.max(1, keepdim=True)[1]
                loss.backward()

                optimizer.step()

                # Accumulate performance of shadow model
                train_acc += pred.eq(labels.data.view_as(pred)).sum()
                train_loss += loss.item()

            log_train_str = (
                f"Epoch: {epoch+1}/{epochs} | Train Loss: {train_loss/len(shadow_train_loader):.8f} | "
                f"Train Acc: {float(train_acc)/len(shadow_train_loader.dataset):.8f}")
            self.logger.info(log_train_str)
        shadow_model.to("cpu")

        # saving parameters
        configuration = {}
        configuration["init_params"] = self.get_shadow_model_init_params()
        configuration["train_indices"] = dataset_indices
        configuration["num_train"] = len(dataset_indices)
        configuration["optimizer"] = type(optimizer).__name__
        configuration["criterion"] = type(loss_func).__name__
        configuration["batch_size"] = shadow_train_loader.batch_size
        configuration["epochs"] = epochs
        configuration["learning_rate"] = lr
        configuration["weight_decay"] = weight_decay

        return {"model": shadow_model, "metrics": {"accuracy": train_acc, "loss": train_loss}, "configuration": configuration}

    # def get_signals_from_model(self, model: torch.nn.Module, dataloader: DataLoader) -> np.ndarray:
    #     logits = []
    #     true_indices = []
    #     for x, y in dataloader:
    #         with torch.no_grad():
    #             # Get logits for each data point
    #             logits_batch = model(x.to(model.device))
    #             # TODO: check if dimensions add up correctly
    #             logits.extend(logits_batch.tolist())
    #             true_indices.extend(y.tolist())
    #     logits = np.array(logits)
    #     true_indices = np.array(true_indices)
    #     signals = softmax(all_logits=logits, temperature = self.configs["audit"]["attack_list"]["rmia"]["temperature"] , true_label_indices=true_indices)
    #     return signals