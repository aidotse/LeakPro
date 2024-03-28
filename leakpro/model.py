"""Module containing the Model class, an interface to query a model without any assumption on how it is implemented."""

from abc import ABC, abstractmethod
from copy import deepcopy

# typing package not available form < python-3.11, typing_extensions backports new and experimental type hinting features to older Python versions
try:
    from typing import Callable, List, Self
except ImportError:
    from typing_extensions import Callable, List, Self

import numpy as np
import torch

from scipy.special import softmax

########################################################################################################################
# MODEL CLASS
########################################################################################################################


class Model(ABC):
    """Interface to query a model without any assumption on how it is implemented."""

    def __init__(self:Self, model_obj: torch.nn.Module, loss_fn: torch.nn.modules.loss._Loss) -> None:
        """Initialize the Model.

        Args:
        ----
            model_obj: Model object.
            loss_fn: Loss function.

        """
        self.model_obj = model_obj
        self.loss_fn = loss_fn

    @abstractmethod
    def get_logits(self:Self, batch_samples:np.ndarray) -> np.ndarray:
        """Get the model output from a given input.

        Args:
        ----
            batch_samples: Model input.

        Returns:
        -------
            Model output

        """
        pass

    @abstractmethod
    def get_loss(self:Self, batch_samples:np.ndarray, batch_labels: np.ndarray, per_point:bool=True) -> np.ndarray:
        """Get the model loss on a given input and an expected output.

        Args:
        ----
            batch_samples: Model input.
            batch_labels: Model expected output.
            per_point: Boolean indicating if loss should be returned per point or reduced.

        Returns:
        -------
            The loss value, as defined by the loss_fn attribute.

        """
        pass

    @abstractmethod
    def get_grad(self:Self, batch_samples:np.ndarray, batch_labels:np.ndarray) -> np.ndarray:
        """Get the gradient of the model loss with respect to the model parameters, given an input and an expected output.

        Args:
        ----
            batch_samples: Model input.
            batch_labels: Model expected output.

        Returns:
        -------
            A list of gradients of the model loss (one item per layer) with respect to the model parameters.

        """
        pass

    @abstractmethod
    def get_intermediate_outputs(self:Self,
                                 layers:List[int],
                                 batch_samples:np.ndarray,
                                 forward_pass: bool=True) -> List[np.ndarray]:
        """Get the intermediate output of layers (a.k.a. features), on a given input.

        Args:
        ----
            layers: List of integers and/or strings, indicating which layers values should be returned.
            batch_samples: Model input.
            forward_pass: Boolean indicating if a new forward pass should be executed. If True, then a forward pass is
                executed on batch_samples. Else, the result is the one of the last forward pass.

        Returns:
        -------
            A list of intermediate outputs of layers.

        """
        pass

class PytorchModel(Model):
    """Inherits from the Model class, an interface to query a model without any assumption on how it is implemented.

    This particular class is to be used with pytorch models.
    """
    def __init__(self:Self, model_obj:torch.nn.Module, loss_fn:torch.nn.modules.loss._Loss)->None:
        """Initialize the PytorchModel.

        Args:
        ----
            model_obj: Model object.
            loss_fn: Loss function.

        """
        # Imports torch with global scope
        globals()["torch"] = __import__("torch")

        # Initializes the parent model
        super().__init__(model_obj, loss_fn)

        # Add hooks to the layers (to access their value during a forward pass)
        self.intermediate_outputs = {}
        for _, layer in enumerate(list(self.model_obj._modules.keys())):
            getattr(self.model_obj, layer).register_forward_hook(self.__forward_hook(layer))

        # Create a second loss function, per point
        self.loss_fn_no_reduction = deepcopy(loss_fn)
        self.loss_fn_no_reduction.reduction = "none"

    def get_logits(self:Self, batch_samples:np.ndarray)->np.ndarray:
        """Get the model output from a given input.

        Args:
        ----
            batch_samples: Model input.

        Returns:
        -------
            Model output.

        """
        batch_samples_tensor = torch.tensor(
            np.array(batch_samples), dtype=torch.float32
        )
        return self.model_obj(batch_samples_tensor).detach().numpy()

    def get_loss(self:Self, batch_samples:np.ndarray, batch_labels:np.ndarray, per_point:bool=True)->np.ndarray:
        """Get the model loss on a given input and an expected output.

        Args:
        ----
            batch_samples: Model input.
            batch_labels: Model expected output.
            per_point: Boolean indicating if loss should be returned per point or reduced.

        Returns:
        -------
            The loss value, as defined by the loss_fn attribute.

        """
        batch_samples_tensor = torch.tensor(
            np.array(batch_samples), dtype=torch.float32
        )
        batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long)

        if per_point:
            return (
                self.loss_fn_no_reduction(
                    self.model_obj(batch_samples_tensor),
                    batch_labels_tensor,
                )
                .detach()
                .numpy()
            )
        return self.loss_fn(
            self.model_obj(torch.tensor(batch_samples_tensor)),
            torch.tensor(batch_labels_tensor),
        ).item()


    def get_grad(self:Self, batch_samples:np.ndarray, batch_labels:np.ndarray)->np.ndarray:
        """Get the gradient of the model loss with respect to the model parameters, given an input and expected output.

        Args:
        ----
            batch_samples: Model input.
            batch_labels: Model expected output.

        Returns:
        -------
            A list of gradients of the model loss (one item per layer) with respect to the model parameters.

        """
        loss = self.loss_fn(
            self.model_obj(torch.tensor(batch_samples)), torch.tensor(batch_labels)
        )
        loss.backward()
        return [p.grad.numpy() for p in self.model_obj.parameters()]

    def get_intermediate_outputs(self:Self,
                                 layers:List[int],
                                 batch_samples:np.ndarray,
                                 forward_pass:bool=True) -> List[np.ndarray]:
        """Get the intermediate output of layers (a.k.a. features), on a given input.

        Args:
        ----
            layers: List of integers and/or strings, indicating which layers values should be returned.
            batch_samples: Model input.
            forward_pass: Boolean indicating if a new forward pass should be executed. If True, then a forward pass is
                executed on batch_samples. Else, the result is the one of the last forward pass.

        Returns:
        -------
            A list of intermediate outputs of layers.

        """
        if forward_pass:
            _ = self.get_logits(torch.tensor(batch_samples))
        layer_names = []
        for layer in layers:
            if isinstance(layer, str):
                layer_names.append(layer)
            else:
                layer_names.append(list(self.model_obj._modules.keys())[layer])
        return [
            self.intermediate_outputs[layer_name].detach().numpy()
            for layer_name in layer_names
        ]

    def __forward_hook(self:Self, layer_name: str) -> Callable:
        """Private helper function to access outputs of intermediate layers.

        Args:
        ----
            layer_name: Name of the layer to access.

        Returns:
        -------
            A hook to be registered using register_forward_hook.

        """

        def hook(_: torch.Tensor, __: torch.Tensor, output: torch.Tensor) -> None:
            self.intermediate_outputs[layer_name] = output

        return hook

    def get_rescaled_logits(self, batch_samples, batch_labels, hingeloss=False):
        
        """Function to get the model rescaled logits on a given input and an expected output.
        The rescaled logits is proposed in https://arxiv.org/abs/2112.03570.

        Args:
            batch_samples: Model input.
            batch_labels: Model expected output.
            hingeloss: True/False, implementing the hingeloss version, as per Carlini et.el. paper
                    can be used if the final activation function is not used, i.e. if z(x) is available, where f(z(x))
                    is the whole model and f() is the last activation
            XXXX per_point: Boolean indicating if loss should be returned per point or reduced.

        Returns:
            The loss value, as defined by the loss_fn attribute.
        """

        hinge_loss = torch.nn.MultiMarginLoss(reduction='none')
        
        self.batch_size = 32
            
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
            
        self.model_obj.to(device)
        self.model_obj.eval()
        
        with torch.no_grad():
            rescaled_list = []
            batched_samples = torch.split(torch.tensor(np.array(batch_samples), dtype=torch.float32), self.batch_size)
            batched_labels = torch.split(torch.tensor(np.array(batch_labels), dtype=torch.float32), self.batch_size)
            
            for x, y in zip(batched_samples, batched_labels):
                COUNT = len(x)
                x = x.to(device)
                y = y.to(device)
                pred = self.model_obj(x)

                # Is this really unstble or is this the log sum_y'=/=y f(x)_y'
                if not hingeloss:
                    y = y.to("cpu").numpy().astype(int)
                    confi = softmax(pred.detach().cpu(), axis=1)
                    confi_corret = confi[np.arange(COUNT), y].astype(float)
                    confi[np.arange(COUNT), y] = 0
                    confi_wrong = np.sum(confi, axis=1)
                    # print("-----\n", np.log(confi_corret + 1e-10), "asdsa")
                    # print(np.log(confi_wrong + 1e-10), "asdsa")
                    logit = np.log(confi_corret + 1e-16) - np.log(confi_wrong + 1e-16)
                    # print(logit, "asdsa")
                    # print(hinge_loss(pred.detach().to("cpu"), torch.from_numpy(y)), "\n")
                else:
                    logit = hinge_loss(pred.detach(), y).numpy()
                
                rescaled_list.append(logit)
            all_rescaled_logits = np.concatenate(rescaled_list)
        self.model_obj.to("cpu")
        return all_rescaled_logits
        