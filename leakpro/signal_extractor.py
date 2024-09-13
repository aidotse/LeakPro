"""Module containing the Model class, an interface to query a model without any assumption on how it is implemented."""

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from torch import IntTensor, Tensor, cat, cuda, exp, flatten, log, max, nn, no_grad, sigmoid, sum
from torch.utils.data import DataLoader

from leakpro.import_helper import Callable, List, Optional, Self, Tuple
from leakpro.signals.utils.HopSkipJumpDistance import HopSkipJumpDistance


class Model(ABC):
    """Interface to query a model without any assumption on how it is implemented."""

    def __init__(self:Self, model_obj:nn.Module, loss_fn: nn.modules.loss._Loss) -> None:
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

    def __init__(self:Self, model_obj:nn.Module, loss_fn:nn.modules.loss._Loss)->None:
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

        device = "cuda:0" if cuda.is_available() else "cpu"
        self.model_obj.to(device)
        self.model_obj.eval()

        with no_grad():
            logits = self.model_obj(batch_samples.to(device))

        self.model_obj.to("cpu")
        return logits.cpu().numpy()

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
        batch_samples_tensor = Tensor(np.array(batch_samples))
        batch_labels_tensor = batch_labels.clone().detach()

        if batch_labels_tensor.dim() == 1:
            batch_labels_tensor = batch_labels_tensor.unsqueeze(1)

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
            self.model_obj(Tensor(batch_samples_tensor)),
            Tensor(batch_labels_tensor),
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
            self.model_obj(Tensor(batch_samples)), Tensor(batch_labels)
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
            _ = self.get_logits(Tensor(batch_samples))
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

        def hook(_: Tensor, __: Tensor, output: Tensor) -> None:
            self.intermediate_outputs[layer_name] = output

        return hook

    def get_rescaled_logits(self:Self, batch_samples:np.ndarray, batch_labels:np.ndarray) -> np.ndarray:
            """Get the rescaled logits of the model on a given input and expected output.

            Args:
            ----
                batch_samples: Model input.
                batch_labels: Model expected output.

            Returns:
            -------
                The rescaled logit value.

            """

            device = "cuda:0" if cuda.is_available() else "cpu"
            self.model_obj.to(device)
            self.model_obj.eval()
            with no_grad():

                x = batch_samples.to(device)
                y = batch_labels.to(device)
                all_logits = self.model_obj(x)

                if all_logits.shape[1] == 1:
                    positive_class_prob = sigmoid(all_logits)
                    predictions = cat([1 - positive_class_prob, positive_class_prob], dim=1)
                else:
                    predictions = all_logits - max(all_logits, dim=1, keepdim=True).values
                    predictions = exp(predictions)
                    predictions = predictions/sum(predictions,dim=1, keepdim=True)

                count = predictions.shape[0]
                y_true = predictions[np.arange(count), y.type(IntTensor)]
                predictions[np.arange(count), y.type(IntTensor)] = 0

                y_wrong = sum(predictions, dim=1)
                output_signals = flatten(log(y_true+1e-45) - log(y_wrong+1e-45)).cpu().numpy()

            self.model_obj.to("cpu")
            return output_signals

    def get_hop_skip_jump_distance(self:Self,  # noqa: D417
                                    data_loader: DataLoader,
                                    norm: int ,
                                    y_target: Optional[int] ,
                                    image_target: Optional[int] ,
                                    initial_num_evals: int ,
                                    max_num_evals: int ,
                                    stepsize_search: str,
                                    num_iterations: int ,
                                    gamma: float,
                                    constraint: int ,
                                    batch_size: int,
                                    epsilon_threshold: float,
                                    verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the hop-skip-jump distance for a given set of inputs.

        Args:
        ----
            data_loader: DataLoader object containing the input data.
            norm: Integer indicating the norm to be used for distance calculation.
            y_target: Optional integer indicating the target class label.
            image_target: Optional integer indicating the target image index.
            initial_num_evals: Integer indicating the initial number of evaluations.
            max_num_evals: Integer indicating the maximum number of evaluations.
            stepsize_search: String indicating the stepsize search method.
            num_iterations: Integer indicating the number of iterations.
            gamma: Float indicating the gamma value.
            constraint: Integer indicating the constraint value.
            batch_size: Integer indicating the batch size.
            verbose: Boolean indicating whether to print verbose output.
            clip_min: Float indicating the minimum clipping value.
            clip_max: Float indicating the maximum clipping value.
            epsilon_threshold: Float indicating the epsilon threshold.

        Returns:
        -------
            A tuple containing the perturbed images and the hop-skip-jump distances.

        """
        hop_skip_jump_instance= HopSkipJumpDistance(self.model_obj,
                                                    data_loader,
                                                    norm,
                                                    y_target,
                                                    image_target,
                                                    initial_num_evals,
                                                    max_num_evals,
                                                    stepsize_search,
                                                    num_iterations,
                                                    gamma,
                                                    constraint,
                                                    batch_size,
                                                    epsilon_threshold,
                                                    verbose)
        hop_skip_jump_perturbed_img, hop_skip_jump_distances = hop_skip_jump_instance.hop_skip_jump()
        return hop_skip_jump_perturbed_img, hop_skip_jump_distances
