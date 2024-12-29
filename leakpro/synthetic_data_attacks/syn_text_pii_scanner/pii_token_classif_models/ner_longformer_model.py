"""Module containing tokenizer and modified LongFormer model to be used for NER/PII/token classification tasks."""
from typing import Any, Dict, Optional

import torch
from safetensors.torch import load_file
from transformers import LongformerModel, LongformerTokenizerFast
from transformers.modeling_outputs import TokenClassifierOutput

# Longformer base model name
longformer_base_model_name: str = "allenai/longformer-base-4096"

def get_tokenizer() -> LongformerTokenizerFast:
    """Create and return a LongformerTokenizerFast instance.

    Returns:
        LongformerTokenizerFast: The tokenizer for the specified model.

    """
    return LongformerTokenizerFast.from_pretrained(
        longformer_base_model_name,
        clean_up_tokenization_spaces = True
    )

class NERLongformerModel(torch.nn.Module):
    """Longformer based Name Entity recognition model.

    Model uses Longformer's architecture and adds a linear
    classification layer on top for token classification.
    """

    def __init__(self, *,
        num_labels: int,
        weights_crossentropy: Optional[torch.Tensor] = None,
        crossentropy_ignore_index: int = -100,
        device: Optional[torch.device] = None
    ) -> None:
        """Initialize the Model.

        Args:
            num_labels (int): The number of label classes for classification.
            weights_crossentropy (Optional[torch.Tensor]): Weights tensor for cross-entropy loss. Defaults to None.
            crossentropy_ignore_index (int): Ignore index for cross-entropy loss. Defaults to -100.
            device (Optional[torch.device]): torch.device to move cross-entropy loss to. Defaults to None.

        """
        super().__init__()
        # Validate weights_crossentropy
        if weights_crossentropy is not None:
            assert weights_crossentropy.dim() == 1, "weights_crossentropy.dim must be 1 in ner_longformer_model.py"
            assert weights_crossentropy.numel() == num_labels, "weights_crossentropy.numel and num_labels do not match in ner_longformer_model.py" # noqa: E501
        # Set longformer model
        self.longformer: LongformerModel = LongformerModel.from_pretrained(longformer_base_model_name)
        # Enable gradient computation for all parameters
        for param in self.longformer.parameters():
            param.requires_grad = True
        # Set num_labels
        self.num_labels = num_labels
        # Set classifier
        self.classifier: torch.nn.Linear = torch.nn.Linear(self.longformer.config.hidden_size, num_labels)
        # Set loss_fun
        self.loss_fun = torch.nn.CrossEntropyLoss(ignore_index=crossentropy_ignore_index, weight=weights_crossentropy)
        if device is not None and device.type == "cuda":
            self.loss_fun  = self.loss_fun.cuda()

    def state_dict(self, *args, **kwargs) -> Dict: # noqa: ANN002, ANN003
        """Modified state_dict fun for extra attributes."""
        # Get the original state_dict
        state = super().state_dict(*args, **kwargs)
        # Add num_labels and loss_fun.ignore index
        state["num_labels"] = torch.tensor(self.num_labels)
        state["loss_fun.ignore_index"] = torch.tensor(self.loss_fun.ignore_index)
        return state

    def load_state_dict(self, state_dict: Dict, strict: bool = True) -> None:
        """Modified load_state_dict fun for extra attributes."""
        #Get device
        device = self.loss_fun.weight.device
        #Restore num_labels and loss_fun.ignore index
        self.num_labels = int(state_dict.pop("num_labels").item())
        self.loss_fun.ignore_index = int(state_dict.pop("loss_fun.ignore_index").item())
        #Restore other weights shapes
        self.loss_fun.weight = torch.rand(self.num_labels, dtype=torch.float32).to(device)
        cfw1 = state_dict["classifier.weight"].shape[1]
        self.classifier.weight = torch.nn.Parameter(torch.rand((self.num_labels, cfw1), dtype=torch.float32).to(device))
        self.classifier.bias = torch.nn.Parameter(torch.rand(self.num_labels, dtype=torch.float32).to(device))
        # Load the rest of the state_dict as usual
        super().load_state_dict(state_dict, strict=strict)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> TokenClassifierOutput:
        """Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tensor of shape (batch_size, sequence_length).
            attention_mask (torch.Tensor): Tensor of shape (batch_size, sequence_length).
            labels (Optional[torch.Tensor]): Tensor of shape (batch_size, sequence_length) or None.

        Returns:
            torch.Tensor: Logits for each token, shape (batch_size, sequence_length, num_labels)

        """
        #Longformer model cannot manage more than 4096 tokens
        input_ids = input_ids[:, 0:4096].clone()
        attention_mask = attention_mask[:, 0:4096].clone()
        #Get logits
        outputs: Any = self.longformer(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        sequence_output: torch.Tensor = outputs.last_hidden_state
        logits: torch.Tensor = self.classifier(sequence_output)
        #Get loss
        loss = None
        if labels is not None:
            labels = labels[:, 0:4096].clone()
            loss = self.loss_fun(logits.view(-1, self.num_labels), labels.view(-1))
        return TokenClassifierOutput(
            loss = loss,
            logits = logits,
            hidden_states = None,
            attentions = None,
        )

def load_model(*,
    num_labels: int,
    weights_crossentropy: Optional[torch.Tensor] = None,
    crossentropy_ignore_index: int = -100,
    device: Optional[torch.device] = None,
    path: Optional[str] = None
) -> NERLongformerModel:
    """Inits and returns a NERLongformerModel on specified device.

    If path is provided, weights will be loaded to returned model.
    Note: path must point to a safetensors file.

    Args:
        num_labels (int): The number of label classes for classification.
        weights_crossentropy (Optional[torch.Tensor]): Weights tensor for cross-entropy loss. Defaults to None.
        crossentropy_ignore_index (int): Ignore index for cross-entropy loss. Defaults to -100.
        device (Optional[torch.device]): torch.device to move model and cross-entropy loss to. Defaults to None.
        path (Optinal[str]): if provided, weights will be loaded from safetensors file. Defaults to None.

    """
    # Instantiate model
    model = NERLongformerModel(
        num_labels = num_labels,
        weights_crossentropy = weights_crossentropy,
        crossentropy_ignore_index = crossentropy_ignore_index,
        device = device
    ).to(device)
    # Load weights
    if path is not None:
        state_dict = load_file(path)
        model.load_state_dict(state_dict)
    return model
