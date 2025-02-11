"""Module containing tokenizer and modified LongFormer model to be used for NER/PII/token classification tasks."""
from typing import Any, Optional

import torch
from huggingface_hub import PyTorchModelHubMixin
from transformers import LongformerModel, LongformerTokenizerFast
from transformers.modeling_outputs import TokenClassifierOutput

from leakpro.utils.logger import logger

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

class NERLongformerModel(torch.nn.Module, PyTorchModelHubMixin):
    """Longformer based Name Entity recognition model.

    Model uses Longformer's architecture and adds a linear
    classification layer on top for token classification.
    """

    def __init__(self, *,
        num_labels: int,
        weights_crossentropy: Optional[torch.Tensor] = None,
        crossentropy_ignore_index: int = -100
    ) -> None:
        """Initialize the Model.

        Args:
            num_labels (int): The number of label classes for classification.
            weights_crossentropy (Optional[torch.Tensor]): Weights tensor for cross-entropy loss. Defaults to None.
            crossentropy_ignore_index (int): Ignore index for cross-entropy loss. Defaults to -100.

        """
        super().__init__()
        # Validate weights_crossentropy
        if weights_crossentropy is None:
            weights_crossentropy = torch.ones(num_labels, dtype=torch.float32)
        else:
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
        if input_ids.shape[1] > 4096:
            logger.warning("Longformer model cannot handle more than 4096 tokens as input. Trunctating sequences to max 4096.")
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
