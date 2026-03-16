"""Code to save and validate text data."""
import pathlib
from typing import BinaryIO, List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import LongformerTokenizerFast


def validate_tokens(original_dataloader: DataLoader, recreated_dataloader: DataLoader, name: str = "examples") -> None:
    """Validate tokens."""
    orig = None
    recr = None
    for org, rec in zip(original_dataloader, recreated_dataloader):

        x = org["embedding"][0].cpu().numpy()
        y = org["labels"][0].cpu().numpy()

        x_ = rec["embedding"][0].detach().cpu().numpy()
        ind = np.where(np.array(y)!=0)[0]

        if orig is None:
            orig = x[ind]
            recr = x_[ind]
        else:
            orig = np.concatenate((orig, x[ind]), axis=0)
            recr = np.concatenate((recr, x_[ind]), axis=0)
    examples = [orig,recr]
    np.save(name, examples)

def save_text(tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[str, pathlib.Path, BinaryIO],

) -> None:
     """Save a given Tensor into a text file.

     Args:
        tensor (Tensor or list): Textloader to be saved.
        fp (string or file object): A filename or a file object.

     """
     bert = "allenai/longformer-base-4096"
     tokenizer = LongformerTokenizerFast.from_pretrained(bert)
     with open(fp, "w", encoding="utf-8") as f:
        for batch in tensor:
            input_ids = batch["embedding"].argmax(dim=-1).tolist()  # shape: (batch_size, seq_len)
            texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            for text in texts:
                f.write(text.strip() + "\n")  # Skriv varje text på en egen rad

