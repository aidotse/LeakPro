from torch.utils.data import DataLoader, Dataset
import torch
from transformers import LongformerTokenizerFast
import numpy as np
import os
from typing import Any, BinaryIO, List, Optional, Tuple, Union
import pathlib



def save_text_old(original_dataloader: DataLoader, recreated_dataloader: DataLoader) -> None:
    """Save the text from the original and recreated dataloaders to text files.

    Args:
    ----
        original_dataloader (torch.utils.data.DataLoader): Dataloader containing original text.
        recreated_dataloader (torch.utils.data.DataLoader): Dataloader containing recreated text.

    Returns:
    -------
        None

    """
    

    for X in original_dataloader:
       
        
        x = X['embedding'][0]
        xx = torch.argmax(x, dim=1)
        print("xx: ", xx.shape)

    x_ = xx.tolist()
    
    bert = "allenai/longformer-base-4096"
   
    tokenizer2 = LongformerTokenizerFast.from_pretrained(bert)
    
    text = tokenizer2.decode(x_, skip_special_tokens=True)
    print("-- original text --")
    print
    print(text)
    print("text printed!")

    for X in recreated_dataloader:
       
        
        x = X['embedding'][0]
        xx = torch.argmax(x, dim=1)
        print("xx: ", xx.shape)

    x_ = xx.tolist()
    
    
    text = tokenizer2.decode(x_, skip_special_tokens=True)
    print("-- recreated text --")
    print
    print(text)
    print("text printed!")


def validate_tokens(original_dataloader: DataLoader, recreated_dataloader: DataLoader, name: str = "examples") -> None:


    for X, X_ in zip(original_dataloader, recreated_dataloader):

        x = X['embedding'][0].cpu().numpy()
        y = X['labels'][0].cpu().numpy()

        x_ = X_['embedding'][0].detach().cpu().numpy()
        y_ = X_['labels'][0].cpu().numpy()
        ind = np.where(np.array(y)!=0)[0]
    examples = [x[ind],x_[ind]]

    np.save(name, examples)
    return 0.0

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

