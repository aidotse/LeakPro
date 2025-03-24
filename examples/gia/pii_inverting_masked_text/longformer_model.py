from typing_extensions import TypedDict
import torch.nn.functional as F
from typing import List,Any
from transformers import LongformerModel
from tokenizers import Encoding
from torch import nn
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
from torch.utils.data.dataloader import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn


class Model(nn.Module):
    """
    Full fine=tuning of all Longofrmer's parameters, with a linear classification layer on top.
    """
    def __init__(self, model, num_labels):
        super().__init__()
        self._bert = LongformerModel.from_pretrained(model)

        for param in self._bert.parameters():
           param.requires_grad = True

        self.classifier = nn.Linear(768, num_labels)
        
    def forward(self, batch):
        b = self._bert(
            input_ids=batch["input_ids"], attention_mask=batch["attention_masks"]
        )
        pooler = b.last_hidden_state
        return self.classifier(pooler)

class Model2(nn.Module):
    """
    Full fine=tuning of all Longofrmer's parameters, with a linear classification layer on top.
    """
    def __init__(self, model, num_labels):
        super().__init__()
        self._bert = LongformerModel.from_pretrained(model)

        for param in self._bert.parameters():
           param.requires_grad = True

        self.classifier = nn.Linear(768, num_labels)
        
    def forward(self, batch):

        #embed = self._bert.embeddings.word_embeddings(batch["input_ids"]).detach()
        b = self._bert(
            inputs_embeds=batch["embedding"], attention_mask=batch["attention_masks"]
        )
        pooler = b.last_hidden_state
        return self.classifier(pooler)


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class OneHotBERT(nn.Module):
    def __init__(self, model, num_labels, token_mask=None, norm_data=False):
        super().__init__()
        self.bert = LongformerModel.from_pretrained(model) # Load pre-trained BERT
        self.vocab_size = self.bert.config.vocab_size
        self.hidden_size = self.bert.config.hidden_size

        # Replace the embedding layer with a Linear layer
        self.embedding_layer = nn.Linear(self.vocab_size, self.hidden_size, bias=False)
        self.token_mask = torch.ones(self.vocab_size)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.token_mask = self.token_mask.to(device)
        self.norm_data = norm_data
        for param in self.bert.parameters():
           param.requires_grad = True
       
        #if token_mask is not None:
        #    

            # Create a mask (1 for known tokens, 0 for others)
        #    self.token_mask = torch.zeros(self.vocab_size)
        #    self.token_mask[token_mask] = 1
            #token_mask = token_mask.to(device)
        #else:
        #    self.token_mask = torch.ones(self.vocab_size)
    
        # Initialize with BERT’s embedding matrix
        with torch.no_grad():
            self.embedding_layer.weight.copy_(self.bert.embeddings.word_embeddings.weight.T)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, batch):
        """
        one_hot_input: Tensor of shape (batch_size, seq_len, vocab_size)
        """
   
        #print("batch[embedding] shape: ", batch["embedding"].shape)
        # Compute embeddings using matrix multiplication
        #embeddings = self.embedding_layer(batch["embedding"])  # (batch_size, seq_len, hidden_size)
        #softmax_input = batch["embedding"]*torch.sum(batch["embedding"], dim=-1)
        #norm_factor = torch.sum(batch["embedding"],dim=-1)[:,:, None]
        #print("norm_factor shape: ", norm_factor.shape)
        #normalized = batch["embedding"]/norm_factor
        #print("normalized shape: ", normalized.shape)
        #embeddings = self.embedding_layer(normalized * self.token_mask) # (batch_size, seq_len, hidden_size)

        #embeddings = self.embedding_layer(batch["embedding"]**2 * self.token_mask) # (batch_size, seq_len, hidden_size)

        #embeddings = self.embedding_layer(torch.sigmoid(batch["embedding"])*2-1 * self.token_mask) # (batch_size, seq_len, hidden_size)
        #print("using self.norm_data: ", self.norm_data)
        if self.norm_data:
            
            #print("torch.sum(batch[embedding], dim=-1): ", torch.sum(batch["embedding"], dim=-1))
            print("batch[embedding] shape: ", batch["embedding"].shape)
            normalized = batch["embedding"]/torch.sum(batch["embedding"], dim=-1)[:,:, None]
        else:
           
            normalized = batch["embedding"]
        #print("normalized sum shape: ", torch.sum(normalized, dim=-1).shape)
        #print("normalized sum:", torch.sum(normalized, dim=-1))

        embeddings = self.embedding_layer(normalized * self.token_mask)
        
        b = self.bert(inputs_embeds=embeddings, attention_mask=batch["attention_masks"])
        pooler = b.last_hidden_state
        return self.classifier(pooler)

