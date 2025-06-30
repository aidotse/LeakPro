"""Module with functions for preparing the dataset for training the target models."""
import torchvision
from torch import Tensor, as_tensor, randperm
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from transformers import LongformerTokenizerFast

import sys
import os

# Add the path where your module is located
sys.path.append(os.path.abspath("examples/gia/pii_inverting_1_image"))


from data_.data_manipulation import training_raw
from data_.data import pre_process_data, Dataset, LabelSet, TrainingBatch


from leakpro.fl_utils.data_utils import get_meanstd


def get_pii_dataset(num_docs:int =1, example_id:int = 44) -> tuple[DataLoader, Tensor, Tensor]:
    """Get the full dataset for CIFAR10."""
    
    bert = "allenai/longformer-base-4096"    
    tokenizer = LongformerTokenizerFast.from_pretrained(bert)
    label_set = LabelSet(labels=["MASK"])
    training_examples = pre_process_data(data=training_raw[example_id:example_id+num_docs], tokenizer=tokenizer,bert=bert, label_set=label_set, tokens_per_batch=4096)
    train_dataset = Dataset(training_examples)
    data_mean, data_std = 0, 1
    

    
    return train_dataset, data_mean, data_std
