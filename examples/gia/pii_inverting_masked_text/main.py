"""Inverting on a single image."""

#from model import ResNet
from torchvision.models.resnet import BasicBlock

from data.pii_data import get_pii_dataset
from leakpro.attacks.gia_attacks.invertinggradients_text import InvertingGradients, InvertingConfig
from leakpro.run import run_gia_attack
from leakpro.fl_utils.data_utils import GiaNERExtension
#from leakpro.fl_utils.gia_train import train
from train import train
from transformers import LongformerTokenizerFast
from longformer_model import Model, OneHotBERT
from leakpro.utils.seed import seed_everything
import torch
from torch import cuda, device
from data.data import pre_process_data, Dataset, LabelSet, TrainingBatch
from dataclasses import dataclass, field
from leakpro.fl_utils.gia_optimizers import MetaAdam, MetaSGD
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # If using GPU
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True  # Makes CuDNN deterministic
    torch.backends.cudnn.benchmark = False  # Slows down but ensures consistency


if __name__ == "__main__":
    #seed_everything(1234)
 
    set_seed(42)

    # define model
    bert = "allenai/longformer-base-4096"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    label_set = LabelSet(labels=["MASK"])
    model = OneHotBERT(model = bert, num_labels = len(label_set.ids_to_label.values()))
    model = model.to(device)
    tokenizer = LongformerTokenizerFast.from_pretrained(bert)

    # attack setup
    configs = InvertingConfig()
    configs.top10norms = False
    configs.median_pooling = False
    configs.at_iterations = 10000
    configs.tv_reg = 1.0e-06
    configs.attack_lr = 0.001#0.00001
    configs.data_extension = GiaNERExtension()
    configs.image_data = False
    

    # define dataloader
    client_dataset, data_mean, data_std = get_pii_dataset(num_docs=1)
    client_dataloader = torch.utils.data.DataLoader(client_dataset, collate_fn=TrainingBatch, batch_size=1, shuffle=False)

    

    # meta train function designed to work with GIA
   
    # baseline config
    #configs = InvertingConfig(attack_lr=0.0001)#, optimizer = field(default_factory=lambda: MetaSGD()))
    #configs = InvertingConfig(attack_lr=0.00001, data_extension = GiaNERExtension, image_data=False)#, optimizer = field(default_factory=lambda: MetaSGD()))
    attack_object = InvertingGradients(model, client_dataloader, data_mean, data_std, configs=configs, train_fn=train)
    run_gia_attack(attack_object, "TestInverting")



