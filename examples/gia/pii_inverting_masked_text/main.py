"""Inverting on a single image."""

#from model import ResNet
from torchvision.models.resnet import BasicBlock

from data.pii_data import get_pii_dataset
from leakpro.attacks.gia_attacks.invertinggradients import InvertingConfig
from leakpro.attacks.gia_attacks.huang import HuangConfig
from leakpro.run import run_gia_attack
from leakpro.fl_utils.data_utils import GiaNERExtension
#from leakpro.fl_utils.gia_train import train
from train import train
from transformers import LongformerTokenizerFast
from longformer_model import Model, OneHotBERT
from leakpro.utils.seed import seed_everything
import torch
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


    bert = "allenai/longformer-base-4096"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    label_set = LabelSet(labels=["MASK"])
    model = OneHotBERT(model = bert, num_labels = len(label_set.ids_to_label.values()))
    model = model.to(device)

    client_dataset, data_mean, data_std = get_pii_dataset(num_docs=1)
    client_dataloader = torch.utils.data.DataLoader(client_dataset, collate_fn=TrainingBatch, batch_size=1, shuffle=False)

    tokenizer = LongformerTokenizerFast.from_pretrained(bert)
    

    # meta train function designed to work with GIA
    train_fn = train
    # baseline config
    #configs = InvertingConfig(attack_lr=0.0001)#, optimizer = field(default_factory=lambda: MetaSGD()))
    configs = InvertingConfig(attack_lr=0.00001, data_extension = GiaNERExtension, image_data=False)#, optimizer = field(default_factory=lambda: MetaSGD()))

    result = run_gia_attack(model, client_dataloader, train_fn, data_mean, data_std, configs)




    

"""Inverting on a single image."""

#from model import ResNet
from torchvision.models.resnet import BasicBlock

from data.pii_data import get_pii_dataset
from leakpro.attacks.gia_attacks.invertinggradients import InvertingConfig
from leakpro.attacks.gia_attacks.huang import HuangConfig
from leakpro.run import run_huang
from leakpro.run import run_inverting
from leakpro.fl_utils.data_utils import GiaNERExtension
#from leakpro.fl_utils.gia_train import train
from train import train
from transformers import LongformerTokenizerFast
from longformer_model import Model, OneHotBERT
from leakpro.utils.seed import seed_everything
import torch
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
    #model = ResNet(BasicBlock, [5, 5, 5], num_classes=10, base_width=16 * 10)
    #seed_everything(1234)
 


    set_seed(42)


    bert = "allenai/longformer-base-4096"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    label_set = LabelSet(labels=["MASK"])
    model = OneHotBERT(model = bert, num_labels = len(label_set.ids_to_label.values()))
    model = model.to(device)

    client_dataset, data_mean, data_std = get_pii_dataset(num_docs=1)
    client_dataloader = torch.utils.data.DataLoader(client_dataset, collate_fn=TrainingBatch, batch_size=1, shuffle=False)

    tokenizer = LongformerTokenizerFast.from_pretrained(bert)
    

    # meta train function designed to work with GIA
    train_fn = train
    # baseline config
    #configs = InvertingConfig(attack_lr=0.0001)#, optimizer = field(default_factory=lambda: MetaSGD()))
    configs = InvertingConfig(attack_lr=0.001, data_extension = GiaNERExtension, image_data=False)#, optimizer = field(default_factory=lambda: MetaSGD()))
        #lr = 0.001
    result = run_inverting(model, client_dataloader, train_fn, data_mean, data_std, configs)




    

