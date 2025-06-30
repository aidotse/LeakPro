"""Inverting on a single document."""
from data_.pii_data import get_pii_dataset
from data_.data_extension import GiaNERExtension
from leakpro.attacks.gia_attacks.invertinggradients_text import InvertingGradients, InvertingConfig
from leakpro.run import run_gia_attack
from leakpro.utils.seed import seed_everything
from train import train
from transformers import LongformerTokenizerFast
from longformer_model import OneHotBERT
import torch
from torch import device
from data_.data import LabelSet, TrainingBatch

if __name__ == "__main__":
   
 
    seed_everything(42)

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
    configs.attack_lr = 0.001
    configs.data_extension = GiaNERExtension()
    configs.image_data = False
    

    # define dataloader
    client_dataset, data_mean, data_std = get_pii_dataset(num_docs=1)
    client_dataloader = torch.utils.data.DataLoader(client_dataset, collate_fn=TrainingBatch, batch_size=1, shuffle=False)

    # baseline config
    attack_object = InvertingGradients(model, client_dataloader, data_mean, data_std, configs=configs, train_fn=train)
    run_gia_attack(attack_object, "TestInverting")



