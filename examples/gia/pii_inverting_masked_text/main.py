"""Inverting on a single document."""
from data_.pii_data import get_pii_dataset
from data_.data_extension import GiaNERExtension
from leakpro.attacks.gia_attacks.invertinggradients_text import InvertingGradients, InvertingConfig
from leakpro.schemas import OptunaConfig
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
    configs.at_iterations = 1000
    configs.tv_reg = 1.0e-06
    configs.attack_lr = 0.001
    configs.data_extension = GiaNERExtension()
    

    # define dataloader
    client_dataset, data_mean, data_std = get_pii_dataset(num_docs=1)
    client_dataloader = torch.utils.data.DataLoader(client_dataset, collate_fn=TrainingBatch, batch_size=1, shuffle=False)

    trial_data = []
    for i in range(5):
        trial_ds, _, _ = get_pii_dataset(num_docs=1, example_id=44+i)
        trial_data.append(trial_ds)

    # baseline config
    attack_object = InvertingGradients(model, client_dataloader, data_mean, data_std, configs=configs, train_fn=train, optuna_trial_data=trial_data)
    optuna_config = OptunaConfig()
    optuna_config.n_trials = 100
    optuna_config.check_interval = 100
    attack_object.run_with_optuna(optuna_config=optuna_config)




