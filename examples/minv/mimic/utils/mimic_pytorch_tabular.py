import pandas as pd
import torch
import pytorch_tabular as pt
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
from sklearn.model_selection import train_test_split

def create_trained_model_and_metadata(train_data, data_config):
    
        
    # Load some tabular data (Example)
    df = train_data

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # PyTorch Tabular Configurations

    model_config = CategoryEmbeddingModelConfig(
        task="classification",
        layers=[64, 32],
        activation="ReLU",
    )

    trainer_config = TrainerConfig(
        max_epochs=50,
        batch_size=16,
    )

    optimizer_config = OptimizerConfig()

    # Train PyTorch-Tabular Model
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        trainer_config=trainer_config,
        optimizer_config=optimizer_config,
    )

    tabular_model.fit(train=train_df, validation=test_df)

    return tabular_model.model
