import os
import sys
import yaml
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


# Path to the dataset zip file
data_folder = "./data"

project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.append(project_root)

# Load the config.yaml file
with open('train_config.yaml', 'r') as file:
    train_config = yaml.safe_load(file)

# Generate the dataset and dataloaders
path = os.path.join(os.getcwd(), train_config["data"]["data_dir"])
data_dir =  train_config["data"]["data_dir"] + "/private_df.pkl"

df = pd.read_pickle(data_dir)

df_train = df.sample(frac=train_config["data"]["f_train"], random_state=123)
df_val = df.drop(df_train.index)
# For all entries in df_val, if the identity is not in df_train, remove it
df_val = df_val[df_val["identity"].isin(df_train["identity"])]
df_val = df_val.reset_index(drop=True)

train = False
if train:
    #train_loader, test_loader = get_celebA_train_testloader(train_config, random_state=123)

    df = pd.read_pickle(data_dir)

    df_train = df.sample(frac=train_config["data"]["f_train"], random_state=123)
    df_val = df.drop(df_train.index)
    # For all entries in df_val, if the identity is not in df_train, remove it
    df_val = df_val[df_val["identity"].isin(df_train["identity"])]
    df_val = df_val.reset_index(drop=True)

    # Continous column names
    continuous_col_names = ["lefteye_x", "lefteye_y", "righteye_x", "righteye_y","nose_x", "nose_y", "leftmouth_x", "leftmouth_y", "rightmouth_x", "rightmouth_y"]
    # Categorical column names, the rest are categorical
    categorical_col_names = [col for col in df.columns if col not in continuous_col_names]
    # Remove the target column
    categorical_col_names.remove("identity")

    data_config = DataConfig(
        target=['identity'],
        continuous_cols=continuous_col_names,
        categorical_cols=categorical_col_names,
        #continuous_feature_transform="quantile_normal",
        #normalize_continuous_features=True,
    )

    trainer_config = TrainerConfig(
        auto_lr_find=True,
        batch_size=train_config["train"]["batch_size"],
        max_epochs=100,
        early_stopping='train_loss_0'
    )

    optimizer_config = OptimizerConfig()

    model_config = CategoryEmbeddingModelConfig(
        task="classification",
        layers="1024-512-512",
        activation="ReLU",
        learning_rate=1e-3
    )

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config
    )

    tabular_model.fit(train=df_train, validation=df_val)
    results = tabular_model.evaluate(df_val)
    pred_df = tabular_model.predict(df_val.drop(columns=["identity"]))

    print("validation preds: ", pred_df["identity_prediction"].value_counts())
    # Save the model
    tabular_model.save_model("./target")

from leakpro import LeakPro
from examples.minv.celebA_attributes.celebA_tabular_plgmi_handler import CelebA_InputHandler
config_path = "audit.yaml"


# Initialize the LeakPro object
leakpro = LeakPro(CelebA_InputHandler, config_path)

# Run the audit
results = leakpro.run_audit(return_results=True)