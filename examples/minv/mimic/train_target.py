import os
import sys
import yaml
import warnings
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig, TabNetModelConfig, GANDALFConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
import pandas as pd

# Suppres warnings, pytorch_tabular is very verbose
warnings.filterwarnings("ignore")

# Redefine variables in case upper cell is not run
# Path to the dataset zip file
data_folder = "./data"

project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.append(project_root)

# List of all possible continuous columns in MIMIC
continuous_col_names = ['length_of_stay', 'num_procedures', 'num_medications', 'BMI',
       'BMI (kg/m2)', 'Height', 'Height (Inches)', 'Weight', 'Weight (Lbs)',
       'eGFR', 'systolic', 'diastolic']

audit_file = 'audit_id_0.yaml'
data_file = '/private_df_id_0.pkl'

# Load the config.yaml file
with open('train_config.yaml', 'r') as file:
    train_config = yaml.safe_load(file)

with open(audit_file, 'r') as file:
    audit_config = yaml.safe_load(file)

# Access the first attack in the attack_list
plgmi_attack = next(
    attack for attack in audit_config["audit"]["attack_list"] if attack["attack"] == "plgmi"
)

# Extract num_classes
num_classes = plgmi_attack["num_classes"]

# Generate the dataset and dataloaders
path = os.path.join(os.getcwd(), train_config["data"]["data_dir"])
data_dir =  train_config["data"]["data_dir"] + data_file

df = pd.read_pickle(data_dir)

# Reset index to have a clean, sequential integer index
df = df.reset_index(drop=True)

# Remove the columns from continuous_col_names that are not in the dataframe
continuous_col_names = [col for col in continuous_col_names if col in df.columns]
# Categorical column names are all columns that are not continuous
categorical_col_names = [col for col in df.columns if col not in continuous_col_names]
# Remove the target column
categorical_col_names.remove("identity")

# Ensure df_train contains at least one sample for every class
df_train_min = df.groupby("identity").head(1)  
remaining_df = df.drop(df_train_min.index)

# Determine the fraction for the remaining samples:
desired_frac = train_config["data"]["f_train"]
frac_remaining = desired_frac - (len(df_train_min) / len(df))
df_train_remaining = remaining_df.sample(frac=frac_remaining, random_state=123)

# Merge the guaranteed and random samples.
# Note: we keep the original indices here (do not use ignore_index) so that we can compute df_val correctly.
train_indices = df_train_min.index.union(df_train_remaining.index)
df_train = df.loc[train_indices]

# Create df_val by taking the rest of the samples
df_test = df.drop(train_indices)
df_test = df_test[df_test["identity"].isin(df_train["identity"])]
df_test = df_test.reset_index(drop=True)

# Prints
print("Number of unique classes in df_train: ", df_train["identity"].nunique())
print("Shape of df_train: ", df_train.shape)
print("Shape of df_test: ", df_test.shape)


data_config = DataConfig(
    target=['identity'],
    continuous_cols=continuous_col_names,
    categorical_cols=categorical_col_names,
    normalize_continuous_features=False,
)

trainer_config = TrainerConfig(
    auto_lr_find=False,
    batch_size=256,
    max_epochs=150,
    early_stopping='train_loss_0',
)

optimizer_config = OptimizerConfig()

# model_config = CategoryEmbeddingModelConfig(
#     task="classification",
#     layers="2048-1024-512-256",
#     activation="ReLU",
#     learning_rate=1e-3,
# )

model_config = GANDALFConfig(
task="classification",
gflu_stages=32,
gflu_dropout=0.01,
embedding_dropout=0.1,
learning_rate=1e-3,
)

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config
)

tabular_model.fit(train=df_train) # Defaults 80% train, 20% val split
results = tabular_model.evaluate(df_test)
pred_df = tabular_model.predict(df_test.drop(columns=["identity"]))

# Create target directory if it does not exist
if not os.path.exists("./target/"):
    os.makedirs("./target/")

# Save the model
tabular_model.save_model("./target/")