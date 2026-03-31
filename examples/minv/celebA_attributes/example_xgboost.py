import os
import sys
import yaml
from sklearn.preprocessing import LabelEncoder
import pickle

# Path to the dataset zip file
data_folder = "./data"


project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.append(project_root)

from examples.minv.celebA_attributes.utils.celebA_tabular_data import get_celebA_train_testloader, get_celebA_publicloader

# Load the config.yaml file
with open('train_config.yaml', 'r') as file:
    train_config = yaml.safe_load(file)

# Generate the dataset and dataloaders
path = os.path.join(os.getcwd(), train_config["data"]["data_dir"])

#print(train_config)

train_loader, test_loader = get_celebA_train_testloader(train_config, random_state=123)

public_loader = get_celebA_publicloader(train_config)


from examples.minv.celebA_attributes.utils.celebA_tabular_model_xgboost import train_xgboost_model
le = LabelEncoder()
le.fit(train_loader.dataset.labels)
train_loader.dataset.labels = le.transform(train_loader.dataset.labels)

# Create the model and metadata
#train_acc, test_acc, train_loss = train_xgboost_model(train_loader.dataset.features, train_loader.dataset.labels, test_loader.dataset.features, test_loader.dataset.labels, log_dir=train_config["run"]["log_dir"])

#print(f"Training Accuracy: {train_acc:.4f}, Training Loss (mlogloss): {train_loss:.4f}")


from leakpro import LeakPro
from examples.minv.celebA_attributes.celebA_tabular_plgmi_handler import CelebA_InputHandler
config_path = "audit.yaml"


# Initialize the LeakPro object
leakpro = LeakPro(CelebA_InputHandler, config_path)

# Run the audit
results = leakpro.run_audit(return_results=True)