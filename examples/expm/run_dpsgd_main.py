import os

from torch import zeros
from utils.data_handler import get_mimic_dataloaders, get_mimic_dataset
from utils.dpsgd_model import *

# Import and initialize ReportHandler
from leakpro.reporting.report_handler import ReportHandler

# Generate the dataset and dataloaders
path = os.path.join(os.getcwd(), "examples/expm/data/mimic/")
target_model_dir = "./examples/expm/target_dpsgd"
epsilons = [.0001, .001, .01, .1, .5, 1, 2, 3.5, 5, 7, 10] # epsilons to run over
delta = 1e-5
target_epsilon = 3.5

train_frac = 0.4
valid_frac = 0.0
test_frac = 0.0
early_stop_frac = 0.4
batch_size = 55
use_LR = False # True if you want to use the LR model, False if you want to use the GRUD model

dataset, train_indices, validation_indices, test_indices, early_stop_indices= get_mimic_dataset(path,
                                                                            train_frac ,
                                                                            valid_frac,
                                                                            test_frac,
                                                                            early_stop_frac,
                                                                            use_LR)

train_loader, validation_loader, test_loader, early_stop_loader = get_mimic_dataloaders(dataset,
                                                            train_indices,
                                                            validation_indices,
                                                            test_indices,
                                                            early_stop_indices,
                                                            batch_size)

sample_rate = 1/len(train_loader) # already incorporates batchsize

noise_multiplier_dict = {
    "target_epsilon": target_epsilon,
    "target_delta": delta,
    "sample_rate": sample_rate,
    "epochs": 21,
    "epsilon_tolerance": 0.01,
    "accountant": "prv",
    "eps_error": 0.01,
    "max_grad_norm": 1,
}





optimized_hyperparams ={
    "cell_size": 58,
    "hidden_size": 78,
    "learning_rate": 0.0004738759319792616,
    "num_epochs":50,
    "patience_early_stopping": 20,
    "patience_lr_scheduler": 5,
    "batch_size": 59,
    "seed": 4410,
    "min_delta": 0.00001,
    "epsilon": 3.5,
    "max_grad_norm": 1,
    }
n_features = int(dataset.x.shape[1]/3)
X_mean = zeros(1,dataset.x.shape[2],n_features)

model_params = {k: optimized_hyperparams[k] for k in ["cell_size", "hidden_size", "batch_size"]}

# Add other required parameters to model_params
model_params.update({
    "input_size": n_features,
    "X_mean": X_mean,
    "output_last": False,
    "bn_flag": False,
    # "droupout": 0.33,
})


# Initialize the model with filtered parameters
model = GRUD_DPSGD(**model_params)
# Train the model
results= dpsgd_gru_trained_model_and_metadata(
                                            model,
                                            train_loader,
                                            early_stop_loader,
                                            noise_multiplier_dict,
                                            epochs=optimized_hyperparams["num_epochs"],
                                            patience_early_stopping = optimized_hyperparams["patience_early_stopping"],
                                            patience_lr= optimized_hyperparams["patience_lr_scheduler"],
                                            min_delta = optimized_hyperparams["min_delta"],
                                            learning_rate = optimized_hyperparams["learning_rate"],
                                            target_model_dir = target_model_dir,)
train_losses, test_losses , train_acc, test_acc, best_model,niter_per_epoch, privacy_engine  = results


import matplotlib.pyplot as plt

# Convert losses to numpy-compatible lists directly
train_losses_cpu = [float(loss) for loss in train_losses]
test_losses_cpu = [float(loss) for loss in test_losses]

# Plot training and test accuracy
plt.figure(figsize=(5, 4))

plt.subplot(1, 2, 1)
plt.plot(train_acc, label="Train Accuracy")
plt.plot(test_acc, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()

# Plot training and test loss
plt.subplot(1, 2, 2)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig("psgd_gru.png")



from dpsgd_handler import MimicInputHandlerGRU

from leakpro import LeakPro

# Read the config file
config_path = "./examples/expm/audit.yaml"

# Prepare leakpro object
leakpro = LeakPro(MimicInputHandlerGRU, config_path)

# Run the audit
mia_results = leakpro.run_audit(return_results=True)



# report_handler = ReportHandler()
report_handler = ReportHandler(report_dir="./examples/expm/leakpro_output/results")

# Save MIA resuls using report handler
for res in mia_results:
    report_handler.save_results(attack_name=res.attack_name, result_data=res, config=res.configs)

