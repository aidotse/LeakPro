import os

from opacus.accountants.utils import get_noise_multiplier
from torch import zeros
from utils.gru_model_handler import *

from examples.expm.utils.data_preparation import get_mimic_dataloaders, get_mimic_dataset

# Generate the dataset and dataloaders
path = os.path.join(os.getcwd(), "examples/expm/data/mimic/")
epsilons = [.0001, .001, .01, .1, .5, 1, 2, 3.5, 5, 7, 10] # epsilons to run over
delta = 1e-5


train_frac = 0.4
valid_frac = 0.0
test_frac = 0.0
early_stop_frac = 0.4
batch_size = 59
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

try:
    noise_multiplier = get_noise_multiplier(target_epsilon = 2,
                                        target_delta = delta,
                                        sample_rate = sample_rate,
                                        epochs = 21,
                                        epsilon_tolerance = 0.01,
                                        accountant = "prv",
                                        eps_error = 0.01)
except:
    # the prv accountant is not robust to large epsilon (even epsilon = 10)
    # so we will use rdp when it fails, so the actual epsilon may be slightly off
    # see https://github.com/pytorch/opacus/issues/604
    noise_multiplier = get_noise_multiplier(target_epsilon = 2,
                                            target_delta = delta,
                                            sample_rate = sample_rate,
                                            epochs = 21,
                                            epsilon_tolerance = 0.01,
                                            accountant = "rdp")


# Initialize model with the best hyperparameters
model_params = {
    "X_mean": X_mean,
    "input_size": X_mean.shape[2],
    "device_id": device,
    "cell_size": h["cell_size"],
    "hidden_size": h["hidden_size"],
    "batch_size": h["batch_size"],
    "apply_sigmoid": h["loss"] == "l2",  # Only add apply_sigmoid for 'l2' loss
    "use_bn": h["use_bn"] != "nobn",  # Use batch norm unless 'nobn' specified
}


optimized_hyperparams ={
    "cell_size": 58,
    "hidden_size": 78,
    "learning_rate": 0.0004738759319792616,
    "num_epochs":37,
    "patience_early_stopping": 20,
    "patience_lr_scheduler": 5,
    "batch_size": 59,
    "seed": 4410,
    "min_delta": 0.00001,
    "epsilon": 10,
    "max_grad_norm": 1,
    }
n_features = int(dataset.x.shape[1]/3)
X_mean = zeros(1,dataset.x.shape[2],n_features)

model_params = {k: optimized_hyperparams[k] for k in ["cell_size", "hidden_size", "batch_size"]}

# Add other required parameters to model_params
model_params.update({
    "input_size": n_features,
    "X_mean": X_mean,
    "output_last": False
})


# if h["loss"] == "l2":
#     loss_fn = lambda preds, targets: torch.nn.MSELoss()(torch.squeeze(preds), targets)
# elif h["loss"] == "bce":
#     loss_fn = torch.nn.BCEWithLogitsLoss()



# Initialize the model with filtered parameters
model = GRUD(**model_params)
# Train the model
results= dpsgd_gru_trained_model_and_metadata(
                                            model,
                                            train_loader,
                                            early_stop_loader,
                                            noise_multiplier,
                                            max_grad_norm = optimized_hyperparams["max_grad_norm"],
                                            epochs=optimized_hyperparams["num_epochs"],
                                            patience_early_stopping = optimized_hyperparams["patience_early_stopping"],
                                            patience_lr= optimized_hyperparams["patience_lr_scheduler"],
                                            min_delta = optimized_hyperparams["min_delta"],
                                            learning_rate = optimized_hyperparams["learning_rate"])
train_losses, test_losses , train_acc, test_acc, best_model,niter_per_epoch, privacy_engine  = results

optimized_hyperparams ={
    "cell_size": 58,
    "hidden_size": 78,
    "learning_rate": 0.0004738759319792616,
    "num_epochs":37,
    "patience_early_stopping": 20,
    "patience_lr_scheduler": 5,
    "batch_size": 59,
    "seed": 4410,
    "min_delta": 0.00001,
    "epsilon": 10,
    }

