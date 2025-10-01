import os
import yaml
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100
from torch import cat, tensor
import pickle
from torch import save, optim, nn
from target_model_class import WideResNet
from leakpro import LeakPro
from cifar_handler import CifarInputHandler
import pickle
import matplotlib.pyplot as plt
import csv

def prepare_cifar_data(dataset_name):

    root = train_config["data"]["data_dir"]
    path = os.path.join(os.getcwd(), root)
    # Load the CIFAR train and test datasets
    if dataset_name == "cifar10":
        trainset = CIFAR10(root=root, train=True, download=True)
        testset = CIFAR10(root=root, train=False, download=True)
    elif dataset_name == "cifar100":
        trainset = CIFAR100(root=root, train=True, download=True)
        testset = CIFAR100(root=root, train=False, download=True)
    else:
        raise ValueError("Unknown dataset type")

    train_data = tensor(trainset.data).permute(0, 3, 1, 2).float() / 255  # (N, C, H, W)
    test_data = tensor(testset.data).permute(0, 3, 1, 2).float() / 255

    # Ensure train and test data looks correct
    assert train_data.shape[0] == 50000, "Train data should have 50000 samples"
    assert test_data.shape[0] == 10000, "Test data should have 10000 samples"
    assert train_data.shape[1] == 3, "Data should have 3 channels"
    assert test_data.shape[1] == 3, "Data should have 3 channels"
    assert train_data.max() <= 1 and train_data.min() >= 0, "Data should be normalized"
    assert test_data.max() <= 1 and test_data.min() >= 0, "Data should be normalized"

    # Concatenate train and test data into the population
    data = cat([train_data.clone().detach(), test_data.clone().detach()], dim=0)
    targets = cat([tensor(trainset.targets), tensor(testset.targets)], dim=0)
    # Create UserDataset object
    population_dataset = CifarInputHandler.UserDataset(data, targets)

    assert len(population_dataset) == 60000, "Population dataset should have 60000 samples"

    # Store the population dataset to be used by LeakPro
    file_path =  "data/"+ dataset_name + ".pkl"
    if not os.path.exists(file_path):
        with open(file_path, "wb") as file:
            pickle.dump(population_dataset, file)
            print(f"Save data to {file_path}")
    return data, targets
            
def get_datasets(train_config, data, targets, seed=0):
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader
    import numpy as np

    train_fraction = train_config["data"]["f_train"]
    test_fraction = train_config["data"]["f_test"]
    batch_size = train_config["train"]["batch_size"]

    dataset_size = 60000
    train_size = int(train_fraction * dataset_size)
    test_size = int(test_fraction * dataset_size)

    np.random.seed(seed)
    selected_index = np.random.choice(np.arange(dataset_size), train_size + test_size, replace=False)
    train_indices, test_indices = train_test_split(selected_index, test_size=test_size)

    train_subset = CifarInputHandler.UserDataset(data[train_indices], targets[train_indices])
    test_subset = CifarInputHandler.UserDataset(data[test_indices], targets[test_indices], **train_subset.return_params())

    train_loader = DataLoader(train_subset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_subset, batch_size = batch_size, shuffle = False)

    # Evaluate mean and variance of the train data
    train_mean = train_subset.mean
    train_std = train_subset.std
    print (f"Train mean: {train_mean}, Train std: {train_std}")
    
    return train_loader, test_loader, train_indices, test_indices

def train_target_model(train_config, train_loader, test_loader, train_indices, test_indices, dataset_name):

    # Train the model
    if not os.path.exists("target"):
        os.makedirs("target")
    if train_config["data"]["dataset"] == "cifar10":
        num_classes = 10
    elif train_config["data"]["dataset"] == "cifar100":
        num_classes = 100
    else:
        raise ValueError("Invalid dataset name")

    # Create instance of target model
    #model = ResNet18(num_classes = num_classes)
    model =  WideResNet(depth=28, num_classes=num_classes, widen_factor=2, dropRate=0.3)

    # Read out the relevant parameters for training
    lr = train_config["train"]["learning_rate"]
    weight_decay = train_config["train"]["weight_decay"]
    momentum = train_config["train"]["momentum"]
    nesterov = train_config["train"]["nesterov"]
    epochs = train_config["train"]["epochs"]
        
    # Create optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)

    # train target model
    train_result = CifarInputHandler().train(dataloader=train_loader,
                                model=model,
                                criterion=criterion,
                                optimizer=optimizer,
                                epochs=epochs)

    # Evaluate on test set
    test_result = CifarInputHandler().eval(test_loader, model, criterion)

    # Store the model and metadata
    model = train_result.model
    model.to("cpu")
    with open(train_config["run"]["log_dir"]+"/target_model.pkl", "wb") as f:
        save(model.state_dict(), f)

    # Create metadata to be used by LeakPro
    from leakpro import LeakPro
    meta_data = LeakPro.make_mia_metadata(train_result = train_result,
                                        optimizer = optimizer,
                                        loss_fn = criterion,
                                        dataloader = train_loader,
                                        test_result = test_result,
                                        epochs = epochs,
                                        train_indices = train_indices,
                                        test_indices = test_indices,
                                        dataset_name = dataset_name)

    with open("target/model_metadata.pkl", "wb") as f:
        pickle.dump(meta_data, f)
    
    return meta_data

# define the main function
if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    n_train_models = 1
    
    # Load the config.yaml file
    script_dir = os.path.dirname(__file__)  # path to neurips.py
    config_path = 'train_config.yaml'
    audit_path = "audit.yaml"
    
    with open(config_path, 'r') as file:
        train_config = yaml.safe_load(file)
    dataset_name = train_config["data"]["dataset"]
    # Prepare the data loaders
    data, targets = prepare_cifar_data(dataset_name)
    mia_results = {}
    mia_results["target"] = {"train_acc": [], "test_acc": []}
    for i in range(n_train_models):
        # print(f"Running MIA for run {i}")
        # train_loader, test_loader, train_indices, test_indices = get_datasets(train_config, data, targets, seed=i)
        # meta_data = train_target_model(train_config, train_loader, test_loader, train_indices, test_indices, dataset_name)
        # print(f"Target model trained and metadata saved for run {i}")
        # print(f"Train accuracy: {meta_data.train_result.accuracy}")
        # print(f"Test accuracy: {meta_data.test_result.accuracy}")
        # mia_results["target"]["train_acc"].append(meta_data.train_result.accuracy)
        # mia_results["target"]["test_acc"].append(meta_data.test_result.accuracy)

        # Create a new instance of LeakPro
        leakpro = LeakPro(CifarInputHandler, audit_path)

        # Run the audit 
        result = leakpro.run_audit(create_pdf=False, use_optuna=False)
        
        for res in result:
            if res.id not in mia_results:
                mia_results[res.id] = {"roc_auc": [], "tpr": [], "fpr": [], "thresholds": []}
            mia_results[res.id]["roc_auc"].append(res.roc_auc)
            mia_results[res.id]["tpr"].append(res.tpr)
            mia_results[res.id]["fpr"].append(res.fpr)
            mia_results[res.id]["thresholds"].append(res.thresholds)
        
        # Remove the cached logits in leakpro_output/attack_cache to store new ones matching the new train indices
        for file in os.listdir("leakpro_output/attack_cache"):
            if file.endswith(".npy"):
                os.remove(os.path.join("leakpro_output/attack_cache", file))
                print(f"Removed {file} from leakpro_output/attack_cache")
        
mia_averages = {}

# Define a common FPR grid (1000 points from 0 to 1)
mean_fpr = np.logspace(-4, 0, 1000)  # from 1e-4 to 1 (log scale), 100 points

for res_id, metrics in mia_results.items():
    if res_id == "target":
        mia_averages[res_id] = {
            "train_acc": np.mean(mia_results["target"]["train_acc"]),
            "test_acc": np.mean(mia_results["target"]["test_acc"]),
            "std_train_acc": np.std(mia_results["target"]["train_acc"]),
            "std_test_acc": np.std(mia_results["target"]["test_acc"]),
        }
        continue

    # Average scalar metric
    roc_auc_avg = float(np.mean(metrics["roc_auc"]))
    roc_auc_avg_std = float(np.std(metrics["roc_auc"]))

    # Interpolate TPRs onto mean_fpr grid
    interpolated_tprs = []
    for tpr, fpr in zip(metrics["tpr"], metrics["fpr"]):
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0  # Ensure starts at (0,0)
        interpolated_tprs.append(interp_tpr)
        
    #-----------
    # Compute mean and std of thresholds at specific FPR targets
    fpr_targets = [0.01, 0.001]
    threshold_at_fpr_targets = {}
    for target in fpr_targets:
        thresholds_at_fpr = []
        for fpr, thresholds in zip(metrics["fpr"], metrics["thresholds"]):
            mask = fpr <= target
            if np.any(mask):
                thresholds_at_fpr.append(thresholds[mask].min())
            else:
                thresholds_at_fpr.append(np.nan) 
        threshold_at_fpr_targets[f"thres_at_fpr_{target}"] = np.mean(thresholds_at_fpr)
        threshold_at_fpr_targets[f"thres_at_fpr_{target}_std"] = np.std(thresholds_at_fpr)
    
    #-----------
    metrics_at_threshold_target = {}
    threshold_targets = [1 - x for x in fpr_targets]  # if this is your intent

    for threshold_target in threshold_targets:
        tmp_fpr = []
        tmp_tpr = []
        for fpr, tpr, thresholds in zip(metrics["fpr"], metrics["tpr"], metrics["thresholds"]):
            thresholds = np.array(thresholds)
            idx = (np.abs(thresholds - threshold_target)).argmin()
            tmp_fpr.append(fpr[idx])
            tmp_tpr.append(tpr[idx])
        
        label = f"{threshold_target:.3f}".rstrip("0").rstrip(".")  # clean label
        metrics_at_threshold_target[f"fpr_at_threshold_{label}"] = np.mean(tmp_fpr)
        metrics_at_threshold_target[f"fpr_at_threshold_{label}_std"] = np.std(tmp_fpr)
        metrics_at_threshold_target[f"tpr_at_threshold_{label}"] = np.mean(tmp_tpr)
        metrics_at_threshold_target[f"tpr_at_threshold_{label}_std"] = np.std(tmp_tpr)
            
        
    tpr_array = np.stack(interpolated_tprs, axis=0)  # (n_runs, 100)
    tpr_avg = np.mean(tpr_array, axis=0)
    tpr_std = np.std(tpr_array, axis=0)

    mia_averages[res_id] = {
        "roc_auc": roc_auc_avg,
        "roc_auc_std": roc_auc_avg_std,
        "fpr": mean_fpr,
        "tpr": tpr_avg,
        "tpr_std": tpr_std,
        "metrics_at_threshold_target": metrics_at_threshold_target,
        "threshold_at_fpr_targets": threshold_at_fpr_targets,
    }

# Save the results
with open("mia_results.pkl", "wb") as f:
    pickle.dump(mia_averages, f)
    print(f"Saved results to mia_results.pkl")
        

import textwrap

def wrap_label(label, width=30):
    return "\n".join(textwrap.wrap(label, width=width))

fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")  # mpl>=3.6; else use constrained_layout=True
title = f"Average ROC Curve for {dataset_name} (n_train_models={n_train_models})"

# random-chance line
ax.plot([1e-4, 1.0], [1e-4, 1.0], linestyle="--", color="gray", linewidth=1)

for res_id, metrics in mia_averages.items():
    if res_id == "target":
        continue
    fpr = metrics["fpr"]
    tpr = metrics["tpr"]
    tpr_std = metrics.get("tpr_std")
    auc = metrics["roc_auc"]
    std = metrics["roc_auc_std"]

    label = f"{wrap_label(res_id)}\n(AUC = {auc:.3f} ± {std:.3f})"
    ax.plot(fpr, tpr, label=label)
    if tpr_std is not None:
        tpr_upper = np.minimum(tpr + tpr_std, 1)
        tpr_lower = np.maximum(tpr - tpr_std, 0)
        ax.fill_between(fpr, tpr_lower, tpr_upper, alpha=0.2)

# axes cosmetics
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e-4, 1.0)
ax.set_ylim(0.0, 1.05)
ax.set_xlabel("False Positive Rate (log scale)")
ax.set_ylabel("True Positive Rate")
ax.set_title(title)
ax.grid(True, which="both", linestyle="--", linewidth=0.5)

# FIGURE-LEVEL legend (so layout engine allocates space)
# Get handles and labels from the axis
handles, labels = ax.get_legend_handles_labels()

# Put legend below plot
fig.legend(
    handles, labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.05),   # (x, y) relative to figure
    ncol=min(4, len(labels)),
    fontsize="x-small",
    frameon=False,
    handlelength=1.3,
    handletextpad=0.4,
    columnspacing=0.8
)

# Adjust layout so space is reserved
fig.subplots_adjust(bottom=0.8)  # increase if legend still overlaps


plt.show()

save_path = "average_roc_curve.png"
if save_path:
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
print(f"ROC curve saved to: {save_path}")

plt.show()

import os
import csv

os.makedirs("mia_csv_per_attack", exist_ok=True)

for res_id, metrics in mia_averages.items():
    if res_id == "target":
        #store target model results
        filename = f"mia_csv_per_attack/{res_id}.csv"
        with open(filename, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["train_acc", "test_acc", "std_train_acc", "std_test_acc"])
            writer.writerow([metrics["train_acc"], metrics["test_acc"], metrics["std_train_acc"], metrics["std_test_acc"]])
        continue
    fpr = metrics["fpr"]
    tpr = metrics["tpr"]
    tpr_std = metrics.get("tpr_std", None)

    filename = f"mia_csv_per_attack/{res_id}.csv"
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["fpr", "tpr", "tpr_std"])
        for f_val, t_val, s_val in zip(fpr, tpr, tpr_std):
            writer.writerow([f_val, t_val, s_val])