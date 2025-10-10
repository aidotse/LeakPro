import os
import yaml
import numpy as np
import pickle
from leakpro import LeakPro
import pickle
import matplotlib.pyplot as plt
import csv

from examples.mia.celebA_HQ.utils.celeb_hq_data import get_celebA_hq_dataloader
from examples.mia.celebA_HQ.utils.celeb_hq_model import ResNet18, create_trained_model_and_metadata
from celebA_HQ_handler import CelebAHQInputHandler
from leakpro import LeakPro

def make_datasets(configs, data, targets, seed=0):
    """Create train and test datasets with given random seed."""
    np.random.seed(seed)
    num_samples = len(data)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    split = int(configs["data"]["train_fraction"] * num_samples)
    train_indices = indices[:split].tolist()
    test_indices = indices[split:].tolist()

    train_data = data[train_indices]
    train_targets = targets[train_indices]
    test_data = data[test_indices]
    test_targets = targets[test_indices]

    return train_data, test_data, train_targets, test_targets, train_indices, test_indices

# define the main function
if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    n_train_models = 1
    
    # Load the config.yaml file
    script_dir = os.path.dirname(__file__)  # path to neurips.py
    config_path = 'train_config.yaml'
    audit_path = "audit.yaml"
    
    # Load the config.yaml file
    with open('train_config.yaml', 'r') as file:
        train_config = yaml.safe_load(file)

    # Generate the dataset and dataloaders
    path = os.path.join(os.getcwd(), train_config["data"]["data_dir"])
        
    num_classes =307
    model = ResNet18(num_classes = num_classes)
    train_loader, test_loader = get_celebA_hq_dataloader(path, train_config)
    

    # Prepare the data loaders
    mia_results = {}
    mia_results["target"] = {"train_acc": [], "test_acc": []}
    for i in range(n_train_models):
        print(f"Running MIA for run {i}")
        train_loader, test_loader, train_indices, test_indices = get_datasets(train_config, data, targets, seed=i)
        train_acc, train_loss, test_acc, test_loss = create_trained_model_and_metadata(model,train_loader,test_loader, train_config)
        
        print(f"Target model trained and metadata saved for run {i}")
        print(f"Train accuracy: {meta_data.train_result.accuracy}")
        print(f"Test accuracy: {meta_data.test_result.accuracy}")
        mia_results["target"]["train_acc"].append(meta_data.train_result.accuracy)
        mia_results["target"]["test_acc"].append(meta_data.test_result.accuracy)

        # Create a new instance of LeakPro
        leakpro = LeakPro(CelebAHQInputHandler, audit_path)

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

plt.figure(figsize=(8, 6))
title = f"Average ROC Curve for {dataset_name} (n_train_models={n_train_models})"
plt.title(title)
# plot the random chance line
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
for res_id, metrics in mia_averages.items():
    if res_id == "target":
        continue
    fpr = metrics["fpr"]
    tpr = metrics["tpr"]
    tpr_std = metrics.get("tpr_std", None)
    auc = metrics["roc_auc"]
    std = metrics["roc_auc_std"]
    # add std
    label = f"{wrap_label(res_id)}\n(AUC = {auc:.3f} ± {std:.3f})"
    plt.plot(fpr, tpr, label=label)

    # Add confidence band if available
    if tpr_std is not None:
        tpr_upper = np.minimum(tpr + tpr_std, 1)
        tpr_lower = np.maximum(tpr - tpr_std, 0)
        plt.fill_between(fpr, tpr_lower, tpr_upper, alpha=0.2)

plt.xscale("log")
plt.yscale("log")
plt.xlim(1e-4, 1.0)
plt.ylim(0.0, 1.05)
plt.xlabel("False Positive Rate (log scale)")
plt.ylabel("True Positive Rate")
plt.title(title)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
           ncol=4, fontsize="x-small", frameon=False)
plt.subplots_adjust(bottom=0.25)
plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave room at the bottom
plt.tight_layout()

save_path = "average_roc_curve.png"
if save_path:
    plt.savefig(save_path, dpi=300)
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