"""Implementation of the loss trajectory attack."""

import os
import pickle
from typing import Self

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, Subset, TensorDataset

from leakpro.metrics.attack_result import CombinedMetricResult
from leakpro.mia_attacks.attack_utils import AttackUtils
from leakpro.mia_attacks.attacks.attack import AttackAbstract
from leakpro.signals.signal import ModelLogits


class AttackLossTrajectory(AttackAbstract):
    """Implementation of the loss trajectory attack."""

    def __init__(self: Self,
                 attack_utils: AttackUtils,
                 configs: dict
                ) -> None:
        """Initialize the LossTrajectoryAttack class.

        Args:
        ----
            attack_utils (AttackUtils): An instance of the AttackUtils class.
            configs (dict): A dictionary containing the attack configurations.

        """
        super().__init__(attack_utils)

        self.shadow_models = attack_utils.attack_objects.shadow_models
        self.distillation_models_target = attack_utils.attack_objects.distillation_models_target
        self.distillation_models_shadow = attack_utils.attack_objects.distillation_models_shadow

        self.target_train_indices = attack_utils.attack_objects.train_test_dataset["train_indices"]
        self.target_test_indices = attack_utils.attack_objects.train_test_dataset["test_indices"]
        self.shadow_train_indices = attack_utils.attack_objects._shadow_train_indices
        self.shadow_test_indices = attack_utils.attack_objects._shadow_test_indices

        self.configs = configs
        self.log_dir = configs["run"]["log_dir"]
        self.f_attack_data_size = configs["audit"]["f_distillation_target_data"]
        self.train_mia_batch_size = configs["audit"]["audit_batch_size"]
        self.num_students = configs["loss_traj"]["num_students"]
        self.number_of_traj =configs["loss_traj"]["number_of_traj"]
        self.num_classes = configs["loss_traj"]["num_classes"]

        self.path_distillation_models_target = f"{self.log_dir}/distillation_models_target"
        self.path_distillation_models_shadow = f"{self.log_dir}/distillation_models_shadow"
        self.path_shadow_model = f"{self.log_dir}/shadow_models/model_0.pkl"
        self.path_target_model = f"{self.log_dir}/model_0.pkl"

        self.read_from_file = False
        self.temperature = 2.0 # temperature for the softmax
        self.signal = ModelLogits()
        self.mia_train_data_loader = None
        self.mia_test_data_loader = None
        self.dim_in_mia = (self.number_of_traj + 1 )
        self.mia_classifer = nn.Sequential(nn.Linear(self.dim_in_mia, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 128),
                                            nn.ReLU(),
                                            nn.Linear(128, 32),
                                            nn.ReLU(),
                                            nn.Linear(32, 2),
                                            nn.Softmax(dim=1)
                                           )


    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "Membership Inference Attacks by Exploiting Loss Trajectory"
        reference_str = "Yiyong Liu, Zhengyu Zhao, Michael Backes, Yang Zhang \
            Membership Inference Attacks by Exploiting Loss Trajectory. (2022)."
        summary_str = " "
        detailed_str = ""
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }

    def softmax(self:Self, all_logits:np.ndarray,
                true_label_indices:np.ndarray,
                return_full_distribution:bool=False) -> np.ndarray:
        """Compute the softmax function.

        Args:
        ----
            all_logits (np.ndarray): Logits for each class.
            true_label_indices (np.ndarray): Indices of the true labels.
            return_full_distribution (bool, optional): return the full distribution or just the true class probabilities.

        Returns:
        -------
            np.ndarray: Softmax output.

        """
        logit_signals = all_logits / self.temperature
        max_logit_signals = np.max(logit_signals,axis=2)
        logit_signals = logit_signals - max_logit_signals.reshape(1,-1,1)
        exp_logit_signals = np.exp(logit_signals)
        exp_logit_sum = np.sum(exp_logit_signals, axis=2)

        if return_full_distribution is False:
            true_exp_logit =  exp_logit_signals[:, np.arange(exp_logit_signals.shape[1]), true_label_indices]
            output_signal = true_exp_logit / exp_logit_sum
        else:
            output_signal = exp_logit_signals / exp_logit_sum[:,:,np.newaxis]
        return output_signal


    def prepare_attack(self:Self) -> None:
        """Prepare the attack by loading the shadow model and target model.

        Args:
        ----
            self (Self): The instance of the class.

        Returns:
        -------
            None

        """
        # Load the shadow model
        if "adult" in self.configs["data"]["dataset"]:
            shadow_model = self.target_model.model_obj.__class__(self.configs["train"]["inputs"],
                                                                 self.configs["train"]["outputs"])
        elif "cifar10" in self.configs["data"]["dataset"]:
            shadow_model = self.target_model.model_obj.__class__()
        elif "cinic10" in self.configs["data"]["dataset"]:
            shadow_model = self.target_model.model_obj.__class__(self.configs)
        shadow_model.load_state_dict(torch.load(f"{self.path_shadow_model}"), strict=False)

        # Load the target model
        if "adult" in self.configs["data"]["dataset"]:
            trained_target_model = self.target_model.model_obj.__class__(self.configs["train"]["inputs"],
                                                                          self.configs["train"]["outputs"])
        elif "cifar10" in self.configs["data"]["dataset"]:
            trained_target_model = self.target_model.model_obj.__class__()
        elif "cinic10" in self.configs["data"]["dataset"]:
            trained_target_model = self.target_model.model_obj.__class__(self.configs)
        trained_target_model.load_state_dict(torch.load(f"{self.path_target_model}"), strict=False)

        # shadow data (train and test) is used as training data for MIA_classifier
        mia_train_data_indices = np.concatenate((self.shadow_test_indices, self.shadow_train_indices))
        train_mask = np.isin(mia_train_data_indices, self.shadow_train_indices)
        self.prepare_mia_trainingset(mia_train_data_indices , train_mask,
                                     shadow_model )

        # Data used in the target (train and test) is used as test data for MIA_classifier
        mia_test_data_indices = np.concatenate((self.target_test_indices, self.target_train_indices))
        test_mask = np.isin(mia_test_data_indices, self.target_train_indices)
        self.prepare_attack_dataset( mia_test_data_indices , test_mask,
                                trained_target_model)


    def prepare_mia_trainingset(self:Self, train_attack_data_indices:np.ndarray,  # noqa: PLR0915
                                membership_status_shadow_train:np.ndarray,
                                shadow_model:nn.Module) -> None:
        """Prepare the training set for the Membership Inference Attack (MIA) classifier.

        Args:
        ----
            train_attack_data_indices (np.ndarray): Indices of the training attack data.
            membership_status_shadow_train (np.ndarray): Membership status of the shadow training data.
            shadow_model (nn.Module): The shadow model.

        Returns:
        -------
            None

        """
        if not os.path.exists(f"{self.log_dir}/trajectory_train_data.npy"):
            device = ("cuda" if torch.cuda.is_available() else "cpu")
            data_attack = Subset(self.population, train_attack_data_indices)
            attack_data_loader = DataLoader(data_attack, batch_size=self.train_mia_batch_size, shuffle=False)
            path_distillation_models_shadow = self.path_distillation_models_shadow

            traget_model_loss = np.array([])
            model_trajectory = np.array([])
            original_labels = np.array([])
            predicted_labels = np.array([])
            predicted_status = np.array([])

            for loader_idx, (data, target) in enumerate(attack_data_loader):
                data = data.to(device)  # noqa: PLW2901
                target = target.to(device)  # noqa: PLW2901

                traj_total_students = np.array([])
                for s in range(self.num_students):

                    trajectory_current = np.array([])
                    for d in range(self.number_of_traj):
                        if "adult" in self.configs["data"]["dataset"]:
                            distillation_model_shadow = self.target_model.model_obj.__class__(self.configs["train"]["inputs"],
                                                                                             self.configs["train"]["outputs"])
                        elif "cifar10" in self.configs["data"]["dataset"]:
                            distillation_model_shadow = self.target_model.model_obj.__class__()
                        elif "cinic10" in self.configs["data"]["dataset"]:
                            distillation_model_shadow = self.target_model.model_obj.__class__(self.configs)

                        # infer from the distillation model
                        distillation_model_shadow.load_state_dict(
                            torch.load(f"{path_distillation_models_shadow}/epoch_{d}.pkl"), strict=False)
                        distillation_model_shadow.to(device)

                        # infer from the shadow model
                        distill_model_soft_output = distillation_model_shadow(data)

                        # Calculate the loss
                        loss = [functional.cross_entropy(logit_target_i.unsqueeze(0),
                                                target_i.unsqueeze(0)) for (logit_target_i, target_i) in
                                                 zip(distill_model_soft_output, target)]
                        loss = np.array([loss_i.detach().cpu().numpy() for loss_i in loss]).reshape(-1, 1)
                        trajectory_current = loss if d == 0 else np.concatenate((trajectory_current, loss), 1)

                    traj_total_students = trajectory_current if s == 0 else traj_total_students + trajectory_current

                shadow_model.to(device)
                batch_logit_target = shadow_model(data)

                _, batch_predict_label = batch_logit_target.max(1)
                batch_predicted_label = batch_predict_label.long().cpu().detach().numpy()
                batch_original_label = target.long().cpu().detach().numpy()
                batch_loss_target = [functional.cross_entropy(batch_logit_target_i.unsqueeze(0), target_i.unsqueeze(0)) for
                                      (batch_logit_target_i, target_i) in zip(batch_logit_target, target)]
                batch_loss_target = np.array([batch_loss_target_i.cpu().detach().numpy() for batch_loss_target_i in
                                              batch_loss_target])
                batch_predicted_status = (torch.argmax(batch_logit_target, dim=1) == target).float().cpu().detach().numpy()
                batch_predicted_status = np.expand_dims(batch_predicted_status, axis=1)

                if loader_idx == 0:
                    traget_model_loss = batch_loss_target
                    model_trajectory = traj_total_students
                    original_labels = batch_original_label
                    predicted_labels = batch_predicted_label
                    predicted_status = batch_predicted_status
                else:
                    traget_model_loss = np.concatenate((traget_model_loss, batch_loss_target), axis=0)
                    model_trajectory = np.concatenate((model_trajectory, traj_total_students), axis=0)
                    original_labels = np.concatenate((original_labels, batch_original_label), axis=0)
                    predicted_labels = np.concatenate((predicted_labels, batch_predicted_label), axis=0)
                    predicted_status = np.concatenate((predicted_status, batch_predicted_status), axis=0)



            data = {
                "traget_model_loss":traget_model_loss,
                "model_trajectory":model_trajectory,
                "original_labels":original_labels,
                "predicted_labels":predicted_labels,
                "predicted_status":predicted_status,
                "member_status":membership_status_shadow_train,
                "nb_classes":self.num_classes
            }

            with open(f"{self.log_dir}/trajectory_train_data.pkl", "wb") as pickle_file:
                pickle.dump(data, pickle_file)

        else:
            data = np.load(f"{self.log_dir}/trajectory_train_data.npy", allow_pickle=True).item()

        # Create the training dataset for the MIA classifier.
        mia_train_input = np.concatenate((data["model_trajectory"],
                                          data["traget_model_loss"][:, None]), axis=1)
        mia_train_dataset = TensorDataset(torch.tensor(mia_train_input), torch.tensor(data["member_status"]))
        self.mia_train_data_loader = DataLoader(mia_train_dataset, batch_size=self.train_mia_batch_size, shuffle=True)




    def prepare_attack_dataset(self: Self, test_attack_data_indices:list,  # noqa: PLR0915
                            membership_status_shadow_train:np.ndarray,
                            trained_target_model:nn.Module) -> None:
        """Prepare the test set for the Membership Inference Attack (MIA) classifer.

        The training set contais concatinated losses of distillation modesl on shadow and the shadow model itslef.


        Args:
        ----
            test_attack_data_indices (list): The indices of the test attack data.
            membership_status_shadow_train (ndarray): The membership status of the shadow training data.
            trained_target_model (Model): The trained target model.

        Returns:
        -------
            None

        """
        if not os.path.exists(f"{self.log_dir}/trajectory_test_data.npy"):
            device = ("cuda" if torch.cuda.is_available() else "cpu")
            data_attack = Subset(self.population, test_attack_data_indices)
            attack_data_loader = DataLoader(data_attack, batch_size=self.train_mia_batch_size, shuffle=False)
            path_distillation_models_target = self.path_distillation_models_target

            traget_model_loss = np.array([])
            model_trajectory = np.array([])
            original_labels = np.array([])
            predicted_labels = np.array([])
            predicted_status = np.array([])
            for loader_idx, (data, target) in enumerate(attack_data_loader):
                data = data.to(device) # noqa: PLW2901
                target = target.to(device) # noqa: PLW2901

                traj_total_students = np.array([])
                for s in range(self.num_students):

                    trajectory_current = np.array([])
                    for d in range(self.number_of_traj):
                        if "adult" in self.configs["data"]["dataset"]:
                            distillation_model_target = self.target_model.model_obj.__class__(self.configs["train"]["inputs"],
                                                                                              self.configs["train"]["outputs"])
                        elif "cifar10" in self.configs["data"]["dataset"]:
                            distillation_model_target = self.target_model.model_obj.__class__()
                        elif "cinic10" in self.configs["data"]["dataset"]:
                            distillation_model_target = self.target_model.model_obj.__class__(self.configs)

                        # infer from the distillation model of the target model
                        distillation_model_target.load_state_dict(torch.load(f"{path_distillation_models_target}/epoch_{d}.pkl"),
                                                                  strict=False)
                        distillation_model_target.to(device)

                        #infer from the target model
                        distill_model_soft_output = distillation_model_target(data)

                        # Calculate the loss
                        loss = [functional.cross_entropy(logit_target_i.unsqueeze(0), target_i.unsqueeze(0)) for (logit_target_i,
                                                                     target_i) in zip(distill_model_soft_output, target)]
                        loss = np.array([loss_i.detach().cpu().numpy() for loss_i in loss]).reshape(-1, 1)
                        trajectory_current = loss if d == 0 else np.concatenate((trajectory_current, loss), 1)

                    traj_total_students = trajectory_current if s == 0 else traj_total_students + trajectory_current

                trained_target_model.to(device)
                batch_logit_target = trained_target_model(data)

                _, batch_predict_label = batch_logit_target.max(1)
                batch_predicted_label = batch_predict_label.long().cpu().detach().numpy()
                batch_original_label = target.long().cpu().detach().numpy()
                batch_loss_target = [functional.cross_entropy(batch_logit_target_i.unsqueeze(0), target_i.unsqueeze(0)) for
                                      (batch_logit_target_i, target_i) in zip(batch_logit_target, target)]
                batch_loss_target = np.array([batch_loss_target_i.cpu().detach().numpy() for batch_loss_target_i in
                                              batch_loss_target])
                batch_predicted_status = (torch.argmax(batch_logit_target, dim=1) == target).float().cpu().detach().numpy()
                batch_predicted_status = np.expand_dims(batch_predicted_status, axis=1)

                if loader_idx == 0:
                    traget_model_loss = batch_loss_target
                    model_trajectory = traj_total_students
                    original_labels = batch_original_label
                    predicted_labels = batch_predicted_label
                    predicted_status = batch_predicted_status
                else:
                    traget_model_loss = np.concatenate((traget_model_loss, batch_loss_target), axis=0)
                    model_trajectory = np.concatenate((model_trajectory, traj_total_students), axis=0)
                    original_labels = np.concatenate((original_labels, batch_original_label), axis=0)
                    predicted_labels = np.concatenate((predicted_labels, batch_predicted_label), axis=0)
                    predicted_status = np.concatenate((predicted_status, batch_predicted_status), axis=0)

            data = {
                "traget_model_loss":traget_model_loss,
                "model_trajectory":model_trajectory,
                "original_labels":original_labels,
                "predicted_labels":predicted_labels,
                "predicted_status":predicted_status,
                "member_status":membership_status_shadow_train,
                "nb_classes":self.num_classes
            }

            with open(f"{self.log_dir}/trajectory_test_data.pkl", "wb") as pickle_file:
                pickle.dump(data, pickle_file)

        else:
            data = np.load(f"{self.log_dir}/trajectory_test_data.npy", allow_pickle=True).item()

        mia_test_input = np.concatenate((data["model_trajectory"] , 
                                         data["traget_model_loss"][:,None]), axis=1)
        mia_test_dataset = TensorDataset(torch.tensor(mia_test_input), torch.tensor(data["member_status"]))
        self.mia_test_data_loader = DataLoader(mia_test_dataset, batch_size=self.train_mia_batch_size, shuffle=True)



    def mia_classifier(self:Self)-> torch.nn.Module:
        """Trains and returns the MIA (Membership Inference Attack) classifier.

        Returns
        -------
            attack_model (torch.nn.Module): The trained MIA classifier model.

        """
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        attack_model = self.mia_classifer
        attack_optimizer = torch.optim.SGD(attack_model.parameters(),
                                           lr=0.01, momentum=0.9, weight_decay=0.0001)
        attack_model = attack_model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        train_loss =[]
        train_prec1 = []

        for _epoch in range(100):
            loss, prec = self.train_mia_step(attack_model, attack_optimizer,
                                                         loss_fn, device)
            train_loss.append(loss)
            train_prec1.append(prec)

        torch.save(attack_model.state_dict(), self.log_dir + "/trajectory_mia_model.pkl")

        train_info = [train_loss, train_prec1]
        with open(f"{self.log_dir}/mia_model_losses.pkl", "wb") as file:
            pickle.dump(train_info, file)


        return attack_model


    def train_mia_step(self:Self, model:nn.Module,
                        attack_optimizer:torch.optim.Optimizer,
                        loss_fn:nn.functional, device:torch.device) -> tuple:
        """Trains the model using the MIA (Membership Inference Attack) method for one step.

        Args:
        ----
            model (torch.nn.Module): The model to be trained.
            attack_optimizer (torch.optim.Optimizer): The optimizer for the attack.
            loss_fn (torch.nn.Module): The loss function to calculate the loss.
            device (torch.device): The device to perform the training on.

        Returns:
        -------
            tuple: A tuple containing the train loss and accuracy.

        """
        model.train()
        train_loss = 0
        num_correct = 0
        mia_train_loader = self.mia_train_data_loader

        for _batch_idx, (data, label) in enumerate(mia_train_loader):
            data = data.to(device) # noqa: PLW2901
            label = label.to(device) # noqa: PLW2901
            label = label.long() # noqa: PLW2901

            pred = model(data)
            loss = loss_fn(pred, label)


            attack_optimizer.zero_grad()
            loss.backward()

            attack_optimizer.step()
            train_loss += loss.item()
            pred_label = pred.max(1, keepdim=True)[1]
            num_correct += pred_label.eq(label).sum().item()

        train_loss /= len(mia_train_loader.dataset)
        accuracy = 100. * num_correct / len(mia_train_loader.dataset)

        return train_loss, accuracy/100.


    def run_attack(self:Self) -> CombinedMetricResult:
        """Run the attack and return the combined metric result.

        Returns
        -------
            CombinedMetricResult: The combined metric result containing predicted labels, true labels,
            predictions probabilities, and signal values.

        """
        self.mia_classifier()
        true_labels, predictions = self.mia_attack(self.mia_classifer)

        #NOTE: We don't have signals in this attack, unlike RMIA. I set it to random to pass the PR before refactoring.
        signals = np.random.rand(*true_labels.shape)

        # compute ROC, TP, TN etc
        return CombinedMetricResult(
            predicted_labels= predictions,
            true_labels=true_labels,
            predictions_proba=None,
            signal_values=signals,
        )


    def mia_attack(self:Self, attack_model:nn.Module) -> tuple:
        """Perform a membership inference attack using the given attack model.

        Args:
        ----
            attack_model: The model used for the membership inference attack.

        Returns:
        -------
            A tuple containing the ground truth labels and the predicted membership probabilities for each data point.
            - auc_ground_truth: The ground truth labels for the data points.
            - member_preds: The predicted membership probabilities for each data point.

        """
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        attack_model.eval()
        test_loss = 0
        correct = 0
        auc_ground_truth = None
        auc_pred = None

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.mia_test_data_loader):
                data = data.to(device) # noqa: PLW2901
                target = target.to(device) # noqa: PLW2901
                target = target.long() # noqa: PLW2901
                pred = attack_model(data)

                test_loss += functional.cross_entropy(pred, target).item()
                pred0, pred1 = pred.max(1, keepdim=True)
                correct += pred1.eq(target).sum().item()
                auc_pred_current = pred[:, -1]
                if batch_idx == 0 :
                    auc_ground_truth = target.cpu().numpy()
                    auc_pred = auc_pred_current.cpu().detach().numpy()
                else:
                     auc_ground_truth = np.concatenate((auc_ground_truth, target.cpu().numpy()), axis=0)
                     auc_pred = np.concatenate((auc_pred, auc_pred_current.cpu().detach().numpy()),axis=0)

        test_loss /= len(self.mia_test_data_loader.dataset)
        accuracy = 100. * correct / len(self.mia_test_data_loader.dataset)  # noqa: F841

        thresholds_1 = np.linspace(0, 1, 1000)
        member_preds = np.array([(auc_pred > threshold).astype(int) for threshold in thresholds_1])

        return auc_ground_truth, member_preds

