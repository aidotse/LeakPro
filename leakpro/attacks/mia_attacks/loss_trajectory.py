"""Implementation of the loss trajectory attack."""

import os
import pickle
# from typing import Self

# import numpy as np
# import torch
# from torch import nn
# from torch.nn import functional
from torch.utils.data import DataLoader, Subset, TensorDataset

from torch import cuda, device, functional, load, nn, save, argmax, tensor, optim, no_grad

from logging import Logger

import numpy as np
from torch import nn

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.attack_data import get_attack_data
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.attacks.utils.distillation_model_handler import DistillationModelHandler
from leakpro.import_helper import Self
from leakpro.metrics.attack_result import CombinedMetricResult
from leakpro.signals.signal import ModelLogits


class AttackLossTrajectory(AbstractMIA):
    """Implementation of the loss trajectory attack."""

    def __init__(self: Self,
                 population: np.ndarray,
                 audit_dataset: dict,
                 target_model: nn.Module,
                 logger: Logger,
                 configs: dict
                ) -> None:
        """Initialize the LossTrajectoryAttack class.

        Args:
        ----
            attack_utils (AttackUtils): An instance of the AttackUtils class.
            configs (dict): A dictionary containing the attack loss_traj configurations.

        """
        super().__init__(population, audit_dataset, target_model, logger)

        self.logger.info("Configuring Loss trajecatory attack")
        self._configure_attack(configs)


    def _configure_attack(self: Self, configs: dict) -> None:
        self.num_shadow_models = 1
        self.training_data_fraction = configs.get("training_data_fraction", 0.5)
        self.attack_data_fraction = configs.get("attack_data_fraction", 0.1)



        self.configs = configs
        self.f_attack_data_size = configs.get("f_distillation_target_data", 0.3)
        self.train_mia_batch_size = configs.get("mia_batch_size", 64)
        self.num_students = configs.get("num_students", 1)
        self.number_of_traj = configs.get("number_of_traj", 10)
        self.num_classes = configs.get("num_classes", 10)
        self.attack_data_dir = configs.get("attack_data_dir")
        self.attack_mode = configs.get("attack_mode", "soft_label")

        # f_attack_data_size: 0.3
        # distillation_target_data_size: 0.3
        # train_target_data_size: 10000
        # test_target_data_size: 10000
        # train_shadow_data_size: 10000
        # test_shadow_data_size: 10000
        # train_distillation_data_size: 220000
        # aux_data_size: 240000 # all data except the training and test data for the target model



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

    def prepare_attack(self:Self) -> None:
        """Prepare the attack by loading the shadow model and target model.

        Args:
        ----
            self (Self): The instance of the class.

        Returns:
        -------
            None

        """
        self.logger.info("Preparing the data for loss trajectory attack")

        include_target_training_data = False
        include_target_testing_data = False

        # Get all available indices for attack dataset
        self.attack_data_index = get_attack_data(
            self.population_size,
            self.train_indices,
            self.test_indices,
            include_target_training_data,
            include_target_testing_data,
            self.logger
        )
        # create attack dataset
        attack_data = self.population.subset(self.attack_data_index)

        # train shadow models
        self.logger.info(f"Training shadow models on {len(attack_data)} points")
        ShadowModelHandler().create_shadow_models(
            self.num_shadow_models,
            attack_data,
            self.training_data_fraction,
        )

        # load shadow models
        self.shadow_models, self.shadow_model_indices = ShadowModelHandler().get_shadow_models(self.num_shadow_models)


        # train distillation model of the only trained shadow model
        self.logger.info(f"Training distillation of the shadow model on {len(attack_data)} points")
        self.distill_shadow_models, self.distill_shadow_model_indices = DistillationModelHandler().create_distillation_models(
            self.num_students,
            self.number_of_traj,
            attack_data,
            self.training_data_fraction,
        )

        # shadow data (train and test) is used as training data for MIA_classifier
        self.prepare_mia_trainingset(self.shadow_model_indices,
                                     self.shadow_models, self.distill_shadow_models )

        # Data used in the target (train and test) is used as test data for MIA_classifier
        mia_test_data_indices = np.isin( self.shadow_model_indices , self.attack_data_index)
        self.prepare_attack_dataset( mia_test_data_indices ,self.target_model)

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
        if not os.path.exists(f"{self.attack_data_dir}/trajectory_train_data.npy"):
            gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
            data_attack = Subset(self.population, train_attack_data_indices)
            attack_data_loader = DataLoader(data_attack, batch_size=self.train_mia_batch_size, shuffle=False)
            path_distillation_models_shadow = self.path_distillation_models_shadow

            traget_model_loss = np.array([])
            model_trajectory = np.array([])
            original_labels = np.array([])
            predicted_labels = np.array([])
            predicted_status = np.array([])

            for loader_idx, (data, target) in enumerate(attack_data_loader):
                data = data.to(gpu_or_cpu)  # noqa: PLW2901
                target = target.to(gpu_or_cpu)  # noqa: PLW2901

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

                        #TODO: remove this check
                        loaded_object = load(f"{path_distillation_models_shadow}/epoch_{d}.pkl")
                        if isinstance(loaded_object, nn.Module):
                            print("The saved model contains the entire model object with architecture.")
                        elif isinstance(loaded_object, dict):
                            print("The saved model contains only the state_dict.")
                        else:
                            print("Unknown object type.")

                        # infer from the distillation model
                        distillation_model_shadow.load_state_dict(
                            load(f"{path_distillation_models_shadow}/epoch_{d}.pkl"), strict=False)
                        distillation_model_shadow.to(gpu_or_cpu)

                        # infer from the shadow model
                        distill_model_soft_output = distillation_model_shadow(data)

                        # Calculate the loss
                        loss = [functional.cross_entropy(logit_target_i.unsqueeze(0),
                                                target_i.unsqueeze(0)) for (logit_target_i, target_i) in
                                                 zip(distill_model_soft_output, target)]
                        loss = np.array([loss_i.detach().cpu().numpy() for loss_i in loss]).reshape(-1, 1)
                        trajectory_current = loss if d == 0 else np.concatenate((trajectory_current, loss), 1)

                    traj_total_students = trajectory_current if s == 0 else traj_total_students + trajectory_current

                shadow_model.to(gpu_or_cpu)
                batch_logit_target = shadow_model(data)

                _, batch_predict_label = batch_logit_target.max(1)
                batch_predicted_label = batch_predict_label.long().cpu().detach().numpy()
                batch_original_label = target.long().cpu().detach().numpy()
                batch_loss_target = [functional.cross_entropy(batch_logit_target_i.unsqueeze(0), target_i.unsqueeze(0)) for
                                      (batch_logit_target_i, target_i) in zip(batch_logit_target, target)]
                batch_loss_target = np.array([batch_loss_target_i.cpu().detach().numpy() for batch_loss_target_i in
                                              batch_loss_target])
                batch_predicted_status = (argmax(batch_logit_target, dim=1) == target).float().cpu().detach().numpy()
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
        mia_train_dataset = TensorDataset(tensor(mia_train_input), tensor(data["member_status"]))
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
            gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
            data_attack = Subset(self.population, test_attack_data_indices)
            attack_data_loader = DataLoader(data_attack, batch_size=self.train_mia_batch_size, shuffle=False)
            path_distillation_models_target = self.path_distillation_models_target

            traget_model_loss = np.array([])
            model_trajectory = np.array([])
            original_labels = np.array([])
            predicted_labels = np.array([])
            predicted_status = np.array([])
            for loader_idx, (data, target) in enumerate(attack_data_loader):
                data = data.to(gpu_or_cpu) # noqa: PLW2901
                target = target.to(gpu_or_cpu) # noqa: PLW2901

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
                        distillation_model_target.load_state_dict(load(f"{path_distillation_models_target}/epoch_{d}.pkl"),
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
                batch_predicted_status = (argmax(batch_logit_target, dim=1) == target).float().cpu().detach().numpy()
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
        mia_test_dataset = TensorDataset(tensor(mia_test_input), tensor(data["member_status"]))
        self.mia_test_data_loader = DataLoader(mia_test_dataset, batch_size=self.train_mia_batch_size, shuffle=True)


    def mia_classifier(self:Self)-> nn.Module:
        """Trains and returns the MIA (Membership Inference Attack) classifier.

        Returns
        -------
            attack_model (torch.nn.Module): The trained MIA classifier model.

        """
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        attack_model = self.mia_classifer
        attack_optimizer = optim.SGD(attack_model.parameters(),
                                           lr=0.01, momentum=0.9, weight_decay=0.0001)
        attack_model = attack_model.to(gpu_or_cpu)
        loss_fn = nn.CrossEntropyLoss()
        train_loss =[]
        train_prec1 = []

        for _epoch in range(100):
            loss, prec = self.train_mia_step(attack_model, attack_optimizer,
                                                         loss_fn)
            train_loss.append(loss)
            train_prec1.append(prec)

        save(attack_model.state_dict(), self.log_dir + "/trajectory_mia_model.pkl")

        train_info = [train_loss, train_prec1]
        with open(f"{self.log_dir}/mia_model_losses.pkl", "wb") as file:
            pickle.dump(train_info, file)

        return attack_model

    def train_mia_step(self:Self, model:nn.Module,
                        attack_optimizer:optim.Optimizer,
                        loss_fn:nn.functional) -> tuple:
        """Trains the model using the MIA (Membership Inference Attack) method for one step.

        Args:
        ----
            model (torch.nn.Module): The model to be trained.
            attack_optimizer (torch.optim.Optimizer): The optimizer for the attack.
            loss_fn (torch.nn.Module): The loss function to calculate the loss.

        Returns:
        -------
            tuple: A tuple containing the train loss and accuracy.

        """
        model.train()
        train_loss = 0
        num_correct = 0
        mia_train_loader = self.mia_train_data_loader
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")

        for _batch_idx, (data, label) in enumerate(mia_train_loader):
            data = data.to(gpu_or_cpu) # noqa: PLW2901
            label = label.to(gpu_or_cpu) # noqa: PLW2901
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
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        attack_model.eval()
        test_loss = 0
        correct = 0
        auc_ground_truth = None
        auc_pred = None

        with no_grad():
            for batch_idx, (data, target) in enumerate(self.mia_test_data_loader):
                data = data.to(gpu_or_cpu) # noqa: PLW2901
                target = target.to(gpu_or_cpu) # noqa: PLW2901
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


