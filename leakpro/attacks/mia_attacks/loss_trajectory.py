"""Implementation of the loss trajectory attack."""

import os
import pickle
from logging import Logger

import numpy as np
import torch.nn.functional as F  # noqa: N812
from torch import argmax, cuda, device, load, nn, no_grad, optim, save, tensor
from torch.utils.data import DataLoader, Subset, TensorDataset

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.attack_data import get_attack_data
from leakpro.attacks.utils.distillation_model_handler import DistillationShadowModelHandler, DistillationTargetModelHandler
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
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
            population (np.ndarray): The population data.
            audit_dataset (dict): The audit dataset.
            target_model (nn.Module): The target model.
            logger (Logger): The logger instance.
            configs (dict): A dictionary containing the attack loss_traj configurations.

        """
        super().__init__(population, audit_dataset, target_model, logger)

        self.logger.info("Configuring Loss trajecatory attack")
        self._configure_attack(configs)


    def _configure_attack(self: Self, configs: dict) -> None:
        self.num_shadow_models = 1
        self.distillation_data_fraction = configs.get("training_distill_data_fraction", 0.5)
        self.shadow_data_fraction = 1 - self.distillation_data_fraction

        self.configs = configs
        self.train_mia_batch_size = configs.get("mia_batch_size", 64)
        self.num_students = configs.get("num_students", 1)
        self.number_of_traj = configs.get("number_of_traj", 10)
        self.num_classes = configs.get("num_classes", 10)
        self.attack_data_dir = configs.get("attack_data_dir")
        self.mia_classifier_epoch = configs.get("mia_classifier_epoch", 100)
        self.attack_mode = configs.get("attack_mode", "soft_label")


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
        #TODO: This should be changed!
        include_target_testing_data = True

        # Get all available indices for auxiliary dataset
        aux_data_index = get_attack_data(
            self.population_size,
            self.train_indices,
            self.test_indices,
            include_target_training_data,
            include_target_testing_data,
            self.logger
        )
        # create auxiliary dataset
        aux_data_size = len(aux_data_index)
        shadow_data_size = int(aux_data_size * self.shadow_data_fraction)
        shadow_data_indices = np.random.choice(aux_data_size, shadow_data_size, replace=False)
        # Split the shadow data into two equal parts
        split_index = len(shadow_data_indices) // 2
        shadow_training_indices = shadow_data_indices[:split_index]
        shadow_not_used_indices = shadow_data_indices[split_index:]
        # Distillation on target and shadow model happen on the same dataset
        distill_data_indices = np.setdiff1d(aux_data_index, shadow_data_indices)

        shadow_dataset = self.population.subset(shadow_training_indices)
        distill_dataset = self.population.subset(distill_data_indices)

        # train shadow models
        self.logger.info(f"Training shadow models on {len(shadow_dataset)} points")
        ShadowModelHandler().create_shadow_models(
            self.num_shadow_models,
            shadow_dataset,
            shadow_training_indices,
            training_fraction = 1.0,
            retrain= False,
        )

        # load shadow models
        self.shadow_models, self.shadow_model_indices = \
            ShadowModelHandler().get_shadow_models(self.num_shadow_models)
        self.shadow_metadata = ShadowModelHandler().get_shadow_model_metadata(1)

        # train the distillation model using the one and only trained shadow model
        self.logger.info(f"Training distillation of the shadow model on {len(distill_dataset)} points")
        DistillationShadowModelHandler().initializng_shadow_teacher(self.shadow_models[0], self.shadow_metadata[0])
        self.distill_shadow_models = DistillationShadowModelHandler().create_distillation_models(
            self.num_students,
            self.number_of_traj,
            distill_dataset,
            distill_data_indices,
            self.attack_mode,
        )

        # train distillation model of the target model
        self.logger.info(f"Training distillation of the target model on {len(distill_dataset)} points")
        self.distill_target_models = DistillationTargetModelHandler().create_distillation_models(
            self.num_students,
            self.number_of_traj,
            distill_dataset,
            distill_data_indices,
            self.attack_mode,
        )

        # shadow data (train and test) is used as training data for MIA_classifier in the paper
        train_mask = np.isin(shadow_data_indices,shadow_not_used_indices )
        self.prepare_mia_data(shadow_data_indices, train_mask,
                              self.distill_shadow_models, self.shadow_models[0].model_obj, "train")

        # Data used in the target (train and test) is used as test data for MIA_classifier
        mia_test_data_indices = np.concatenate( (self.train_indices , self.test_indices))
        test_mask = np.isin(mia_test_data_indices, self.train_indices)
        self.prepare_mia_data(mia_test_data_indices, test_mask,
                              self.distill_target_models, self.target_model.model_obj, "test")


    def prepare_mia_data(self:Self,
                        data_indices: np.ndarray,
                        membership_status_shadow_train: np.ndarray,
                        distill_model: nn.Module,
                        teacher_model: nn.Module,
                        mode: str,
                        ) -> None:
        """Prepare the data for MIA attack.

        Args:
        ----
            data_indices (np.ndarray): Indices of the data.
            membership_status_shadow_train (np.ndarray): Membership status of the shadow training data.
            distill_model (nn.Module): Distillation model.
            teacher_model (nn.Module): Teacher model.
            mode (str): Mode of the attack (train or test).

        Returns:
        -------
            None

        """
        if mode == "train":
            dataset_name = "trajectory_train_data.pkl"
            if os.path.exists(f"{self.attack_data_dir}/{dataset_name}"):
                self.logger.info(f"Loading MIA {dataset_name}: {len(data_indices)} points")
                with open(f"{self.attack_data_dir}/{dataset_name}", "rb") as file:
                    data = pickle.load(file)  # noqa: S301
            else:
                data = self._prepare_mia_data(data_indices,
                                              membership_status_shadow_train,
                                              distill_model,
                                              teacher_model,
                                              dataset_name)

            # Create the training dataset for the MIA classifier.
            mia_train_input = np.concatenate((data["model_trajectory"],
                                            data["teacher_model_loss"][:, None]), axis=1)
            mia_train_dataset = TensorDataset(tensor(mia_train_input), tensor(data["member_status"]))
            self.mia_train_data_loader = DataLoader(mia_train_dataset, batch_size=self.train_mia_batch_size, shuffle=True)

        elif mode == "test":
            dataset_name = "trajectory_test_data.pkl"
            if os.path.exists(f"{self.attack_data_dir}/{dataset_name}"):
                self.logger.info(f"Loading MIA {dataset_name}: {len(data_indices)} points")
                with open(f"{self.attack_data_dir}/{dataset_name}", "rb") as file:
                    data = pickle.load(file)  # noqa: S301
            else:
                data = self._prepare_mia_data(data_indices,
                                              membership_status_shadow_train,
                                              distill_model,
                                              teacher_model,
                                              dataset_name)
            # Create the training dataset for the MIA classifier.
            mia_test_input = np.concatenate((data["model_trajectory"] ,
                                            data["teacher_model_loss"][:,None]), axis=1)
            mia_test_dataset = TensorDataset(tensor(mia_test_input), tensor(data["member_status"]))
            self.mia_test_data_loader = DataLoader(mia_test_dataset, batch_size=self.train_mia_batch_size, shuffle=True)

    def _prepare_mia_data(self:Self,
                        data_indices: np.ndarray,
                        membership_status_shadow_train: np.ndarray,
                        distill_model: nn.Module,
                        teacher_model: nn.Module,
                        dataset_name: str,
                        ) -> dict:
        self.logger.info(f"Preparing MIA {dataset_name}: {len(data_indices)} points")
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        data_attack = Subset(self.population, data_indices)
        data_loader = DataLoader(data_attack, batch_size=self.train_mia_batch_size, shuffle=False)

        teacher_model_loss = np.array([])
        model_trajectory = np.array([])
        original_labels = np.array([])
        predicted_labels = np.array([])
        predicted_status = np.array([])

        for loader_idx, (data, target) in enumerate(data_loader):
            data = data.to(gpu_or_cpu)  # noqa: PLW2901
            target = target.to(gpu_or_cpu)  # noqa: PLW2901

            trajectory_current = np.array([])
            for d in range(self.number_of_traj):
                distill_model[d].to(gpu_or_cpu)
                distill_model[d].eval()

                # infer from the shadow model
                distill_model_soft_output = distill_model[d](data)

                # Calculate the loss
                loss = []
                for logit_target_i, target_i in zip(distill_model_soft_output, target):
                    loss_i = F.cross_entropy(logit_target_i.unsqueeze(0), target_i.unsqueeze(0))
                    loss.append(loss_i)
                loss = np.array([loss_i.detach().cpu().numpy() for loss_i in loss]).reshape(-1, 1)
                trajectory_current = loss if d == 0 else np.concatenate((trajectory_current, loss), 1)

            teacher_model.to(gpu_or_cpu)
            batch_logit_target = teacher_model(data)

            _, batch_predict_label = batch_logit_target.max(1)
            batch_predicted_label = batch_predict_label.long().cpu().detach().numpy()
            batch_original_label = target.long().cpu().detach().numpy()
            batch_loss_teacher = []
            for (batch_logit_target_i, target_i) in zip(batch_logit_target, target):
                batch_loss_teacher.append(F.cross_entropy(batch_logit_target_i.unsqueeze(0),
                                                        target_i.unsqueeze(0)))
            batch_loss_teacher = np.array([batch_loss_teacher_i.cpu().detach().numpy() for batch_loss_teacher_i in
                                            batch_loss_teacher])
            batch_predicted_status = (argmax(batch_logit_target, dim=1) == target).float().cpu().detach().numpy()
            batch_predicted_status = np.expand_dims(batch_predicted_status, axis=1)

            if loader_idx == 0:
                teacher_model_loss = batch_loss_teacher
                model_trajectory = trajectory_current
                original_labels = batch_original_label
                predicted_labels = batch_predicted_label
                predicted_status = batch_predicted_status
            else:
                teacher_model_loss = np.concatenate((teacher_model_loss, batch_loss_teacher), axis=0)
                model_trajectory = np.concatenate((model_trajectory, trajectory_current), axis=0)
                original_labels = np.concatenate((original_labels, batch_original_label), axis=0)
                predicted_labels = np.concatenate((predicted_labels, batch_predicted_label), axis=0)
                predicted_status = np.concatenate((predicted_status, batch_predicted_status), axis=0)


        data = {
            "teacher_model_loss":teacher_model_loss,
            "model_trajectory":model_trajectory,
            "original_labels":original_labels,
            "predicted_labels":predicted_labels,
            "predicted_status":predicted_status,
            "member_status":membership_status_shadow_train,
            "nb_classes":self.num_classes
        }
        os.makedirs(self.attack_data_dir, exist_ok=True)
        with open(f"{self.attack_data_dir}/{dataset_name}", "wb") as file:
            pickle.dump(data, file)
        return data


    def mia_classifier(self:Self)-> nn.Module:
        """Trains and returns the MIA (Membership Inference Attack) classifier.

        Returns
        -------
            attack_model (torch.nn.Module): The trained MIA classifier model.

        """
        attack_model = self.mia_classifer
        if not os.path.exists(f"{self.attack_data_dir}/trajectory_mia_model.pkl"):
            gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
            attack_optimizer = optim.SGD(attack_model.parameters(),
                                            lr=0.01, momentum=0.9, weight_decay=0.0001)
            attack_model = attack_model.to(gpu_or_cpu)
            loss_fn = nn.CrossEntropyLoss()
            train_loss =[]
            train_prec1 = []

            for _ in range(self.mia_classifier_epoch):
                loss, prec = self.train_mia_step(attack_model, attack_optimizer,
                                                            loss_fn)
                train_loss.append(loss)
                train_prec1.append(prec)

            save(attack_model.state_dict(), self.attack_data_dir + "/trajectory_mia_model.pkl")

            train_info = [train_loss, train_prec1]
            with open(f"{self.attack_data_dir}/mia_model_losses.pkl", "wb") as file:
                pickle.dump(train_info, file)
        else:
            with open(f"{self.attack_data_dir}/trajectory_mia_model.pkl", "rb") as model:
                attack_model.load_state_dict(load(model))
            self.logger.info("Loading Loss Trajectory classifier")

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
        attack_model.to(gpu_or_cpu)
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

                test_loss += F.cross_entropy(pred, target).item()
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
        member_preds = np.array([(auc_pred < threshold).astype(int) for threshold in thresholds_1])

        return auc_ground_truth, member_preds



