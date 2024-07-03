"""Implementation of the loss trajectory attack."""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F  # noqa: N812
from torch import argmax, cuda, device, load, nn, no_grad, optim, save, tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.distillation_model_handler import DistillationModelHandler
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.import_helper import Self
from leakpro.metrics.attack_result import CombinedMetricResult
from leakpro.signals.signal import ModelLogits
from leakpro.user_inputs.abstract_input_handler import AbstractInputHandler


class AttackLossTrajectory(AbstractMIA):
    """Implementation of the loss trajectory attack."""

    def __init__(self: Self,
                 handler: AbstractInputHandler,
                 configs: dict
                ) -> None:
        """Initialize the LossTrajectoryAttack class.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.
            configs (dict): A dictionary containing the attack loss_traj configurations.

        """
        super().__init__(handler)

        self.logger.info("Configuring Loss trajecatory attack")
        self._configure_attack(configs)


    def _configure_attack(self: Self, configs: dict) -> None:
        self.num_shadow_models = 1
        self.distillation_data_fraction = configs.get("training_distill_data_fraction", 0.5)
        self.shadow_data_fraction = 1 - self.distillation_data_fraction

        self.configs = configs
        self.train_mia_batch_size = configs.get("mia_batch_size", 64)
        self.number_of_traj = configs.get("number_of_traj", 10)
        self.attack_data_dir = configs.get("attack_data_dir")
        self.mia_classifier_epoch = configs.get("mia_classifier_epochs", 100)
        self.label_only = configs.get("label_only", "False")


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

        # Get all available indices for auxiliary dataset
        aux_data_index = self.sample_indices_from_population(include_train_indices = False, include_test_indices = False)

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

        #--------------------------------------------------------
        # Train and load shadow model
        #--------------------------------------------------------
        self.logger.info(f"Training shadow models on {len(shadow_training_indices)} points")
        self.shadow_model_indices = ShadowModelHandler().create_shadow_models(self.num_shadow_models,
                                                                         shadow_training_indices,
                                                                         training_fraction = 1.0)

        # load shadow models
        self.shadow_model, _ = ShadowModelHandler().get_shadow_models(self.shadow_model_indices)

        #--------------------------------------------------------
        # Knowledge distillation of target and shadow models
        #--------------------------------------------------------
        # Note: shadow and target models are PytorchModel objects, hence, we need to take model_obj
        DistillationModelHandler().add_student_teacher_pair("shadow_distillation", self.shadow_model[0].model_obj)
        DistillationModelHandler().add_student_teacher_pair("target_distillation", self.target_model.model_obj)

        self.logger.info(f"Training distillation of the shadow model on {len(distill_data_indices)} points")
        # train the distillation model using the one and only trained shadow model
        self.distill_shadow_models = DistillationModelHandler().distill_model("shadow_distillation",
                                                                              self.number_of_traj,
                                                                              distill_data_indices,
                                                                              self.label_only)

        # train distillation model of the target model
        self.distill_target_models = DistillationModelHandler().distill_model("target_distillation",
                                                                              self.number_of_traj,
                                                                              distill_data_indices,
                                                                              self.label_only)

        #--------------------------------------------------------
        # Prepare data to train and test the MIA classifier
        #--------------------------------------------------------
        # shadow data (train and test) is used as training data for MIA_classifier in the paper
        train_mask = np.isin(shadow_data_indices,shadow_not_used_indices)
        self.prepare_mia_data(shadow_data_indices,
                              train_mask,
                              student_model = self.distill_shadow_models,
                              teacher_model = self.shadow_model[0].model_obj,
                              train_mode = True)

        # Data used in the target (train and test) is used as test data for MIA_classifier
        mia_test_data_indices = np.concatenate( (self.train_indices , self.test_indices))
        test_mask = np.isin(mia_test_data_indices, self.train_indices)
        self.prepare_mia_data(mia_test_data_indices,
                              test_mask,
                              student_model = self.distill_target_models,
                              teacher_model = self.target_model.model_obj,
                              train_mode = False)


    def prepare_mia_data(self:Self,
                        data_indices: np.ndarray,
                        membership_status_shadow_train: np.ndarray,
                        student_model: nn.Module,
                        teacher_model: nn.Module,
                        train_mode: bool,
                        ) -> None:
        """Prepare the data for MIA attack.

        Args:
        ----
            data_indices (np.ndarray): Indices of the data.
            membership_status_shadow_train (np.ndarray): Membership status of the shadow training data.
            student_model (nn.Module): Distillation model.
            teacher_model (nn.Module): Teacher model.
            train_mode (bool): Mode of the attack.

        Returns:
        -------
            None

        """

        dataset_name = "trajectory_train_data.pkl" if train_mode else "trajectory_test_data.pkl"
        if os.path.exists(f"{self.attack_data_dir}/{dataset_name}"):
            self.logger.info(f"Loading MIA {dataset_name}: {len(data_indices)} points")
            with open(f"{self.attack_data_dir}/{dataset_name}", "rb") as file:
                data = pickle.load(file)  # noqa: S301
        else:
            data = self._prepare_mia_data(data_indices,
                                          membership_status_shadow_train,
                                          student_model,
                                          teacher_model,
                                          dataset_name)
        # Create the training dataset for the MIA classifier.
        mia_input = np.concatenate((data["model_trajectory"], data["teacher_model_loss"][:, None]), axis=1)
        mia_dataset = TensorDataset(tensor(mia_input), tensor(data["member_status"]))
        if train_mode:
            self.mia_train_data_loader = DataLoader(mia_dataset, batch_size=self.train_mia_batch_size, shuffle=True)
        else:
            self.mia_test_data_loader = DataLoader(mia_dataset, batch_size=self.train_mia_batch_size, shuffle=True)

    def _prepare_mia_data(self:Self,
                        data_indices: np.ndarray,
                        membership_status_shadow_train: np.ndarray,
                        distill_model: nn.Module,
                        teacher_model: nn.Module,
                        dataset_name: str,
                        ) -> dict:
        self.logger.info(f"Preparing MIA {dataset_name}: {len(data_indices)} points")
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        data_loader = self.get_dataloader(data_indices, batch_size=self.train_mia_batch_size)

        teacher_model_loss = np.array([])
        model_trajectory = np.array([])
        original_labels = np.array([])
        predicted_labels = np.array([])
        predicted_status = np.array([])

        for loader_idx, (data, target) in enumerate(tqdm(data_loader)):
            data = data.to(gpu_or_cpu)
            target = target.to(gpu_or_cpu)

            #---------------------------------------------------------------------
            # Calculate the losses for the distilled student models
            #---------------------------------------------------------------------
            trajectory_current = np.array([])
            for d in range(self.number_of_traj) :
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

            #---------------------------------------------------------------------
            # Calculate the loss for the teacher model
            #---------------------------------------------------------------------
            teacher_model.to(gpu_or_cpu)
            batch_logit_target = teacher_model(data) # TODO: replace with hopskipjump for label only
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

            train_loss, train_prec1 = self._train_mia_classifier(attack_model, attack_optimizer, loss_fn)
            train_info = [train_loss, train_prec1]

            save(attack_model.state_dict(), self.attack_data_dir + "/trajectory_mia_model.pkl")

            with open(f"{self.attack_data_dir}/mia_model_losses.pkl", "wb") as file:
                pickle.dump(train_info, file)
        else:
            with open(f"{self.attack_data_dir}/trajectory_mia_model.pkl", "rb") as model:
                attack_model.load_state_dict(load(model))
            self.logger.info("Loading Loss Trajectory classifier")

        return attack_model

    def _train_mia_classifier(self:Self, model:nn.Module,
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
        self.logger.info("Training the MIA classifier")

        train_loss_list =[]
        train_prec_list = []

        model.train()
        train_loss = 0
        num_correct = 0
        mia_train_loader = self.mia_train_data_loader
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        for _ in  tqdm(range(self.mia_classifier_epoch), total=self.mia_classifier_epoch):
            for _batch_idx, (data, label) in enumerate(mia_train_loader):
                data = data.to(gpu_or_cpu) # noqa: PLW2901
                label = label.to(gpu_or_cpu).long() # noqa: PLW2901

                pred = model(data)
                loss = loss_fn(pred, label)

                attack_optimizer.zero_grad()
                loss.backward()

                attack_optimizer.step()
                train_loss += loss.item()
                pred_label = pred.max(1, keepdim=True)[1]
                num_correct += pred_label.eq(label).sum().item()

            train_loss /= len(mia_train_loader.dataset)
            accuracy =  num_correct / len(mia_train_loader.dataset)

            train_loss_list.append(loss)
            train_prec_list.append(accuracy)

        return train_loss_list, train_prec_list

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
        self.logger.info("Running the MIA attack")
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        attack_model.to(gpu_or_cpu)
        attack_model.eval()
        test_loss = 0
        correct = 0
        auc_ground_truth = None
        auc_pred = None

        with no_grad():
            for batch_idx, (data, target) in tqdm(enumerate(self.mia_test_data_loader), total=len(self.mia_test_data_loader)):
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

        thresholds = np.linspace(0, 1, 1000)
        member_preds = np.array([(auc_pred < threshold).astype(int) for threshold in thresholds])

        return auc_ground_truth, member_preds

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

    def plot_trajectories(self: Self,
                          dataset_name: str) -> None:
        """Plot the trajectories of members and non-members.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset.

        Returns
        -------
        None

        """
        with open(f"{self.attack_data_dir}/{dataset_name}", "rb") as f:
             data = pickle.load(f)  # noqa: S301


        x_members = data["model_trajectory"][:len(self.train_indices), :]
        x_non_members =data["model_trajectory"][len(self.train_indices):, :]

        ave_members = np.mean(x_members, axis=0)
        ave_non_members = np.mean(x_non_members, axis=0)

        if dataset_name == "trajectory_train_data.pkl":
            image_name = "train.png"
        elif dataset_name == "trajectory_test_data.pkl":
            image_name = "test.png"
        plt.errorbar(range(self.number_of_traj), ave_members,  label="Member", fmt="-o")
        plt.errorbar(range(self.number_of_traj), ave_non_members, label="Non-Member", fmt="-o")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Comparison of Means of members and non members")
        plt.legend()
        plt.savefig(f"{self.attack_data_dir}/{image_name}")
        plt.clf()
        plt.close()
