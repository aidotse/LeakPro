"""Implementation of the sequential MIA attack."""

import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from pydantic import BaseModel, Field
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.distillation_model_handler import DistillationModelHandler
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.signals.signal import ModelLogits
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class LSTM(nn.Module):
    """LSTM model."""

    def __init__(self, input_size: int = 2, hidden_size: int = 4, num_layers: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.label = nn.Linear(hidden_size, num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward call."""
        h0 = Variable(torch.zeros(self.num_layers, x.size(1), self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, x.size(1), self.hidden_size))
        h0, c0 = h0.to(device=self.device), c0.to(device=self.device)

        out, (h1, c1) = self.lstm(x, (h0, c0))
        return self.label(h1)


class LSTMAttention(nn.Module):
    """LSTM model with attention."""

    def __init__(self, input_size: int = 2, hidden_size: int = 4, num_layers: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layers)
        self.layer3 = nn.Linear(hidden_size * 2, num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward call."""
        h0 = Variable(torch.zeros(self.num_layers, x.size(1), self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, x.size(1), self.hidden_size))
        h0, c0 = h0.to(device=self.device), c0.to(device=self.device)

        out, (h1, c1) = self.layer1(x, (h0, c0))

        atten_energies = torch.sum(h1 * out, dim=2)
        scores = F.softmax(atten_energies, dim=0)
        scores = scores.unsqueeze(2)

        context_vector = torch.sum(scores * out, dim=0)
        context_vector = context_vector.unsqueeze(0)

        out = torch.cat((h1, context_vector), 2)
        return self.layer3(out)


class AttackSeqMIA(AbstractMIA):
    """Implementation of the sequential MIA attack."""

    class AttackConfig(BaseModel):
        """Configuration for the sequential MIA attack."""

        distillation_data_fraction: float = \
            Field(default = 0.5, ge = 0.0, le=1.0, description="Fraction of auxiliary data used for distillation")  # noqa: E501
        train_mia_batch_size: int = Field(default=64, ge=1, description="Batch size for training the MIA classifier")
        number_of_traj: int = Field(default=1, ge=1, description="Number of trajectories to consider")
        mia_classifier_epochs: int = Field(default=100, ge=1, description="Number of epochs for training the MIA classifier")
        mia_classifier_lr: float = Field(default = 0.0001, ge = 0.0, description="Learning rate for training the MIA classifier")
        mia_classifier_momentum: float = \
            Field(default = 0.9, ge = 0.0, le = 1.0, description="Momentum for training the MIA classifier")
        mia_classifier_weight_decay: float = \
            Field(default = 0.0, ge = 0.0, le = 1.0, description="Weight decay for training the MIA classifier")
        attention_model: bool = Field(default=True, description="Whether to use an LSTM with or without attention")
        label_only: bool = Field(default=False, description="Whether to use only the labels for the attack")
        temperature: float = Field(default=2.0, ge=0.0, description="Temperature for the softmax")
        input_size: int = Field(default=5, ge=1, description="Input size for the LSTM model")

    def __init__(self: Self,
                 handler: MIAHandler,
                 configs: dict
                 ) -> None:
        """Initialize the AttackSeqMIA class.

        Args:
        ----
            handler (MIAHandler): The input handler object.
            configs (dict): A dictionary containing the attack loss_traj configurations.

        """
        logger.info("Configuring SeqMIA attack")
        self.configs = self.AttackConfig() if configs is None else self.AttackConfig(**configs)

        super().__init__(handler)

        for key, value in self.configs.model_dump().items():
            setattr(self, key, value)

        if self.population_size == self.audit_size:
            raise ValueError("The audit dataset is the same size as the population dataset. \
                    There is no data left for the shadow and distillation models.")

        self.num_shadow_models = 1
        self.shadow_data_fraction = 1 - self.distillation_data_fraction

        output_dir = self.handler.configs.audit.output_dir
        self.storage_dir = f"{output_dir}/attack_objects/loss_traj"

        # set up mia classifier
        self.read_from_file = False
        self.signal = ModelLogits()
        self.mia_train_data_loader = None
        self.mia_test_data_loader = None
        self.dim_in_mia = (self.number_of_traj + 1 )
        if self.attention_model:
            self.mia_model = LSTMAttention(self.input_size)
        else:
            self.mia_model = LSTM(self.input_size)

    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "SeqMIA: Sequential-Metric Based Membership Inference Attack"
        reference_str = "Hao Li, Zheng Li, Siyuan Wu, Chengrui Hu, \
            Yutong Ye, Min Zhang, Dengguo Feng, and Yang Zhang \
            SeqMIA: Sequential-Metric Based Membership Inference Attack. (2024)."
        summary_str = ""
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
        logger.info("Preparing the data for seqMIA attack")

        aux_data_index = self.sample_indices_from_population(include_train_indices = False, include_test_indices = False)

        aux_data_size = len(aux_data_index)
        shadow_data_size = int(aux_data_size * self.shadow_data_fraction)
        shadow_data_indices = np.random.choice(aux_data_index, shadow_data_size, replace=False)
        split_index = len(shadow_data_indices) // 2
        shadow_training_indices = shadow_data_indices[:split_index]
        distill_data_indices = np.setdiff1d(aux_data_index, shadow_data_indices)

        logger.info(f"Training shadow models on {len(shadow_training_indices)} points")
        self.shadow_model_indices = ShadowModelHandler().create_shadow_models(self.num_shadow_models,
                                                                              shadow_training_indices,
                                                                              training_fraction = 0.99)

        self.shadow_model, _ = ShadowModelHandler().get_shadow_models(self.shadow_model_indices)

        DistillationModelHandler().add_student_teacher_pair("shadow_distillation", self.shadow_model[0].model_obj)
        DistillationModelHandler().add_student_teacher_pair("target_distillation", self.target_model.model_obj)

        logger.info(f"Training distillation of the shadow model on {len(distill_data_indices)} points")
        self.distill_shadow_models = DistillationModelHandler().distill_model("shadow_distillation",
                                                                                self.number_of_traj,
                                                                                distill_data_indices,
                                                                                self.label_only)

        self.distill_target_models = DistillationModelHandler().distill_model("target_distillation",
                                                                                self.number_of_traj,
                                                                                distill_data_indices,
                                                                                self.label_only)

        train_mask = np.isin(shadow_data_indices,shadow_training_indices)
        self.prepare_mia_data(shadow_data_indices,
                              train_mask,
                              student_model = self.distill_shadow_models,
                              teacher_model = self.shadow_model[0].model_obj,
                              train_mode = True)

        mia_test_data_indices = self.audit_dataset["data"]
        train_indices = mia_test_data_indices[self.audit_dataset["in_members"]]
        test_mask = np.isin(mia_test_data_indices, train_indices)
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

        dataset_name = "seqmia_train_data.pkl" if train_mode else "seqmia_test_data.pkl"
        if os.path.exists(f"{self.storage_dir}/{dataset_name}"):
            logger.info(f"Loading MIA {dataset_name}: {len(data_indices)} points")
            with open(f"{self.storage_dir}/{dataset_name}", "rb") as file:
                data = pickle.load(file)  # noqa: S301
        else:
            data = self._prepare_mia_data(data_indices,
                                          student_model,
                                          teacher_model,
                                          dataset_name)
        data["member_status"] = membership_status_shadow_train

        os.makedirs(self.storage_dir, exist_ok=True)
        with open(f"{self.storage_dir}/{dataset_name}", "wb") as file:
            pickle.dump(data, file)

        mia_input = data["model_trajectory"]
        mia_dataset = TensorDataset(torch.tensor(mia_input), torch.tensor(data["member_status"]))
        if train_mode:
            self.mia_train_data_loader = DataLoader(mia_dataset, batch_size=self.train_mia_batch_size, shuffle=True)
        else:
            self.mia_test_data_loader = DataLoader(mia_dataset, batch_size=self.train_mia_batch_size, shuffle=True)

    def _prepare_mia_data(self:Self,
                        data_indices: np.ndarray,
                        distill_model: nn.Module,
                        teacher_model: nn.Module,
                        dataset_name: str,
                        ) -> dict:
        logger.info(f"Preparing MIA {dataset_name}: {len(data_indices)} points")
        gpu_or_cpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_loader = self.handler.get_dataloader(data_indices, batch_size=self.train_mia_batch_size, shuffle = False)

        model_trajectory_list = []

        for _loader_idx, (data, target) in tqdm(enumerate(data_loader),
                                               total=len(data_loader),
                                               desc=f"Preparing MIA data {dataset_name}"):
            data = data.to(gpu_or_cpu)
            target = target.to(gpu_or_cpu)

            criterion = DistillationModelHandler().get_criterion()
            trajectory_steps = []

            for d in range(self.number_of_traj) :
                distill_model[d].to(gpu_or_cpu)
                distill_model[d].eval()

                distill_model_soft_output = distill_model[d](data).squeeze()

                loss = []
                for logit_target_i, target_i in zip(distill_model_soft_output, target):
                    p_target_i = torch.exp(logit_target_i - torch.max(logit_target_i))
                    p_target_i = p_target_i / torch.sum(p_target_i)
                    entropy = -torch.sum(p_target_i * torch.log(torch.clamp(p_target_i, min=1e-10)))
                    p_except_target = torch.cat((p_target_i[:target_i],p_target_i[(target_i+1):]))
                    mentropy = \
                        -(1 - p_target_i[target_i]) * torch.log(torch.clamp(p_target_i[target_i],min=1e-10)) - \
                        torch.sum(p_except_target * torch.log(torch.clamp((1 - p_except_target),min=1e-10)))
                    loss_i = torch.stack([
                        criterion(logit_target_i, target_i),
                        torch.max(p_target_i),
                        torch.std(p_target_i),
                        entropy,
                        mentropy
                        ])
                    loss.append(loss_i)
                loss = torch.stack(loss).detach().cpu().numpy()
                trajectory_steps.append(loss[:, np.newaxis, :])

            teacher_model.to(gpu_or_cpu)
            teacher_model.eval()
            model_soft_output = teacher_model(data).squeeze()

            loss = []
            for logit_target_i, target_i in zip(model_soft_output, target):
                p_target_i = torch.exp(logit_target_i - torch.max(logit_target_i))
                p_target_i = p_target_i / torch.sum(p_target_i)
                entropy = -torch.sum(p_target_i * torch.log(torch.clamp(p_target_i, min=1e-10)))
                p_except_target = torch.cat((p_target_i[:target_i],p_target_i[(target_i+1):]))
                mentropy = \
                    -(1 - p_target_i[target_i]) * torch.log(torch.clamp(p_target_i[target_i],min=1e-10)) - \
                    torch.sum(p_except_target * torch.log(torch.clamp((1 - p_except_target),min=1e-10)))
                loss_i = torch.stack([
                    criterion(logit_target_i, target_i),
                    torch.max(p_target_i),
                    torch.std(p_target_i),
                    entropy,
                    mentropy
                    ])
                loss.append(loss_i)
            loss = torch.stack(loss).detach().cpu().numpy()

            trajectory_steps.append(loss[:, np.newaxis, :])
            trajectory_current = np.concatenate(trajectory_steps, 1)
            model_trajectory_list.append(trajectory_current)

        model_trajectory = np.concatenate(model_trajectory_list, axis=0) if model_trajectory_list else np.array([])

        return {"model_trajectory": model_trajectory}

    # --- FIX: Renamed function to train_seqmia_classifier to avoid collision ---
    def train_seqmia_classifier(self:Self)-> nn.Module:
        """"Trains and returns the MIA (Membership Inference Attack) classifier.

        Returns
        -------
            attack_model (torch.nn.Module): The trained MIA classifier model.

        """
        attack_model = self.mia_model

        if not os.path.exists(f"{self.storage_dir}/seqmia_model.pkl"):
            gpu_or_cpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            attack_optimizer = torch.optim.SGD(attack_model.parameters(),
                                               lr=self.mia_classifier_lr,
                                               momentum=self.mia_classifier_momentum,
                                               weight_decay=self.mia_classifier_weight_decay)
            attack_model = attack_model.to(gpu_or_cpu)
            loss_fn = nn.CrossEntropyLoss()

            train_loss, train_prec1 = self._train_mia_classifier(attack_model, attack_optimizer, loss_fn)
            train_info = [train_loss, train_prec1]

            torch.save(attack_model.state_dict(), self.storage_dir + "/seqmia_model.pkl")

            with open(f"{self.storage_dir}/mia_model_losses.pkl", "wb") as file:
                pickle.dump(train_info, file)
        else:
            with open(f"{self.storage_dir}/seqmia_model.pkl", "rb") as model:
                attack_model.load_state_dict(torch.load(model))
            logger.info("Loading SeqMIA classifier")

        return attack_model

    def _train_mia_classifier(self:Self, model:nn.Module,
                        attack_optimizer:torch.optim.Optimizer,
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
        logger.info("Training the MIA classifier")

        train_loss_list =[]
        train_prec_list = []

        model.train()
        train_loss = 0
        num_correct = 0
        mia_train_loader = self.mia_train_data_loader
        gpu_or_cpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for _ in  tqdm(range(self.mia_classifier_epochs), total=self.mia_classifier_epochs):
            for _batch_idx, (data, label) in enumerate(mia_train_loader):
                data = data.to(gpu_or_cpu) # noqa: PLW2901
                label = label.to(gpu_or_cpu).long() # noqa: PLW2901

                pred = model(torch.permute(data,(1,0,2)))
                pred = torch.squeeze(pred)
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
            - auc_pred: The predicted membership probabilities for each data point.

        """
        logger.info("Running the MIA attack")
        gpu_or_cpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        attack_model.to(gpu_or_cpu)
        attack_model.eval()
        test_loss = 0
        correct = 0

        auc_ground_truth_list = []
        auc_pred_list = []

        with torch.no_grad():
            for _batch_idx, (data, target) in tqdm(enumerate(self.mia_test_data_loader), total=len(self.mia_test_data_loader)):
                data = data.to(gpu_or_cpu) # noqa: PLW2901
                target = target.to(gpu_or_cpu) # noqa: PLW2901
                target = target.long() # noqa: PLW2901
                pred = attack_model(torch.permute(data,(1,0,2)))
                pred = torch.squeeze(pred)

                test_loss += F.cross_entropy(pred, target).item()
                pred0, pred1 = pred.max(1, keepdim=True)
                correct += pred1.eq(target).sum().item()
                auc_pred_current = pred[:, -1]

                auc_ground_truth_list.append(target.cpu().numpy())
                auc_pred_list.append(auc_pred_current.cpu().detach().numpy())

        if auc_ground_truth_list:
            auc_ground_truth = np.concatenate(auc_ground_truth_list, axis=0)
            auc_pred = np.concatenate(auc_pred_list, axis=0)
        else:
            auc_ground_truth = np.array([])
            auc_pred = np.array([])

        test_loss /= len(self.mia_test_data_loader.dataset)

        return auc_ground_truth, auc_pred

    def run_attack(self:Self) -> MIAResult:
        """Run the attack and return the combined metric result.

        Returns
        -------
            MIAResult: The result of the attack.

        """
        self.train_seqmia_classifier()
        true_labels, signals = self.mia_attack(self.mia_model)

        return MIAResult.from_full_scores(true_membership=true_labels,
                                          signal_values=signals,
                                          result_name="SeqMIA",
                                          metadata=self.configs.model_dump())
