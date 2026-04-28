#
# Copyright 2023-2026 AI Sweden
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Implementation of the OSLO attack."""

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from tqdm import tqdm

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class AttackOSLO(AbstractMIA):
    """Implementation of the One-Shot Label-Only Membership Inference Attacks attack."""

    class AttackConfig(BaseModel):
        """Configuration for the OSLO attack."""

        training_data_fraction: float = Field(default=0.5, ge=0.0, le=1.0, description="Fraction of auxilary dataset to use for each shadow model training")  # noqa: E501
        online: bool = Field(default=False, description="Perform online or offline attack")
        num_source_models: int = Field(default=9, ge=1, description="Number of source shadow models to train")
        num_validation_models: int = Field(default=3, ge=1, description="Number of validation shadow models to train")
        num_sub_procedures: int = Field(default=80, ge=1, description='Number of iterations of the "subprocedure"')
        num_iterations: int = Field(default=5, ge=1, description='Number of iterations in each "subprocedure"')
        step_size: float = Field(default=1e-2, ge=0.0, description="Step size for optimization of xprime")
        max_perturbation_size: float = Field(default=80/255, ge=0.0, description="Maximum distance between x and xprime")
        min_threshold: float = Field(default=1e-4, gt=0.0, description="Minimum threshold for the early stopping criterion")
        max_threshold: float = Field(default=1, ge=0.0, description="Maximum threshold for the early stopping criterion")
        n_thresholds: int = Field(default=5, ge=1, description="Number of thresholds to use for the early stopping criterion")
        n_audits: int = Field(default=500, ge=1, description="Number of data points to audit")

    def __init__(self:Self,
                 handler: AbstractInputHandler,
                 configs: dict
                 ) -> None:
        """Initialize the OSLO attack.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.
            configs (dict): Configuration parameters for the attack.

        """
        logger.info("Configuring OSLO attack")
        self.configs = self.AttackConfig() if configs is None else self.AttackConfig(**configs)
        super().__init__(handler)

        # Assign the configuration parameters to the object
        for key, value in self.configs.model_dump().items():
            setattr(self, key, value)

        if self.online is False and self.population_size == self.audit_size:
            raise ValueError("The audit dataset is the same size as the population dataset. \
                    There is no data left for the shadow models.")

        tmp = self.handler.get_dataloader([0], batch_size=1)
        tmp_features, _ = next(iter(tmp))
        logits = self.target_model.get_logits(tmp_features)
        self.output_shape = logits.shape[1]
        self.binary_output = (self.output_shape == 1)
        if self.binary_output:
            self.loss = BCEWithLogitsLoss(reduction = "none")
        else:
            self.loss = CrossEntropyLoss(reduction = "none")

    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "One-Shot Label-Only Membership Inference Attacks (OSLO) attack"

        reference_str = "Peng Y, et al. OSLO: One-Shot Label-Only Membership Inference Attacks"

        summary_str = "OSLO is a membership inference attack based on the predicted hard labels of a black-box model"

        detailed_str = "The attack is executed according to: \
            1. Train surrogate (source and validation) models using \
                an auxiliary data set. \
            2. For data points not used in the surrogate model training, \
                generate an adversarial example based on a minimal perturbation.\
            3. If the same perturbation is insufficient to ``fool'' the target \
                model, the null hypothesis (data point not in training data) is rejected."

        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }

    def prepare_attack(self:Self)->None:
        """Prepares data to obtain metric on the target model and dataset, using signals computed on the auxiliary model/dataset.

        It selects a balanced subset of data samples from in-group and out-group members
        of the audit dataset, prepares the data for evaluation, and computes the logits
        for both shadow models and the target model.
        """

        self.thresholds = np.linspace(np.log(self.max_threshold), np.log(self.min_threshold), self.n_thresholds)
        logger.info(f"thresholds: {self.thresholds}")

        self.attack_data_indices = self.sample_indices_from_population(include_train_indices = True,
                                                                       include_test_indices = True)

        if self.n_audits > len(self.attack_data_indices):
            raise ValueError(f"n_audits ({self.n_audits}) exceeds available attack data ({len(self.attack_data_indices)})")

        self.audit_data_indices = np.random.choice(self.attack_data_indices, self.n_audits, replace=False)
        self.attack_data_indices = np.setdiff1d(self.attack_data_indices, self.audit_data_indices)

        self.shadow_model_indices = ShadowModelHandler().create_shadow_models(
            num_models = self.num_source_models + self.num_validation_models,
            shadow_population =  self.attack_data_indices,
            training_fraction = self.training_data_fraction,
            online = self.online
        )

        self.shadow_models, _ = ShadowModelHandler().get_shadow_models(self.shadow_model_indices)
        self.source_models, _ = ShadowModelHandler().get_shadow_models(self.shadow_model_indices[:self.num_source_models])
        self.validation_models, _ = ShadowModelHandler().get_shadow_models(self.shadow_model_indices[self.num_source_models:])

    def _compute_loss(
        self: Self,
        x0: Tensor,
        dx: Tensor,
        y: Tensor,
        models: list
    ) -> float:
        if self.binary_output:
            loss = -torch.cat([self.loss(m.model_obj(x0 + dx)[0], y) for m in models], dim=0)
        else:
            loss = -torch.cat([self.loss(m.model_obj(x0 + dx), y) for m in models], dim=0)
        return loss.sum()

    def _optimization_objective(
        self: Self,
        x0: Tensor,
        dx: Tensor,
        y: Tensor
    ) -> float:
        return self._compute_loss(x0, dx, y, self.source_models)

    def _stop_criterion(
        self: Self,
        x0: Tensor,
        dx: Tensor,
        y: Tensor
    ) -> float:
        return self._compute_loss(x0, dx, y, self.validation_models)

    def _generate_adversarial_example(
        self: Self,
        x0: Tensor,
        y: Tensor
    ) -> Tensor:
        dx = torch.zeros(x0.shape, device = x0.device)
        x = torch.zeros((self.n_thresholds,) + tuple(x0.shape), device = x0.device)
        dx.requires_grad = True

        optim = torch.optim.SGD([dx], lr = self.step_size)
        i = 0
        for k in range(1, self.num_sub_procedures + 1):
            for _ in range(self.num_iterations):
                optim.zero_grad()
                loss = self._optimization_objective(x0, dx, y)
                loss.backward()
                optim.step()

                with torch.no_grad():
                    norm = k * self.max_perturbation_size / self.num_sub_procedures / (torch.linalg.vector_norm(dx) + 1e-30)
                    dx[:] = dx * torch.clamp(norm, max = 1)
                while i < self.n_thresholds and self._stop_criterion(x0, dx, y) < self.thresholds[i]:
                    x[i,:] = x0 + dx
                    i += 1
                if i >= self.n_thresholds:
                    return x
        while i < self.n_thresholds:
            x[i,:] = x0 + dx
            i += 1

        return x

    def run_attack(self:Self) -> MIAResult:
        """Runs the attack on the target model and dataset and assess privacy risks or data leakage.

        This method evaluates how the target model's output (logits) for a specific dataset
        compares to the output of shadow models to determine if the dataset was part of the
        model's training data or not.

        Returns
        -------
        Result(s) of the metric. An object containing the metric results, including predictions,
        true labels, and signal values.

        """

        data_loader = self.handler.get_dataloader(self.audit_data_indices, batch_size=1, shuffle=False)

        predictions = []

        in_members = set(self.audit_dataset["data"][self.audit_dataset["in_members"]].tolist())
        true_labels = np.array([i in in_members for i in self.audit_data_indices])

        device_name = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.target_model.model_obj.to(device_name)
        self.target_model.model_obj.eval()
        for model in self.source_models + self.validation_models:
            model.model_obj.eval()
            model.model_obj.to(device_name)
            model.model_obj.requires_grad_(False)

        for _, (data, labels) in tqdm(enumerate(data_loader),
                                      total = len(data_loader),
                                      desc="Optimizing queries",
                                      leave=False):
            xprime = self._generate_adversarial_example(data.to(device_name), labels.to(device_name))

            lprime = np.array([self.target_model.get_logits(xprime[j]) for j in range(xprime.shape[0])])
            lprime = 1 * (lprime > 0) if self.binary_output else np.argmax(lprime, axis = 2)

            batch_predictions = np.equal(lprime, labels.numpy())
            predictions.append(batch_predictions)

        predictions = np.array(predictions).reshape(-1, self.n_thresholds)
        signal_values = predictions[:, -1].astype(float).reshape(-1, 1)

        logger.info(f"Accuracy: {np.sum(predictions == true_labels[:,np.newaxis], axis=0)/predictions.shape[0]}")
        logger.info(f"TPR: {np.sum(predictions * true_labels[:,np.newaxis], axis=0)/np.sum(true_labels)}")
        logger.info(f"FPR: {np.sum(predictions * (1 - true_labels[:,np.newaxis]), axis=0)/np.sum(1 - true_labels)}")

        return MIAResult.from_full_scores(
            true_membership=true_labels,
            signal_values=signal_values,
            result_name="OSLO",
            metadata=self.configs.model_dump(),
        )
