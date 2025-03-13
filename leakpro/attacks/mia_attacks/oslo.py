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
from leakpro.metrics.attack_result import MIAResult
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
        step_size: float = Field(default=1e-2, ge=0.0, description="Learning rate for optimization of xprime")
        max_perturbation_size: float = Field(default=80/255, ge=0.0, description="Maximum distance between x and xprime")
        min_threshold: float = Field(default=1e-4, ge=0.0, description="Minimum threshold for the early stopping criterion")
        max_threshold: float = Field(default=1, ge=0.0, description="Maximum threshold for the early stopping criterion")
        n_thresholds: float = Field(default=5, ge=0.0, description="Number of thresholds to use for the early stopping criterion")
        n_audits: int = Field(default=500, ge=1, description="Number of data points to audit")

        lr_xprime_optimization: float = Field(default=1e-3, ge=0.0, description="Learning rate for optimization of xprime")
        max_iterations: int = Field(default=35, ge=1, description="Maximum number of iterations for optimization of xprime")

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
        self.configs = self.AttackConfig() if configs is None else self.AttackConfig(**configs)
        super().__init__(handler)

        # Assign the configuration parameters to the object
        for key, value in self.configs.model_dump().items():
            setattr(self, key, value)

        if self.online is False and self.population_size == self.audit_size:
            raise ValueError("The audit dataset is the same size as the population dataset. \
                    There is no data left for the shadow models.")

        tmp = self.handler.get_dataloader(0, batch_size=1)
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

        self.audit_data_indices = self.attack_data_indices[:self.n_audits]
        self.attack_data_indices = self.attack_data_indices[self.n_audits:]

        self.shadow_model_indices = ShadowModelHandler().create_shadow_models(
            num_models = self.num_source_models + self.num_validation_models,
            shadow_population =  self.attack_data_indices,
            training_fraction = self.training_data_fraction,
            online = True
        )

        self.shadow_models, _ = ShadowModelHandler().get_shadow_models(self.shadow_model_indices)
        self.source_models, _ = ShadowModelHandler().get_shadow_models(self.shadow_model_indices[:self.num_source_models])
        self.validation_models, _ = ShadowModelHandler().get_shadow_models(self.shadow_model_indices[self.num_source_models:])

    def _optimization_objective(
        self: Self,
        x0: Tensor,
        dx: Tensor,
        y: Tensor
    ) -> float:
        if self.binary_output:
            loss = -torch.cat([self.loss(shadow_model.model_obj(x0 + dx)[0],y) for shadow_model in self.source_models], dim = 0)
        else:
            loss = -torch.cat([self.loss(shadow_model.model_obj(x0 + dx),y) for shadow_model in self.source_models], dim = 0)
        return loss.sum()

    def _stop_criterion(
        self: Self,
        x0: Tensor,
        dx: Tensor,
        y: Tensor
    ) -> float:
        if self.binary_output:
            loss = -torch.cat(
                [self.loss(shadow_model.model_obj(x0 + dx)[0],y) for shadow_model in self.validation_models], dim = 0
            )
        else:
            loss = -torch.cat([self.loss(shadow_model.model_obj(x0 + dx),y) for shadow_model in self.validation_models], dim = 0)
        return loss.sum()

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

        data_loader = self.handler.get_dataloader(self.audit_data_indices, batch_size=1)

        predictions = []
        signal_values = []

        in_members = self.audit_dataset["data"][self.audit_dataset["in_members"]]
        true_labels = np.array([i in in_members for i in self.audit_data_indices])

        device_name = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for model in self.source_models:
            model.model_obj.eval()
            model.model_obj.to(device_name)
        for model in self.validation_models:
            model.model_obj.eval()
            model.model_obj.to(device_name)

        for _, (data, labels) in tqdm(enumerate(data_loader),
                                      total = len(data_loader),
                                      desc="Optimizing queries",
                                      leave=False):
            xprime = self._generate_adversarial_example(data.to(device_name), labels)

            lprime = np.array([self.target_model.get_logits(xprime[j]) for j in range(xprime.shape[0])])
            lprime = 1 * (lprime > 0) if self.binary_output else np.argmax(lprime, axis = 2)

            batch_predictions = np.equal(lprime,labels)
            predictions.append(batch_predictions)

        predictions = np.array(predictions).reshape(-1,self.n_thresholds)
        signal_values = predictions[:,-1].copy().reshape(-1,1)

        # Prepare true labels array, marking 1 for training data and 0 for non-training data
        in_members = self.audit_dataset["data"][self.audit_dataset["in_members"]]
        true_labels = np.array([i in in_members for i in self.audit_data_indices])

        logger.info(f"Accuracy: {np.sum(predictions == true_labels[:,np.newaxis], axis = 0)/predictions.shape[0]}")
        logger.info(f"TPR: {np.sum(predictions * true_labels[:,np.newaxis], axis = 0)/np.sum(true_labels)}")
        logger.info(f"FPR: {np.sum(predictions * (1 - true_labels[:,np.newaxis]), axis = 0)/np.sum(1 - true_labels)}")

        # Output in a format that can be used to generate ROC curve.
        predictions = predictions.T

        # Return a result object containing predictions, true labels, and the signal values for further evaluation
        return MIAResult(
            predicted_labels=predictions,
            true_labels=true_labels,
            predictions_proba=None,  # Note: Direct probability predictions are not computed here
            signal_values=signal_values,
        )
