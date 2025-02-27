"""Implementation of the OSLO attack."""

import numpy as np
import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from tqdm import tqdm

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.metrics.attack_result import CombinedMetricResult
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class AttackOSLO(AbstractMIA):
    """Implementation of the You Only Query Once attack."""

    def __init__(self:Self,
                 handler: AbstractInputHandler,
                 configs: dict
                 ) -> None:
        """Initialize the YOQO attack.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.
            configs (dict): Configuration parameters for the attack.

        """
        # Initializes the parent metric
        super().__init__(handler)

        tmp = self.handler.get_dataloader(0, batch_size=1)
        tmp_features, _ = next(iter(tmp))
        logits = self.target_model.get_logits(tmp_features)
        self.output_shape = logits.shape[1]
        self.binary_output = (self.output_shape == 1)
        if self.binary_output:
            self.loss = BCEWithLogitsLoss(reduction = "none")
        else:
            self.loss = CrossEntropyLoss(reduction = "none")

        self._configure_attack(configs)

    def _configure_attack(self:Self, configs: dict) -> None:
        """Configure the YOQO attack.

        Args:
        ----
            configs (dict): Configuration parameters for the attack.

        """
        self.shadow_models = []
        self.num_shadow_models = configs.get("num_shadow_models", 64)
        self.online = configs.get("online", False)
        self.training_data_fraction = configs.get("training_data_fraction", 0.5)

        # OSLO specific
        self.num_source_models = configs.get("num_source_models", 3)
        self.num_validation_models = configs.get("num_validation_models", 9)

        self.num_sub_procedures = configs.get("num_sub_procedures", 80)
        self.num_iterations = configs.get("num_iterations", 5)
        self.step_size = configs.get("step_size", 0.01)
        self.max_perturbation_size = configs.get("max_perturbation_size", 80/255)
        self.min_threshold = configs.get("min_threshold", 0.0001)
        self.max_threshold = configs.get("max_threshold", 1)
        self.n_thresholds = configs.get("n_thresholds", 5)
        self.n_audits = configs.get("n_audits", -1)

        # Define the validation dictionary as: {parameter_name: (parameter, min_value, max_value)}
        validation_dict = {
            "num_shadow_models": (self.num_shadow_models, 1, None),
            "training_data_fraction": (self.training_data_fraction, 0, 1),
        }

        # Validate parameters
        for param_name, (param_value, min_val, max_val) in validation_dict.items():
            self._validate_config(param_name, param_value, min_val, max_val)

    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "You Only Query Once (YOQO) attack"

        reference_str = "Wu Y, et al. You Only Query Once: An Efficient Label- Only Membership Inference Attack"

        summary_str = "YOQO is a membership inference attack based on the predictecd hard labels of a black-box model"

        detailed_str = "The attack is executed according to: \
            1. A fraction of the target model dataset is sampled to be included (in-) or excluded (out-) \
            from the shadow model training dataset. \
            2. The shadow models are used to find perturbations to the datapoints such that the performance \
            difference between in- and out-models is maximized.\
            3. Membership status in the target model is determined using a single query (for each datapoint) \
            at the perturbed datapoint. \
            4. The attack is evaluated on an audit dataset to determine the attack performance."

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

        self.shadow_model_indices = ShadowModelHandler().create_shadow_models(num_models = self.num_source_models + self.num_validation_models,
                                                                              shadow_population =  self.attack_data_indices,
                                                                              training_fraction = self.training_data_fraction,
                                                                              online = True)

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
            loss = -torch.cat([self.loss(shadow_model.model_obj(x0 + dx),y) for shadow_model in self.source_models], dim = 0)
        else:
            loss = -torch.stack([self.loss(shadow_model.model_obj(x0 + dx),y) for shadow_model in self.source_models], dim = 2)
        return loss.sum()

    def _stop_criterion(
        self: Self,
        x0: Tensor,
        dx: Tensor,
        y: Tensor
    ) -> float:
        if self.binary_output:
            loss = -torch.cat([self.loss(shadow_model.model_obj(x0 + dx),y) for shadow_model in self.validation_models], dim = 0)
        else:
            loss = -torch.stack([self.loss(shadow_model.model_obj(x0 + dx),y) for shadow_model in self.validation_models], dim = 2)
        return loss.sum()

    def _generate_adversarial_example(
        self: Self,
        x0: Tensor,
        y: int
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
                    dx[:] = dx * torch.clamp(k * self.max_perturbation_size / self.num_sub_procedures / (torch.linalg.vector_norm(dx) + 1e-30), max = 1)
                while i < self.n_thresholds and self._stop_criterion(x0, dx, y) < self.thresholds[i]:
                    x[i,:] = x0 + dx
                    i += 1
                if i >= self.n_thresholds:
                    return x
        while i < self.n_thresholds:
            x[i,:] = x0 + dx
            i += 1

        return x

    def run_attack(self:Self) -> CombinedMetricResult:
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

        for i, (data, labels) in tqdm(enumerate(data_loader),
                                      total = len(data_loader),
                                      desc="Optimizing queries",
                                      leave=False):
            xprime = self._generate_adversarial_example(data[0].to(device_name), labels)

            lprime = self.target_model.get_logits(xprime)
            lprime = 1 * (lprime > 0)

            batch_predictions = np.equal(lprime,labels)
            predictions.append(batch_predictions)

        predictions = np.array(predictions).reshape(-1,self.n_thresholds)
        logger.info(f"predictions.shape: {predictions.shape}")
        signal_values = predictions.copy().reshape(-1,1)

        # Prepare true labels array, marking 1 for training data and 0 for non-training data
        in_members = self.audit_dataset["data"][self.audit_dataset["in_members"]]
        true_labels = np.array([i in in_members for i in self.audit_data_indices])

        logger.info(f"{predictions.shape}, {true_labels[:,np.newaxis].shape}, {np.sum(predictions, axis = 0)}, {np.sum(true_labels)}")
        logger.info(f"Accuracy: {np.sum(predictions == true_labels[:,np.newaxis], axis = 0)/predictions.shape[0]}")
        logger.info(f"TPR: {np.sum(predictions * true_labels[:,np.newaxis], axis = 0)/np.sum(true_labels)}")
        logger.info(f"FPR: {np.sum(predictions * (1 - true_labels[:,np.newaxis]), axis = 0)/np.sum(1 - true_labels)}")

        # Output in a format that can be used to generate ROC curve.
        predictions = np.concatenate(
            [np.zeros((1,predictions.shape[0])), predictions.T, np.ones((1,predictions.shape[0]))], axis = 0
        )

        # Return a result object containing predictions, true labels, and the signal values for further evaluation
        return CombinedMetricResult(
            predicted_labels=predictions,
            true_labels=true_labels,
            predictions_proba=None,  # Note: Direct probability predictions are not computed here
            signal_values=signal_values,
        )
