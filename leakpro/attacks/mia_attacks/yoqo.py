"""Implementation of the LiRA attack."""

import numpy as np
from tqdm import tqdm

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.metrics.attack_result import CombinedMetricResult
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger

import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

class AttackYOQO(AbstractMIA):
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
        if self.output_shape == 1:
            self.loss = BCEWithLogitsLoss(reduction = 'none')
            #self.loss = BCEWithLogitsLoss()
        else: 
            self.loss = CrossEntropyLoss(reduction = 'none')
            #self.loss = CrossEntropyLoss()

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

        # YOQO specific
        self.alpha = configs.get("alpha", 2)
        self.n_audits = configs.get("n_audits", -1)
        self.lr_xprime_optimization = configs.get("lr_xprime_optimization", 1e-3)
        self.stop_criterion = configs.get("stop_criterion", self.num_shadow_models / 8)
        self.max_iterations = configs.get("max_iterations", 30)

        # Define the validation dictionary as: {parameter_name: (parameter, min_value, max_value)}
        validation_dict = {
            "num_shadow_models": (self.num_shadow_models, 1, None),
            "training_data_fraction": (self.training_data_fraction, 0, 1),
            "lr_xprime_optimization": (self.lr_xprime_optimization, 0, None),
            #"max_iterations": (self.max_iterations, 1, None),
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

        self.attack_data_indices = self.sample_indices_from_population(include_train_indices = self.online,
                                                                       include_test_indices = self.online)

        self.shadow_model_indices = ShadowModelHandler().create_shadow_models(num_models = self.num_shadow_models,
                                                                              shadow_population =  self.attack_data_indices,
                                                                              training_fraction = self.training_data_fraction,
                                                                              online = self.online)

        self.shadow_models, _ = ShadowModelHandler().get_shadow_models(self.shadow_model_indices)

        logger.info("Create masks for all IN and OUT samples")
        self.in_indices_masks = ShadowModelHandler().get_in_indices_mask(self.shadow_model_indices, self.audit_dataset["data"])

        if self.online:
            # Exclude all audit points that have either no IN or OUT samples
            num_shadow_models_seen_points = np.sum(self.in_indices_masks, axis=1)
            mask = (num_shadow_models_seen_points > 0) & (num_shadow_models_seen_points < self.num_shadow_models)

            # Filter the audit data
            self.audit_data_indices = self.audit_dataset["data"][mask]
            self.in_indices_masks = self.in_indices_masks[mask, :]

            # Filter IN and OUT members
            self.in_members = np.arange(np.sum(mask[self.audit_dataset["in_members"]]))
            num_out_members = np.sum(mask[self.audit_dataset["out_members"]])
            self.out_members = np.arange(len(self.in_members), len(self.in_members) + num_out_members)

            assert len(self.audit_data_indices) == len(self.in_members) + len(self.out_members)

            if len(self.audit_data_indices) == 0:
                raise ValueError("No points in the audit dataset are used for the shadow models")

            if self.output_shape == 1:
                self._optimization_objective = self._optimization_objective_online_binary
            else:
                self._optimization_objective = self._optimization_objective_online
        else:
            self.audit_data_indices = self.audit_dataset["data"]
            self.in_members = self.audit_dataset["in_members"]
            self.out_members = self.audit_dataset["out_members"]

            self.mse = MSELoss()

            if self.output_shape == 1:
                self._optimization_objective = self._optimization_objective_offline_binary
            else:
                self._optimization_objective = self._optimization_objective_offline

        # Check offline attack for possible IN- sample(s)
        if not self.online:
            count_in_samples = np.count_nonzero(self.in_indices_masks)
            if count_in_samples > 0:
                logger.info(f"Some shadow model(s) contains {count_in_samples} IN samples in total for the model(s)")
                logger.info("This is not an offline attack!")

        # Select a subset of the data to audit
        if self.n_audits < 0:
            self.n_audits = len(self.audit_data_indices)
        else:
            self.in_members = self.in_members[:(self.n_audits//2)]
            self.out_members = self.out_members[:(self.n_audits - len(self.in_members))]
            self.audit_data_indices = np.concatenate(
                (
                    self.audit_data_indices[self.in_members], 
                    self.audit_data_indices[self.out_members]
                )
            )
            self.in_indices_masks = np.concatenate(
                (
                    self.in_indices_masks[self.in_members,:],
                    self.in_indices_masks[self.out_members,:]
                ),
                axis = 0
            )

        self.weights = 1 + (self.alpha - 1) * self.in_indices_masks
        self.weights = Tensor(self.weights)
        self.batch_size = 5000

        data_loader = self.handler.get_dataloader(self.audit_data_indices, batch_size=self.batch_size)
        self.target_output = []
        for i, (data, labels) in tqdm(enumerate(data_loader), desc=f"Calculating target labels", leave=False):
            if self.output_shape == 1:
                batch_target_output = labels[:,np.newaxis] * self.in_indices_masks[(i * self.batch_size):((i + 1) * self.batch_size),:] + (1 - labels[:,np.newaxis]) * (1 - self.in_indices_masks[(i * self.batch_size):((i + 1) * self.batch_size),:])
                self.target_output.extend(batch_target_output)
            else: 
                labels_true = labels
                labels_false = torch.stack([shadow_model.model_obj(data) for shadow_model in self.shadow_models])
                labels_false[:,np.arange(labels_true.shape[0]),labels_true] = -1e10
                labels_false = torch.argmax(labels_false, dim = 2)
                batch_target_output = labels_true[:,np.newaxis] * self.in_indices_masks[(i * self.batch_size):((i + 1) * self.batch_size),:] + \
                    labels_false.T * (1 - self.in_indices_masks[(i * self.batch_size):((i + 1) * self.batch_size),:])
                self.target_output.extend(batch_target_output)
        self.target_output = np.array(self.target_output)
        self.target_output = Tensor(self.target_output)

    def _optimization_objective_online(self: Self, x0: Tensor, dx: Tensor, target_output: Tensor, weights: Tensor):
        loss = self.loss(torch.stack([shadow_model.model_obj(x0 + dx) for shadow_model in self.shadow_models], dim = 2), target_output)
        loss = (loss * weights).sum()
        return loss

    def _optimization_objective_online_binary(self: Self, x0: Tensor, dx: Tensor, target_output: Tensor, weights: Tensor):
        loss = self.loss(torch.cat([shadow_model.model_obj(x0 + dx) for shadow_model in self.shadow_models], dim = 1), target_output)
        loss = (loss * weights).sum()
        return loss

    def _optimization_objective_offline(self: Self, x0: Tensor, dx: Tensor, target_output: Tensor, weights: Tensor):
        loss = self.loss(torch.stack([shadow_model.model_obj(x0 + dx) for shadow_model in self.shadow_models], dim = 2),target_output) 
        loss = (loss * weights).sum()
        loss = loss + self.alpha * self.mse(dx, torch.zeros(dx.shape, device = dx.device))
        return loss

    def _optimization_objective_offline_binary(self: Self, x0: Tensor, dx: Tensor, target_output: Tensor, weights: Tensor):
        loss = self.loss(torch.cat([shadow_model.model_obj(x0 + dx) for shadow_model in self.shadow_models], dim = 1),target_output)
        loss = (loss * weights).sum()
        loss = loss + self.alpha * self.mse(dx, torch.zeros(dx.shape, device = dx.device))
        return loss

    def _optimize_xprime(self: Self, x0: Tensor, target_output: Tensor, weights: Tensor) -> Tensor:
        dx = torch.zeros(x0.shape, device = x0.device)
        dx.requires_grad = True

        optim = torch.optim.SGD([dx], lr = self.lr_xprime_optimization)
        for i in range(self.max_iterations):
            loss = self._optimization_objective(x0, dx, target_output, weights)
            if loss < self.stop_criterion:
                break
            loss.backward()
            optim.step()
            optim.zero_grad()

        return (x0 + dx)

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

        device_name = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

        for model in self.shadow_models:
            model.model_obj.eval()
            model.model_obj.to(device_name)

        for i, (data, labels) in tqdm(enumerate(data_loader), desc=f"Optimizing queries", leave=False):
            xprime = self._optimize_xprime(data.to(device_name), 
                                           self.target_output[i:(i + 1)].to(device_name),
                                           self.weights[i:(i + 1), :].to(device_name))

            lprime = self.target_model.get_logits(xprime)
            lprime = np.argmax(lprime, axis = 1)

            batch_predictions = np.equal(lprime,labels)
            predictions.extend(batch_predictions)

        predictions = np.array(predictions).reshape(1,-1)
        signal_values = predictions.copy().reshape(-1,1)

        for model in self.shadow_models:
            model.model_obj.train()

        # Prepare true labels array, marking 1 for training data and 0 for non-training data
        true_labels = np.concatenate(
            [np.ones(len(self.in_members)), np.zeros(len(self.out_members))]
        )

        logger.info(f"Accuracy: {np.sum(predictions == true_labels)/predictions.size}")

        predictions = np.concatenate(
            [np.zeros((1,predictions.size)), predictions, np.ones((1,predictions.size))], axis = 0
        )

        # Return a result object containing predictions, true labels, and the signal values for further evaluation
        return CombinedMetricResult(
            predicted_labels=predictions,
            true_labels=true_labels,
            predictions_proba=None,  # Note: Direct probability predictions are not computed here
            signal_values=signal_values,
        )
