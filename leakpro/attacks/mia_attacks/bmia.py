"""Implementation of the BMIA attack."""

import numpy as np
import torch
from laplace import Laplace
from pydantic import BaseModel, Field
from scipy.special import logsumexp
from scipy.stats import t as t_dist

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.signals.signal import ModelLogits
from leakpro.signals.signal_extractor import PytorchModel
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class BMIAConfig(BaseModel):
    """Configuration for the BMIA attack."""

    n_samples: int = Field(default=50, description="Number of Laplace approximation samples")
    training_data_fraction: float = Field(default=0.5, ge=0.0, le=1.0, description="Part of available attack data to use for shadow model")  # noqa: E501


class AttackBMIA(AbstractMIA):
    """Implementation of the BMIA attack."""

    AttackConfig = BMIAConfig # required config for attack

    def __init__(self:Self,
                 handler: AbstractInputHandler,
                 configs: dict
                 ) -> None:
        """Initialize the BMIA attack.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.
            configs (dict): Configuration parameters for the attack.

        """
        logger.info("Configuring the BMIA attack")
        # Initializes the pydantic object using the user-provided configs
        # This will ensure that the user-provided configs are valid
        self.configs = BMIAConfig() if configs is None else BMIAConfig(**configs)

        # Call the parent class constructor. It will check the configs.
        super().__init__(handler)

        # Assign the configuration parameters to the object
        for key, value in self.configs.model_dump().items():
            setattr(self, key, value)

        self.signal = ModelLogits()
        self.epsilon = 1e-6
        self.shadow_model = None
        self.la = None


    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "BMIA attack"
        reference_str = "No reference"
        summary_str = "Bayesian Membership Inference Attack using Laplace Approximation."  # noqa: E501
        detailed_str = "The attack is executed according to: \
            1. Train a shadow model and fit a laplace approximation. \
            2. Use the Laplace approximation to sample final layer parameters. \
            3. Compute hinge loss values using the target model and sampled moedels from the Laplace approximation. \
            4. Compute the t-statistic and make predictions by extrapolation."
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }

    def hinge_loss(self:Self, pred:np.ndarray, truth:np.ndarray) -> np.ndarray:
        """Compute the hinge loss for the given predictions and ground truth.

        Arguments:
        ---------
            pred: np.ndarray
                The predicted logits from the model.
            truth: np.ndarray The ground truth labels.

        Returns:
        -------
            np.ndarray: The computed hinge loss.

        """
        num_classes = pred.shape[-1]
        flat_pred = pred.reshape(-1, num_classes)
        flat_truth = truth.reshape(-1)
        mask = np.ones_like(flat_pred, dtype=np.bool_)
        mask[np.arange(flat_truth.shape[0]), flat_truth] = False
        flat_hinge_loss = flat_pred[~mask] - logsumexp(flat_pred[mask].reshape(flat_truth.shape[0], -1), axis=1)
        return flat_hinge_loss.reshape(truth.shape)

    def write_flat_params_to_layer(self:Self, flat_params:torch.Tensor, layer:torch.nn.Module) -> None:
        state_dict = {}
        idx = 0
        for name, params in layer.state_dict().items():
            numel = params.numel()
            new_params = flat_params[idx: idx + numel].view_as(params)
            state_dict[name] = new_params
            idx += numel
        layer.load_state_dict(state_dict)

    def sample_models(self:Self) -> list:
        sampled_weights = self.la.sample(self.n_samples)
        sampled_models = []
        for i in range(self.n_samples):
            sampled_model, criterion, _ = ShadowModelHandler()._get_model_criterion_optimizer()
            sampled_model.load_state_dict(self.shadow_model.model_obj.state_dict())
            # Update weights in last layer, which is assumed to be named fc
            self.write_flat_params_to_layer(sampled_weights[i], sampled_model.fc)
            sampled_models.append(PytorchModel(sampled_model, criterion))
        return sampled_models

    def prepare_attack(self:Self) -> None:
        """Prepare data needed for running the attack on the target model and dataset.

        Signals are computed on the auxiliary model(s) and dataset.
        """
        logger.info("Preparing shadow model for BMIA attack")
        # Check number of shadow models that are available

        # sample dataset to compute histogram
        logger.info("Preparing attack data for training the BMIA attack")

        # Get all available indices for attack dataset, if self.online = True, include training and test data
        self.attack_data_indices = self.sample_indices_from_population(include_train_indices = False,
                                                                       include_test_indices = True)

        # train shadow models
        logger.info(f"Check for 1 shadow model (dataset: {len(self.attack_data_indices)} points)")
        shadow_model_indices = ShadowModelHandler().create_shadow_models(
            num_models = 1,
            shadow_population = self.attack_data_indices,
            training_fraction = self.training_data_fraction,
            online = False,)
        # load shadow models
        shadow_models, _ = ShadowModelHandler().get_shadow_models(shadow_model_indices)
        self.shadow_model = shadow_models[0]
        train_indices = ShadowModelHandler().get_shadow_model_metadata(shadow_model_indices)[0].train_indices
        train_loader = self.handler.get_dataloader(train_indices, shuffle=False)
        self.la = Laplace(self.shadow_model.model_obj,
            "classification",
            subset_of_weights="last_layer",
            hessian_structure="kron")
        self.la.fit(train_loader)
        self.la.optimize_prior_precision(pred_type='glm', link_approx='probit')

    def run_attack(self:Self) -> MIAResult:
        """Run the attack on the target model and dataset.

        Returns
        -------
            Result(s) of the metric.

        """

        audit_data_indices = self.audit_dataset["data"]
        in_members = self.audit_dataset["in_members"]
        out_members = self.audit_dataset["out_members"]

        logger.info("Running BMIA attack")

        ground_truth_indices = self.handler.get_labels(audit_data_indices)

        # run target points through real model to get logits
        logits_target = ShadowModelHandler().load_logits(name="target")
        # collect the log confidence output of the correct class (which is the negative cross-entropy loss)
        target_scores = self.hinge_loss(logits_target, ground_truth_indices)

        # Sample models using Laplace approximation
        sampled_models = self.sample_models()

        # Compute hinge values for all sampled models
        sampled_logits = np.array(self.signal(sampled_models, self.handler, self.audit_dataset["data"]))
        ground_truth = np.tile(ground_truth_indices, (self.n_samples, 1))
        sampled_scores = self.hinge_loss(sampled_logits, ground_truth)

        # Compute t distributed statistic
        diff = target_scores - sampled_scores
        diff_mean = diff.mean(axis=0)
        diff_stdn = diff.std(axis=0) / np.sqrt(self.n_samples)
        t = diff_mean / (diff_stdn + self.epsilon)
        score = t_dist.cdf(t, self.n_samples - 1)

        # pick out the in-members and out-members signals
        self.in_member_signals = score[in_members].reshape(-1,1)
        self.out_member_signals = score[out_members].reshape(-1,1)

        # set true labels for being in the training dataset
        true_labels = np.concatenate([np.ones(len(self.in_member_signals)),np.zeros(len(self.out_member_signals)),])
        signal_values = np.concatenate([self.in_member_signals, self.out_member_signals])

        # compute ROC, TP, TN etc
        return MIAResult.from_full_scores(true_membership=true_labels,
                                    signal_values=signal_values,
                                    result_name="BMIA",
                                    metadata=self.configs.model_dump())
