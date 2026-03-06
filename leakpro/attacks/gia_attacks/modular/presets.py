"""Pre-configured attacks from the GIA literature.

This module provides factory functions for classic gradient inversion attacks
using the modular component system.

Each preset function returns an AttackConfig that can be modified before building.

Example:
    # Simple usage - build immediately
    attack = inverting_gradients_attack().build()

    # Modify before building
    config = inverting_gradients_attack()
    config.tv_weight = 1e-6
    config.max_iterations = 10000
    config.learning_rate = 0.01
    attack = config.build()

Attacks:
    - DLG: Deep Leakage from Gradients (Zhu et al. 2019)
    - iDLG: Improved Deep Leakage from Gradients (Zhao et al. 2020)
    - InvertingGradients: Inverting Gradients (Geiping et al. 2020)
    - Huang: Huang et al. with BN statistics (2021)
    - GIA Running: Inferred BN statistics
    - GIA Estimate: Proxy data BN statistics

"""

from dataclasses import dataclass

from torch import nn

from leakpro.attacks.gia_attacks.modular.components.composable_optimizer import ComposableOptimizer
from leakpro.attacks.gia_attacks.modular.components.initialization import (
    RandomNoiseInitialization,
)
from leakpro.attacks.gia_attacks.modular.components.label_inference import (
    IDLGLabelInference,
    JointLabelOptimization,
    OracleLabels,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.bn_statistics_strategies import (
    InferredBNStatisticsStrategy,
    ProxyBNStatisticsStrategy,
    RunningBNStatisticsStrategy,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.constraints import (
    ClipConstraint,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.label_strategies import (
    FixedLabels,
    JointLabelOptimizationStrategy,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.loss_components import (
    BNStatisticsRegularization,
    GradientMatchingLoss,
    LabelEntropyRegularization,
    TVRegularization,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.step_strategies import (
    StandardStepStrategy,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.training_simulator import (
    MultiEpochTrainingSimulation,
)
from leakpro.attacks.gia_attacks.modular.core.threat_model import (
    model_a_eavesdropper,
    model_b_informed,
    model_c_parameter_aware,
    model_d_data_enhanced,
    model_e_statistical,
)
from leakpro.attacks.gia_attacks.modular.orchestrator import ModularGIAOrchestrator


@dataclass
class AttackConfig:
    """Configuration for building a gradient inversion attack.

    This stores all parameters needed to construct an attack. Modify any
    attributes before calling .build() to customize the attack.
    """

    # Attack strategy
    threat_model_type: str = "model_a"  # "model_a" through "model_h" (literature taxonomy)
    labels: str = "oracle"  # "oracle", "idlg", "joint"

    # Loss components
    gradient_loss_type: str = "cosine"  # "l2", "cosine", "fisher"
    tv_weight: float = 0.0  # Total variation regularization weight
    bn_weight: float = 0.0  # Batch normalization statistics weight
    bn_strategy: str = "running"  # "running", "inferred", "proxy"
    bn_momentum: float = 0.1  # Only for inferred strategy
    label_entropy_weight: float = 0.0  # Entropy regularization for joint label optimization

    # Optimizer settings
    learning_rate: float = 0.1
    label_learning_rate: float | None = None  # If None, uses learning_rate
    max_iterations: int = 300
    optimizer_type: str = "adam"  # "adam", "lbfgs", "sgd"
    scheduler_type: str | None = None  # "cosine", "step", "exponential", None
    patience: int = 10000  # Early stopping patience
    log_interval: int | None = None  # Logging frequency

    # Step strategy
    use_gradient_sign: bool = False  # Use sign of gradients in updates

    # Training simulation
    training_simulator_epochs: int = 1
    training_simulator_compute_mode: str = "gradients"  # "gradients" or "updates"
    training_simulator_model_mode: str = "eval"  # "eval" or "train"

    # Constraint
    use_clip_constraint: bool = True  # Clip reconstruction to valid range

    # Loss function
    loss_fn: nn.Module | None = None  # Custom loss function (default: CrossEntropyLoss)

    def build(self, training_simulator: MultiEpochTrainingSimulation = None) -> ModularGIAOrchestrator:
        """Build the attack orchestrator from this configuration."""
        # Build threat model using literature taxonomy
        threat_model_factories = {
            "model_a": model_a_eavesdropper,
            "model_b": model_b_informed,
            "model_c": model_c_parameter_aware,
            "model_d": model_d_data_enhanced,
            "model_e": model_e_statistical,
        }

        if self.threat_model_type not in threat_model_factories:
            raise ValueError(
                f"Unknown threat_model_type: {self.threat_model_type}. "
                f"Available: {list(threat_model_factories.keys())}"
            )

        threat_model = threat_model_factories[self.threat_model_type]()

        # Build label inference and strategy
        label_factories = {
            "oracle": (OracleLabels, FixedLabels),
            "idlg": (IDLGLabelInference, FixedLabels),
            "joint": (JointLabelOptimization, JointLabelOptimizationStrategy),
        }

        label_inference_cls, label_strategy_cls = label_factories[self.labels]

        label_inference = label_inference_cls()
        label_strategy = label_strategy_cls()

        # Build initialization
        initialization = RandomNoiseInitialization(mean=0.0, std=1.0)

        # Build training simulator first (needed by loss components)
        if training_simulator is None:
            training_simulator = MultiEpochTrainingSimulation(
                epochs=self.training_simulator_epochs,
                compute_mode=self.training_simulator_compute_mode,
                model_mode=self.training_simulator_model_mode,
            )

        # Build loss components (pass training_simulator explicitly to those that need it)
        loss_components = [
            GradientMatchingLoss(
                loss_type=self.gradient_loss_type,
                weight=1.0,
                training_simulator=training_simulator,
            )
        ]

        if self.tv_weight > 0:
            loss_components.append(TVRegularization(weight=self.tv_weight))

        if self.bn_weight > 0:
            if self.bn_strategy == "running":
                bn_strategy_obj = RunningBNStatisticsStrategy()
            elif self.bn_strategy == "inferred":
                bn_strategy_obj = InferredBNStatisticsStrategy(momentum=self.bn_momentum)
            elif self.bn_strategy == "proxy":
                bn_strategy_obj = ProxyBNStatisticsStrategy()
            else:
                raise ValueError(f"Unknown bn_strategy: {self.bn_strategy}")

            loss_components.append(
                BNStatisticsRegularization(strategy=bn_strategy_obj, weight=self.bn_weight)
            )

        if self.label_entropy_weight > 0 and self.labels == "joint":
            loss_components.append(
                LabelEntropyRegularization(weight=self.label_entropy_weight)
            )

        # Build constraint
        constraint = ClipConstraint() if self.use_clip_constraint else None

        # Build step strategy
        step_strategy = StandardStepStrategy(use_gradient_sign=self.use_gradient_sign) if self.use_gradient_sign else None

        # Build optimizer
        optimization = ComposableOptimizer(
            loss_components=loss_components,
            constraint=constraint,
            label_strategy=label_strategy,
            step_strategy=step_strategy,
            learning_rate=self.learning_rate,
            label_learning_rate=self.label_learning_rate,
            max_iterations=self.max_iterations,
            optimizer_type=self.optimizer_type,
            scheduler_type=self.scheduler_type,
            patience=self.patience,
            log_interval=self.log_interval,
            training_simulator=training_simulator,
            loss_fn=self.loss_fn,
        )

        return ModularGIAOrchestrator(
            threat_model=threat_model,
            label_inference=label_inference,
            initialization=initialization,
            optimization=optimization,
        )


def dlg_attack() -> AttackConfig:
    """Deep Leakage from Gradients (DLG).

    Original attack by Zhu et al. 2019. Optimizes both data and labels jointly
    using L-BFGS optimizer.

    Threat Model: Model A (Eavesdropper)
    - Has access to: Gradients only
    - Does not have: Model parameters, BN statistics

    Components:
    - Joint label optimization (optimizes labels during attack)
    - Random noise initialization
    - L-BFGS optimization with L2 gradient matching
    - No constraints

    Default Settings:
        - learning_rate: 1.0 (paper default)
        - max_iterations: 300
        - optimizer_type: "lbfgs"
        - labels: "joint" (can change to "oracle" if you have true labels)
        - patience: 50

    Usage:
        config = dlg_attack()
        config.max_iterations = 1200  # Adjust as needed
        config.labels = "oracle"  # If you have true labels
        attack = config.build()

    Reference:
        Zhu, L., Liu, Z., & Han, S. (2019). Deep leakage from gradients.
        NeurIPS 2019.

    """
    return AttackConfig(
        threat_model_type="model_a",
        labels="joint",
        gradient_loss_type="l2",
        tv_weight=0.0,
        learning_rate=1.0,
        max_iterations=300,
        optimizer_type="lbfgs",
        patience=50,
        use_gradient_sign=False,
        use_clip_constraint=False,
        training_simulator_epochs=1,
        training_simulator_model_mode="eval",
    )


def idlg_attack() -> AttackConfig:
    """Improved Deep Leakage from Gradients (iDLG).

    Improvement by Zhao et al. 2020. Analytically infers labels from gradient
    signs, then optimizes only the data.

    Threat Model: Model A (Eavesdropper)
    - Has access to: Gradients only
    - Does not have: Model parameters, BN statistics

    Components:
    - iDLG label inference (analytical from gradient signs)
    - Random noise initialization
    - L-BFGS optimization with L2 gradient matching
    - No constraints

    Default Settings:
        - learning_rate: 1.0
        - max_iterations: 300
        - optimizer_type: "lbfgs"
        - labels: "idlg" (analytical label inference)

    Usage:
        config = idlg_attack()
        config.max_iterations = 500  # Adjust as needed
        attack = config.build()

    Reference:
        Zhao, B., Mopuri, K. R., & Bilen, H. (2020). iDLG: Improved deep
        leakage from gradients. arXiv preprint arXiv:2001.02610.

    """
    return AttackConfig(
        threat_model_type="model_a",
        labels="idlg",
        gradient_loss_type="l2",
        tv_weight=0.0,
        learning_rate=1.0,
        max_iterations=300,
        optimizer_type="lbfgs",
        use_gradient_sign=False,
        use_clip_constraint=False,
        training_simulator_epochs=1,
        training_simulator_model_mode="eval",
    )


def inverting_gradients_attack() -> AttackConfig:
    """Inverting Gradients attack (Geiping et al. 2020).

    More sophisticated attack with:
    - Total variation regularization
    - Cosine similarity metric
    - Adam optimizer for better convergence

    Threat Model: Model B (Informed Eavesdropper)
    - Has access to: Gradients + auxiliary knowledge (TV regularization assumes smooth images)

    Components:
    - Oracle label inference (assumes labels are known)
    - Random noise initialization
    - Adam with TV regularization
    - Clip constraints

    Default Settings:
        - learning_rate: 0.1
        - max_iterations: 4000
        - optimizer_type: "adam"
        - gradient_loss_type: "cosine"
        - tv_weight: 1e-3
        - labels: "oracle" (can change to "idlg" or "joint")
        - use_gradient_sign: True

    Reference:
        Geiping, J., Bauermeister, H., Dröge, H., & Moeller, M. (2020).
        Inverting gradients-how easy is it to break privacy in federated learning?
        NeurIPS 2020.

    """
    return AttackConfig(
        threat_model_type="model_b",  # Uses auxiliary knowledge (TV regularization)
        labels="oracle",
        gradient_loss_type="cosine",
        tv_weight=1e-3,
        label_entropy_weight=0.0,
        learning_rate=0.1,
        label_learning_rate=None,
        max_iterations=4000,
        optimizer_type="adam",
        scheduler_type="step",
        use_gradient_sign=True,
        use_clip_constraint=True,
        training_simulator_epochs=1,
        training_simulator_model_mode="eval",
    )


def huang_attack() -> AttackConfig:
    """Huang et al. attack with running BN statistics regularization.

    This attack uses the model's running BN statistics (running_mean and
    running_var) as targets for regularization. This is the approach from
    Huang et al. 2021.

    Threat Model: Model E (Statistical-Informed Eavesdropper)
    - Has access to: Gradients + BN statistics + auxiliary knowledge + hyperparameters + surrogate data

    Components:
    - Oracle label inference (assumes labels are known)
    - Random noise initialization
    - Adam with TV + BN regularization using running statistics
    - Cosine LR scheduler

    Default Settings:
        - learning_rate: 0.1
        - max_iterations: 10000
        - optimizer_type: "adam"
        - scheduler_type: "step"
        - gradient_loss_type: "cosine"
        - tv_weight: 0.052 (paper default)
        - bn_weight: 0.00016 (paper default)
        - labels: "oracle"

    Reference:
        Huang, Y., et al. "Evaluating Gradient Inversion Attacks and Defenses
        in Federated Learning." NeurIPS 2021.

    """
    return AttackConfig(
        threat_model_type="model_e",  # Has BN statistics + full capabilities
        labels="oracle",
        gradient_loss_type="cosine",
        tv_weight=0.052,
        bn_weight=0.00016,
        bn_strategy="running",
        learning_rate=0.1,
        max_iterations=10000,
        optimizer_type="adam",
        scheduler_type="step",
        use_clip_constraint=True,
        training_simulator_epochs=1,
        training_simulator_model_mode="train",
    )


def gia_running_attack() -> AttackConfig:
    """GIA Running attack with inferred BN statistics.

    This attack infers the client's batch statistics by observing how
    the running statistics changed during training, using the momentum
    parameter. Requires access to both pre- and post-training running stats.

    Threat Model: Model E (Statistical-Informed Eavesdropper)
    - Has access to: Gradients + BN running stats before & after training + hyperparameters
    - The hyperparameters capability includes batch size needed for inference

    Components:
    - Oracle label inference (assumes labels are known)
    - Random noise initialization
    - Adam with TV + inferred BN regularization

    Default Settings:
        - learning_rate: 0.1
        - max_iterations: 3000
        - optimizer_type: "adam"
        - gradient_loss_type: "cosine"
        - tv_weight: 0.052
        - bn_weight: 0.00016
        - bn_strategy: "inferred"
        - bn_momentum: 0.1 (PyTorch default)
        - labels: "oracle"

    Note: Requires passing client observations with pre/post BN stats to run_attack().

    """
    return AttackConfig(
        threat_model_type="model_e",  # Has BN statistics + hyperparameters
        labels="oracle",
        gradient_loss_type="cosine",
        tv_weight=0.052,
        bn_weight=0.00016,
        bn_strategy="inferred",
        bn_momentum=0.1,
        learning_rate=0.1,
        max_iterations=3000,
        optimizer_type="adam",
        scheduler_type="step",
        use_clip_constraint=True,
        training_simulator_epochs=1,
        training_simulator_model_mode="train",
    )


def gia_estimate_attack() -> AttackConfig:
    """GIA Estimate attack using proxy data to estimate BN statistics.

    This attack uses a proxy/surrogate dataset (from the same or similar
    domain) to estimate what the batch statistics might look like. This
    is more realistic than having access to exact running statistics.

    Threat Model: Model D (Data-Enhanced Eavesdropper)
    - Has access to: Gradients + surrogate/proxy dataset + auxiliary knowledge + hyperparameters

    Components:
    - Oracle label inference (assumes labels are known)
    - Random noise initialization
    - Adam with TV + proxy BN regularization

    Default Settings:
        - learning_rate: 0.1
        - max_iterations: 10000
        - optimizer_type: "adam"
        - gradient_loss_type: "cosine"
        - tv_weight: 0.052
        - bn_weight: 0.00016
        - bn_strategy: "proxy"
        - labels: "oracle"

    Usage:
        config = gia_estimate_attack()
        config.bn_weight = 0.001  # Adjust BN weight
        config.labels = "idlg"  # Use different label inference
        attack = config.build()

    Note: Requires passing proxy_dataloader to run_attack().

    Reference:
        Supports multiple normalization types: BatchNorm2d, LayerNorm,
        LayerNorm2d (ConvNeXt), InstanceNorm2d
    """
    return AttackConfig(
        threat_model_type="model_d",  # Has surrogate data
        labels="oracle",
        gradient_loss_type="cosine",
        tv_weight=0.052,
        bn_weight=0.00016,
        bn_strategy="proxy",
        learning_rate=0.1,
        max_iterations=10000,
        optimizer_type="adam",
        scheduler_type="step",
        use_clip_constraint=True,
        training_simulator_epochs=1,
        training_simulator_model_mode="train",
    )


__all__ = [
    "AttackConfig",
    "dlg_attack",
    "idlg_attack",
    "inverting_gradients_attack",
    "huang_attack",
    "gia_running_attack",
    "gia_estimate_attack",
]
