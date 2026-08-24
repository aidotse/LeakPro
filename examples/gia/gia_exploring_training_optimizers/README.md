# Modular GIA Framework - Architecture Overview

This example demonstrates the **modular Gradient Inversion Attack (GIA) framework**, a flexible component-based architecture for implementing and experimenting with gradient inversion attacks in federated learning.

## 🏗️ Architecture Overview

The modular framework decomposes gradient inversion attacks into **composable building blocks**, making it easy to mix and match components to reproduce attacks from the literature or create new variants.

### Core Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ModularGIAOrchestrator                       │
│  Coordinates the complete attack pipeline                       │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├── ThreatModel (validates capabilities)
             │
             ├── 1️⃣ LabelInferenceStrategy
             │   ├── IDLGLabelInference (analytical)
             │   ├── JointLabelOptimization (optimize with data)
             │   └── OracleLabels (ground truth)
             │
             ├── 2️⃣ InitializationStrategy
             │   └── RandomNoiseInitialization
             │
             └── 3️⃣ OptimizationStrategy
                 └── ComposableOptimizer
                     ├── LossComponents[]
                     │   ├── GradientMatchingLoss (core)
                     │   ├── TVRegularization (smoothness)
                     │   ├── BNStatisticsRegularization
                     │   └── LabelEntropyRegularization
                     ├── LabelStrategy
                     │   ├── FixedLabels
                     │   └── JointLabelOptimizationStrategy
                     ├── StepStrategy
                     │   └── StandardStepStrategy
                     ├── ConstraintStrategy
                     │   ├── ClipConstraint
                     │   └── NoConstraint
                     └── TrainingSimulator
                         └── MultiEpochTrainingSimulation
```

## 🧩 Component Types

### 1. **Label Inference** (`leakpro/attacks/gia_attacks/modular/components/label_inference.py`)

Determines labels from gradients before optimizing the reconstruction.

| Component | Description | Paper |
|-----------|-------------|-------|
| `IDLGLabelInference` | Analytical inference using gradient signs | Zhao et al. 2020 (iDLG) |
| `JointLabelOptimization` | Optimize labels jointly with data | Zhu et al. 2019 (DLG) |
| `OracleLabels` | Use ground-truth labels (debugging/upper bound) | N/A |

**Usage:**
```python
from leakpro.attacks.gia_attacks.modular.components.label_inference import IDLGLabelInference

label_inference = IDLGLabelInference()
```

---

### 2. **Initialization** (`leakpro/attacks/gia_attacks/modular/components/initialization.py`)

Creates the starting point for optimization.

| Component | Description |
|-----------|-------------|
| `RandomNoiseInitialization` | Random Gaussian noise N(μ, σ) |

**Usage:**
```python
from leakpro.attacks.gia_attacks.modular.components.initialization import RandomNoiseInitialization

initialization = RandomNoiseInitialization(mean=0.0, std=1.0)
```

---

### 3. **Optimization Building Blocks** (`leakpro/attacks/gia_attacks/modular/components/optimization_building_blocks/`)

#### 3.1 Loss Components (`loss_components.py`)

| Component | Description | Required Capability |
|-----------|-------------|---------------------|
| `GradientMatchingLoss` | Match gradients (L2, cosine, fisher) | `has_gradients` |
| `TVRegularization` | Total variation (smoothness) | `has_auxiliary_knowledge` |
| `BNStatisticsRegularization` | Match batch norm statistics | Depends on strategy |
| `LabelEntropyRegularization` | Encourage confident labels | None |

**Usage:**
```python
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.loss_components import (
    GradientMatchingLoss,
    TVRegularization,
)

loss_components = [
    GradientMatchingLoss(loss_type="cosine", weight=1.0),
    TVRegularization(weight=1e-3),
]
```

#### 3.2 BN Statistics Strategies (`bn_statistics_strategies.py`)

Three strategies for obtaining batch normalization statistics:

| Strategy | Description | Required Capabilities | Paper |
|----------|-------------|----------------------|-------|
| `RunningBNStatisticsStrategy` | Use model's running BN stats | `has_bn_statistics` | Huang et al. 2021 |
| `InferredBNStatisticsStrategy` | Infer from momentum updates | `has_bn_statistics`, `has_local_hyperparameters` | GIA Running |
| `ProxyBNStatisticsStrategy` | Estimate from proxy data | `has_surrogate_data` | GIA Estimate |

**Usage:**
```python
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.bn_statistics_strategies import (
    RunningBNStatisticsStrategy,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.loss_components import (
    BNStatisticsRegularization,
)

bn_strategy = RunningBNStatisticsStrategy()
bn_loss = BNStatisticsRegularization(strategy=bn_strategy, weight=0.00016)
```

#### 3.3 Training Simulators (`training_simulator.py`)

Simulate client-side training to compute gradients or parameter updates:

| Simulator | Description | Use Case |
|-----------|-------------|----------|
| `MultiEpochTrainingSimulation` | Simulate single or multi-epoch training | All attacks (DLG, iDLG, Geiping, FedAvg scenarios). Use `epochs=1, compute_mode="gradients"` for basic gradient-based attacks, or `epochs>1, compute_mode="updates"` for FedAvg scenarios |

**Usage:**
```python
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.training_simulator import (
    MultiEpochTrainingSimulation,
)

training_simulator = MultiEpochTrainingSimulation(
    epochs=1,
    optimizer_type="sgd",
    compute_mode="updates",  # or "gradients"
    model_mode="train",      # or "eval"
)
```

#### 3.4 Label Strategies (`label_strategies.py`)

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `FixedLabels` | Use pre-inferred labels | iDLG, InvertingGradients |
| `JointLabelOptimizationStrategy` | Optimize labels during attack | DLG |

#### 3.5 Step Strategies (`step_strategies.py`)

| Strategy | Description |
|----------|-------------|
| `StandardStepStrategy` | Standard zero_grad → backward → step |

#### 3.6 Constraints (`constraints.py`)

| Constraint | Description |
|-----------|-------------|
| `ClipConstraint` | Clip to valid pixel range [0, 1] |
| `NoConstraint` | No constraints |

---

## 🎯 Threat Models

The framework uses **literature-based threat models** (Models A-H from Carletti et al.) to validate that attacks only use capabilities they should have access to.

| Model | Name | Capabilities | Example Attacks |
|-------|------|--------------|-----------------|
| **A** | Eavesdropper | Gradients only | DLG, iDLG (basic) |
| **B** | Informed Eavesdropper | + Auxiliary knowledge (TV) | InvertingGradients |
| **C** | Parameter-Aware | + Hyperparameters | - |
| **D** | Data-Enhanced | + Surrogate dataset | GIA Estimate |
| **E** | Statistical-Informed | + BN statistics | Huang, GIA Running |
| **F-H** | Active/Sybil | Can modify architecture/inject clients | (Not yet implemented) |

**Capabilities defined:**
- `has_gradients`: Access to model gradients
- `has_bn_statistics`: Access to batch norm statistics
- `has_local_hyperparameters`: Knowledge of learning rate, batch size, etc.
- `has_surrogate_data`: Public dataset from same/similar domain
- `has_auxiliary_knowledge`: General domain assumptions (smoothness, ranges)
- `can_modify_architecture`: Ability to manipulate model
- `can_inject_sybils`: Ability to inject fake clients

---

## 🚀 Usage Examples

### Using Presets (Recommended)

The easiest way to use the framework is through **preset configurations**:

```python
from leakpro.attacks.gia_attacks.modular.presets import (
    dlg_attack,
    idlg_attack,
    inverting_gradients_attack,
    huang_attack,
    gia_running_attack,
    gia_estimate_attack,
)

# 1. Create attack configuration
config = inverting_gradients_attack()

# 2. Customize (optional)
config.max_iterations = 4000
config.learning_rate = 0.1
config.tv_weight = 1e-3

# 3. Build the orchestrator
attack = config.build()

# 4. Run the attack
reconstruction, attack_config = attack.run_attack(
    target_model=model,
    client_observations=client_observations,
    input_shape=(batch_size, 3, 32, 32),
    device=device,
)
```

### Available Presets

| Preset | Description | Threat Model |
|--------|-------------|--------------|
| `dlg_attack()` | Deep Leakage from Gradients (Zhu et al. 2019) | Model A |
| `idlg_attack()` | iDLG with analytical labels (Zhao et al. 2020) | Model A |
| `inverting_gradients_attack()` | Geiping et al. 2020 with TV reg | Model B |
| `huang_attack()` | Huang et al. 2021 with BN stats | Model E |
| `gia_running_attack()` | Inferred BN statistics | Model E |
| `gia_estimate_attack()` | Proxy data BN estimation | Model D |

### Custom Attack Composition

Build your own attack from scratch:

```python
from leakpro.attacks.gia_attacks.modular.orchestrator import ModularGIAOrchestrator
from leakpro.attacks.gia_attacks.modular.core.threat_model import model_b_informed
from leakpro.attacks.gia_attacks.modular.components.label_inference import IDLGLabelInference
from leakpro.attacks.gia_attacks.modular.components.initialization import RandomNoiseInitialization
from leakpro.attacks.gia_attacks.modular.components.composable_optimizer import ComposableOptimizer
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.loss_components import (
    GradientMatchingLoss,
    TVRegularization,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.constraints import ClipConstraint

# 1. Define threat model
threat_model = model_b_informed()

# 2. Choose components
label_inference = IDLGLabelInference()
initialization = RandomNoiseInitialization(mean=0.0, std=1.0)

# 3. Build optimizer with loss components
loss_components = [
    GradientMatchingLoss(loss_type="cosine", weight=1.0),
    TVRegularization(weight=1e-3),
]

optimization = ComposableOptimizer(
    loss_components=loss_components,
    constraint=ClipConstraint(),
    learning_rate=0.1,
    max_iterations=4000,
    optimizer_type="adam",
)

# 4. Create orchestrator
attack = ModularGIAOrchestrator(
    threat_model=threat_model,
    label_inference=label_inference,
    initialization=initialization,
    optimization=optimization,
)

# 5. Run attack
reconstruction, config = attack.run_attack(
    target_model=model,
    client_observations=client_observations,
    input_shape=input_shape,
    device=device,
)
```

---

## 📊 Client Observations

The `FLClientSimulator` generates observations that the server can use:

```python
from leakpro.fl_utils.fl_client_simulator import FLClientSimulator

client_simulator = FLClientSimulator(
    client_data=dataloader,
    data_mean=data_mean,
    data_std=data_std,
    device=device,
    num_classes=10,
)

# Simulate client training and observe what server receives
client_observations = client_simulator.train_and_observe(
    server_model=model,
    training_simulator=training_simulator,
    loss_fn=nn.CrossEntropyLoss(),
    send_labels_to_server=True,  # Oracle labels
    threat_model="gia_running",   # Determines what to send
)
```

**ClientObservations contains:**
- `gradients`: Dict of parameter gradients
- `labels`: Ground truth labels (if `send_labels_to_server=True`)
- `post_bn_stats`: BN statistics after training
- `pre_bn_stats`: BN statistics before training
- `batch_size`: Batch size used for training
- `data_mean`, `data_std`: Normalization parameters

---

## 🔍 Component Validation

The orchestrator **automatically validates** that components are compatible with the threat model:

```python
# This will raise ValueError if TV regularization is used with Model A
# (Model A doesn't have auxiliary knowledge capability)
attack = ModularGIAOrchestrator(
    threat_model=model_a_eavesdropper(),  # Only has gradients
    optimization=optimizer_with_tv_loss,   # Requires auxiliary_knowledge
    # ... other components
)
# ❌ ValueError: Component requires 'has_auxiliary_knowledge' but threat model only provides {...}
```

---

## 🎨 Visualization

Use the provided `visualize.py` to compare multiple attacks:

```python
from visualize import visualize_multiple_attacks

visualize_multiple_attacks(
    original=original_images,
    reconstructions=reconstructions,
    attack_names=["DLG", "iDLG", "Geiping", "Huang"],
    save_path="comparison.png",
)
```

---

## 📁 File Structure

```
examples/gia/gia_modular_basic/
├── README.md           # This file
├── main.py             # Example script running multiple attacks
├── cifar.py            # CIFAR-10 data loading utilities
├── model.py            # ResNet model definition
└── visualize.py        # Visualization utilities

leakpro/attacks/gia_attacks/modular/
├── orchestrator.py                    # Main attack coordinator
├── presets.py                         # Pre-configured attacks
├── core/
│   ├── component_base.py              # Base classes and interfaces
│   └── threat_model.py                # Threat model definitions (A-H)
└── components/
    ├── label_inference.py             # Label inference strategies
    ├── initialization.py              # Initialization strategies
    ├── composable_optimizer.py        # Main optimization component
    └── optimization_building_blocks/
        ├── loss_components.py         # Loss functions
        ├── bn_statistics_strategies.py # BN statistics handling
        ├── training_simulator.py      # Training simulation
        ├── label_strategies.py        # Label handling
        ├── step_strategies.py         # Optimization step execution
        └── constraints.py             # Reconstruction constraints
```

---

## 🔬 Running the Example

```bash
cd examples/gia/gia_modular_basic
python main.py
```

This will:
1. Load CIFAR-10 data (8 images)
2. Simulate client training with a ResNet model
3. Run multiple gradient inversion attacks:
   - InvertingGradients (Geiping et al.)
   - Huang et al. with BN statistics
   - GIA Running with inferred BN stats
   - GIA Estimate with proxy data
4. Visualize and compare reconstructions

---

## 💡 Design Principles

1. **Modularity**: Each component is independent and replaceable
2. **Composability**: Mix and match components to create new attacks
3. **Type Safety**: Strong typing with comprehensive type hints
4. **Validation**: Automatic threat model validation
5. **Extensibility**: Easy to add new components without modifying existing code
6. **Literature Mapping**: Clear connection to published papers
7. **Debugging**: Oracle components for establishing upper bounds

---

## 📚 References

- **DLG**: Zhu, L., Liu, Z., & Han, S. (2019). Deep leakage from gradients. NeurIPS.
- **iDLG**: Zhao, B., Mopuri, K. R., & Bilen, H. (2020). iDLG: Improved deep leakage from gradients.
- **InvertingGradients**: Geiping, J., et al. (2020). Inverting gradients - How easy is it to break privacy in federated learning? NeurIPS.
- **Huang**: Huang, Y., et al. (2021). Evaluating Gradient Inversion Attacks and Defenses in Federated Learning. NeurIPS.
- **Threat Models**: Carletti, V., et al. (2023). SoK: Gradient Inversion Attacks in Federated Learning.

---

## 🤝 Contributing

To add a new component:

1. Inherit from the appropriate base class (`Component`, `LossComponent`, `BNStatisticsStrategy`, etc.)
2. Implement required abstract methods
3. Define `get_metadata()` with required capabilities
4. Add to `__all__` in the module
5. (Optional) Create a preset configuration in `presets.py`

