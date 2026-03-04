"""Threat model system for gradient inversion attacks.

This module defines attacker capabilities using boolean flags.
Components declare which capabilities they require, and the threat model
validates if it satisfies those requirements.

Maps to literature threat models (A-H) from FL GIA taxonomy papers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from leakpro.attacks.gia_attacks.modular.core.component_base import Component


@dataclass
class AttackerCapabilities:
    """Attacker capabilities for gradient inversion attacks.

    Each capability is a boolean: attacker either has it or doesn't.
    Components declare which capabilities they require.
    """

    # =========================================================================
    # Information Access
    # =========================================================================

    has_gradients: bool = True
    """Has access to model gradients (always True in FL)"""

    has_bn_statistics: bool = False
    """Has access to batch normalization running statistics"""

    has_local_hyperparameters: bool = False
    """Knows local training hyperparameters (lr, batch size, epochs, optimizer)"""

    has_surrogate_data: bool = False
    """Has access to public dataset from same/similar domain"""

    has_auxiliary_knowledge: bool = False
    """
    Has easily accessible auxiliary knowledge (e.g., natural images are smooth,
    enabling TV regularization; images are bounded [0,1]; etc.)

    Note: This is NOT data from the actual distribution, just general assumptions
    about the data domain that anyone could reasonably make.
    """

    # =========================================================================
    # Active Capabilities
    # =========================================================================

    can_modify_architecture: bool = False
    """Can manipulate the global model architecture"""

    can_inject_sybils: bool = False
    """Can inject fake clients into the federation"""

    # =========================================================================
    # Metadata
    # =========================================================================

    name: str = "custom"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "has_gradients": self.has_gradients,
            "has_bn_statistics": self.has_bn_statistics,
            "has_local_hyperparameters": self.has_local_hyperparameters,
            "has_surrogate_data": self.has_surrogate_data,
            "has_auxiliary_knowledge": self.has_auxiliary_knowledge,
            "can_modify_architecture": self.can_modify_architecture,
            "can_inject_sybils": self.can_inject_sybils,
            "description": self.description,
        }


class ThreatModel:
    """Threat model defined by attacker capabilities.

    Used to:
    1. Declare what information/access an attacker has
    2. Validate if components can run under this threat model
    """

    def __init__(self, capabilities: AttackerCapabilities) -> None:
        self.capabilities = capabilities

    @property
    def name(self) -> str:
        """Name of the threat model (derived from capabilities)."""
        return self.capabilities.name

    def allows_component(self, required_capabilities: dict[str, bool]) -> tuple[bool, list[str]]:
        """Check if this threat model satisfies component requirements.

        Args:
            required_capabilities: Dict mapping capability_name -> required (True/False)

        Returns:
            (is_satisfied, list_of_missing_capabilities)

        """
        missing = []

        for cap_name, is_required in required_capabilities.items():
            if not is_required:
                continue  # Component doesn't need this capability

            actual = getattr(self.capabilities, cap_name, None)

            if actual is None:
                missing.append(f"Unknown capability: {cap_name}")
            elif not actual:
                # Component requires this but threat model doesn't have it
                missing.append(f"Missing: {cap_name}")

        return len(missing) == 0, missing

    def __repr__(self) -> str:
        """String representation of the threat model."""
        return f"ThreatModel(name='{self.name}', capabilities={self.capabilities})"



# =============================================================================
# Component Requirements
# =============================================================================

def compute_requirements(*components: Component) -> dict[str, bool]:
    """Compute capability requirements from components.

    Takes the union: if any component requires a capability, it's required.
    This is used when a component (like an optimizer) is composed of
    sub-components (like loss functions).

    Args:
        *components: Variable number of component instances or requirement dicts

    Returns:
        Dict of capability_name -> True (only includes required capabilities)

    Example:
        >>> gradient_loss_reqs = {"has_gradients": True}
        >>> tv_loss_reqs = {"has_gradients": True, "has_auxiliary_knowledge": True}
        >>> optimizer_reqs = compute_requirements(gradient_loss_reqs, tv_loss_reqs)
        >>> # Result: {"has_gradients": True, "has_auxiliary_knowledge": True}

    """
    aggregated = {}

    for item in components:
        # Handle different input types
        if isinstance(item, dict):
            reqs = item
        elif hasattr(item, "get_metadata"):
            metadata = item.get_metadata()
            reqs = getattr(metadata, "required_capabilities", {})
        elif hasattr(item, "required_capabilities"):
            reqs = item.required_capabilities
        else:
            continue

        # Merge requirements
        for cap_name, is_required in reqs.items():
            if is_required:
                aggregated[cap_name] = True

    return aggregated


# =============================================================================
# Literature Threat Models (A-H)
# Paper Title: "SoK: Gradient Inversion Attacks in Federated Learning"
# Authors: Carletti et al.
# =============================================================================

def model_a_eavesdropper() -> ThreatModel:
    """Model A: Eavesdropper.

    Only standard FL information:
    - Model architecture (known)
    - Gradients (shared)
    - Number of samples per client
    """
    return ThreatModel(AttackerCapabilities(
        name="model_a",
        description="Eavesdropper: only standard FL information",
        has_gradients=True,
        # All others False
    ))


def model_b_informed() -> ThreatModel:
    """Model B: Informed Eavesdropper.

    Model A + basic auxiliary knowledge (e.g., TV regularization, data ranges).
    """
    return ThreatModel(AttackerCapabilities(
        name="model_b",
        description="Informed Eavesdropper: + auxiliary knowledge (TV reg, etc.)",
        has_gradients=True,
        has_auxiliary_knowledge=True,
    ))


def model_c_parameter_aware() -> ThreatModel:
    """Model C: Parameter-Aware Eavesdropper.

    Model B + knowledge of local training hyperparameters.
    """
    return ThreatModel(AttackerCapabilities(
        name="model_c",
        description="Parameter-Aware Eavesdropper: + training hyperparameters",
        has_gradients=True,
        has_auxiliary_knowledge=True,
        has_local_hyperparameters=True,
    ))


def model_d_data_enhanced() -> ThreatModel:
    """Model D: Data-Enhanced Eavesdropper.

    Model C + surrogate public dataset.
    """
    return ThreatModel(AttackerCapabilities(
        name="model_d",
        description="Data-Enhanced Eavesdropper: + surrogate dataset",
        has_gradients=True,
        has_auxiliary_knowledge=True,
        has_local_hyperparameters=True,
        has_surrogate_data=True,
    ))


def model_e_statistical() -> ThreatModel:
    """Model E: Statistical-Informed Eavesdropper.

    Model D + batch normalization statistics.
    """
    return ThreatModel(AttackerCapabilities(
        name="model_e",
        description="Statistical-Informed Eavesdropper: + BN statistics",
        has_gradients=True,
        has_auxiliary_knowledge=True,
        has_local_hyperparameters=True,
        has_surrogate_data=True,
        has_bn_statistics=True,
    ))


def model_f_active() -> ThreatModel:
    """Model F: Active Manipulator.

    Can modify model architecture for targeted extraction.
    """
    return ThreatModel(AttackerCapabilities(
        name="model_f",
        description="Active Manipulator: can modify architecture",
        has_gradients=True,
        has_auxiliary_knowledge=True,
        has_local_hyperparameters=True,
        can_modify_architecture=True,
    ))


def model_g_data_enhanced_active() -> ThreatModel:
    """Model G: Data-Enhanced Manipulator.

    Model F + surrogate dataset.
    """
    return ThreatModel(AttackerCapabilities(
        name="model_g",
        description="Data-Enhanced Manipulator: architecture + surrogate data",
        has_gradients=True,
        has_auxiliary_knowledge=True,
        has_local_hyperparameters=True,
        has_surrogate_data=True,
        can_modify_architecture=True,
    ))


def model_h_sybil() -> ThreatModel:
    """Model H: Active Client Manipulator.

    Model G + ability to inject Sybil clients.
    """
    return ThreatModel(AttackerCapabilities(
        name="model_h",
        description="Active Client Manipulator: can inject Sybil clients",
        has_gradients=True,
        has_auxiliary_knowledge=True,
        has_local_hyperparameters=True,
        has_surrogate_data=True,
        can_modify_architecture=True,
        can_inject_sybils=True,
    ))


# Registry
THREAT_MODELS = {
    "model_a": model_a_eavesdropper,
    "model_b": model_b_informed,
    "model_c": model_c_parameter_aware,
    "model_d": model_d_data_enhanced,
    "model_e": model_e_statistical,
    "model_f": model_f_active,
    "model_g": model_g_data_enhanced_active,
    "model_h": model_h_sybil,
}


def get_threat_model(name: str) -> ThreatModel:
    """Get threat model by name.

    Args:
        name: One of the registered threat model names

    Returns:
        ThreatModel instance

    Raises:
        ValueError: If name not found

    """
    if name not in THREAT_MODELS:
        available = ", ".join(THREAT_MODELS.keys())
        raise ValueError(f"Unknown threat model '{name}'. Available: {available}")

    return THREAT_MODELS[name]()
