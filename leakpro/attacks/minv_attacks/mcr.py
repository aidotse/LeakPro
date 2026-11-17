"""Membership-Calibrated Reconstruction (MCR) attack on tabular data.

Reconstructs training records from a target model by searching a conditional
generator's latent space, where the search is narrowed at two levels:

    L1 (class)      - pseudo-label conditioning (idea borrowed from PLG-MI)
    L2 (individual) - known-feature constraint (sensitive mode) OR
                      sparse sub-cluster targeting (outlier mode)

The L1 <-> L2 balance is arbitrated *inside the objective* by a membership-
inference calibration signal (RMIA-style population of reference models), so the
search pulls toward individual-specific features only where the target model
actually memorized.

Design doc: plans/tabular-reconstruction-attack-design.md (in the workspace repo)

Status: STUB. Interface + structure only; method bodies are not implemented.
"""
from typing import Any, Dict, Optional

import torch
from pydantic import BaseModel, Field

from leakpro.attacks.minv_attacks.abstract_minv import AbstractMINV
from leakpro.input_handler.minv_handler import MINVHandler
from leakpro.reporting.minva_result import MinvResult
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class AttackMCR(AbstractMINV):
    """Membership-Calibrated Reconstruction attack (tabular, white-box)."""

    class AttackConfig(BaseModel):
        """Configuration for the MCR attack."""

        # --- General ---
        batch_size: int = Field(32, ge=1, description="Batch size for training/evaluation")
        mode: str = Field("sensitive", description="Reconstruction mode: 'sensitive' or 'outlier'")

        # --- L1: pseudo-labeling (search-space restriction by class) ---
        top_n: int = Field(10, ge=1, description="Pseudo-labels selected per class")

        # --- Latent search ---
        dim_z: int = Field(32, ge=1, description="Latent dimension (tabular: likely << image default of 128)")
        z_optimization_iter: int = Field(1000, ge=1, description="Iterations for latent optimization")
        z_optimization_lr: float = Field(2e-4, ge=0.0, description="Learning rate for latent optimization")

        # --- L2: individual restriction ---
        lambda_known: float = Field(1.0, ge=0.0, description="Weight on the known-feature (x_k) constraint (sensitive mode)")

        # --- MIA calibration (reference-model population) ---
        n_reference_models: int = Field(64, ge=1, description="Reference models for RMIA-style calibration")

        # --- Generator ---
        generator: Dict[str, Any] = Field(default_factory=dict, description="Generator (CustomCTGAN) configuration")

    def __init__(self: Self, handler: MINVHandler, configs: dict) -> None:
        """Initialize the MCR attack."""
        logger.info("Configuring MCR attack")
        self.configs = self.AttackConfig() if configs is None else self.AttackConfig(**configs)
        super().__init__(handler)
        for key, value in self.configs.model_dump().items():
            setattr(self, key, value)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # TODO: self.num_classes = self.handler.get_num_classes()

    def description(self: Self) -> dict:
        """Return a description of the attack."""
        return {
            "title_str": "Membership-Calibrated Reconstruction (MCR)",
            "reference": "Design doc: plans/tabular-reconstruction-attack-design.md",
            "summary": "Tabular data reconstruction via two-level latent search "
                       "(pseudo-label class restriction + individual restriction) "
                       "calibrated by membership-inference statistics.",
            "detailed": "White-box reconstruction attack. Trains a conditional generator on "
                        "pseudo-labeled public data (L1), then searches the latent space for "
                        "records the target model memorized (L2), using an RMIA-style membership "
                        "likelihood ratio as the in-loop arbiter and targeting signal. Two modes: "
                        "sensitive-attribute reconstruction (known x_k given) and outlier reconstruction.",
        }

    # ------------------------------------------------------------------ #
    # L1: pseudo-labeling                                                 #
    # ------------------------------------------------------------------ #
    def pseudo_label_selection(self: Self) -> None:
        """L1 restriction: top-n pseudo-label selection on public data via the target model."""
        raise NotImplementedError("TODO: port + clean top_n_selection from the old plgmi.py")

    # ------------------------------------------------------------------ #
    # MIA bridge: reference-model population + calibrated membership      #
    # ------------------------------------------------------------------ #
    def build_reference_population(self: Self) -> None:
        """Train/load the reference-model population for RMIA-style calibration.

        Cheap for tabular (models are small) - this is the 'tabular is the right
        place for this' argument.
        """
        raise NotImplementedError("TODO: wire to main's RMIA / shadow_model_handler")

    def membership_score(self: Self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calibrated membership likelihood ratio Lambda(x) (RMIA population variant).

        Used both as the per-target L1<->L2 throttle and as the outlier targeting signal.
        """
        raise NotImplementedError("TODO: RMIA-style population likelihood ratio")

    # ------------------------------------------------------------------ #
    # Attack lifecycle                                                    #
    # ------------------------------------------------------------------ #
    def prepare_attack(self: Self) -> None:
        """Train the conditional generator (CustomCTGAN) and reference population."""
        raise NotImplementedError("TODO: pseudo-labeling -> train CustomCTGAN -> build reference population")

    def run_attack(self: Self) -> MinvResult:
        """Run reconstruction in the configured mode and return results."""
        if self.mode == "sensitive":
            return self._run_sensitive_mode()
        if self.mode == "outlier":
            return self._run_outlier_mode()
        raise ValueError(f"Unknown mode: {self.mode}")

    # ------------------------------------------------------------------ #
    # L2 mode A: sensitive-attribute reconstruction                      #
    # ------------------------------------------------------------------ #
    def _run_sensitive_mode(self: Self) -> MinvResult:
        """Recover x_s for known members given x_k.

        argmin_z  -log Lambda(G(z,c))  +  lambda_known * || Pi_k(G(z,c)) - x_k ||^2
        then read off  x_s = Pi_s(G(z*, c)).
        """
        raise NotImplementedError("TODO: constrained latent search (see Exp 1 for the reachability core)")

    # ------------------------------------------------------------------ #
    # L2 mode B: outlier reconstruction                                  #
    # ------------------------------------------------------------------ #
    def _run_outlier_mode(self: Self) -> MinvResult:
        """Recover atypical / highly-memorized members.

        argmin_z  -log Lambda(G(z,c)),  restricted to sparse sub-clusters.
        """
        raise NotImplementedError("TODO: membership-ranked latent search over sparse clusters")

    # ------------------------------------------------------------------ #
    # Shared: latent search primitive                                     #
    # ------------------------------------------------------------------ #
    def optimize_z(
        self: Self,
        y: torch.Tensor,
        x_k: Optional[torch.Tensor] = None,
        iter_times: int = 1000,
    ) -> torch.Tensor:
        """White-box latent optimization through the differentiable CTGAN path.

        Uses generator.forward_fakeact_with_labels + generator._pack_for_gandalf so
        gradients flow from the target model back to z. When x_k is given, adds the
        L2 known-feature constraint.
        """
        raise NotImplementedError("TODO: implement; Exp 1 prototypes the x_k-constraint core")
