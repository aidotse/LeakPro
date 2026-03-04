"""Stage 1 — Configure training parameters, select attacks, and optionally enable DP-SGD."""

from __future__ import annotations

import copy

import streamlit as st
import yaml


_AVAILABLE_ATTACKS = [
    ("rmia",  "RMIA — Relative Membership Inference Attack"),
    ("base",  "Base — Shadow-model baseline"),
    ("lira",  "LiRA — Likelihood Ratio Attack"),
    ("ramia", "RaMIA — Randomised Augmentation MIA"),
    ("qmia",  "QMIA — Quantile-based MIA"),
]


def render_configure() -> None:
    """Render the configuration stage."""
    from leakpro.ui.runner import LeakProRunner  # noqa: PLC0415

    st.title("⚙️  Stage 1 — Configure")
    st.caption("Adjust training and audit parameters, then click **Start Audit** to proceed.")

    runner = LeakProRunner()

    # Load defaults if not yet in session state (or reset to None by dashboard)
    if not st.session_state.get("train_config"):
        st.session_state.train_config = runner.default_train_config()
    if not st.session_state.get("audit_config"):
        st.session_state.audit_config = runner.default_audit_config()
    if "dpsgd_enabled" not in st.session_state:
        st.session_state.dpsgd_enabled = False
    if "dpsgd_params" not in st.session_state:
        st.session_state.dpsgd_params = {
            "target_epsilon": 3.5,
            "target_delta": 1e-5,
            "max_grad_norm": 1.2,
            "virtual_batch_size": 16,
        }

    tc = st.session_state.train_config
    ac = st.session_state.audit_config

    # ------------------------------------------------------------------ #
    # Training parameters
    # ------------------------------------------------------------------ #
    st.subheader("Training parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        tc["train"]["epochs"] = st.number_input(
            "Epochs", min_value=1, max_value=500, value=tc["train"]["epochs"]
        )
        tc["train"]["batch_size"] = st.number_input(
            "Batch size", min_value=8, max_value=2048, step=8, value=tc["train"]["batch_size"]
        )
    with col2:
        tc["train"]["learning_rate"] = st.number_input(
            "Learning rate", min_value=1e-5, max_value=1.0,
            value=float(tc["train"]["learning_rate"]), format="%.5f"
        )
        tc["train"]["weight_decay"] = st.number_input(
            "Weight decay", min_value=0.0, max_value=0.1,
            value=float(tc["train"]["weight_decay"]), format="%.6f"
        )
    with col3:
        tc["data"]["dataset"] = st.selectbox(
            "Dataset", ["cifar10", "cifar100"],
            index=0 if tc["data"]["dataset"] == "cifar10" else 1
        )
        tc["data"]["f_train"] = st.slider(
            "Train fraction", 0.1, 0.9, float(tc["data"]["f_train"]), 0.05
        )
        tc["run"]["random_seed"] = st.number_input(
            "Random seed", value=int(tc["run"].get("random_seed", 1236))
        )

    st.markdown("---")

    # ------------------------------------------------------------------ #
    # Attack selection
    # ------------------------------------------------------------------ #
    st.subheader("Attacks to run")
    st.caption("Select which membership inference attacks to include in the audit.")

    # Build a set of currently configured attacks
    current_attacks = {entry["attack"] for entry in ac.get("audit", {}).get("attack_list", [])}

    selected_attacks: list[dict] = []
    cols = st.columns(len(_AVAILABLE_ATTACKS))
    for col, (attack_id, label) in zip(cols, _AVAILABLE_ATTACKS):
        with col:
            checked = st.checkbox(label.split(" — ")[0], value=(attack_id in current_attacks))
            if checked:
                selected_attacks.append({"attack": attack_id})

    if not selected_attacks:
        st.warning("Select at least one attack.")

    # Update audit config with selected attacks
    if "audit" not in ac:
        ac["audit"] = {}
    ac["audit"]["attack_list"] = selected_attacks

    st.markdown("---")

    # ------------------------------------------------------------------ #
    # DP-SGD toggle
    # ------------------------------------------------------------------ #
    st.subheader("Differential Privacy (DP-SGD)")
    dp = st.session_state.dpsgd_params

    st.session_state.dpsgd_enabled = st.toggle(
        "Enable DP-SGD training",
        value=st.session_state.dpsgd_enabled,
        help="Train the target model with differential privacy using Opacus. "
             "Reduces privacy leakage at the cost of some model accuracy.",
    )

    if st.session_state.dpsgd_enabled:
        st.info(
            "Lower ε = stronger privacy guarantee, but lower model accuracy. "
            "Typical range: ε ∈ [1, 10], δ = 1e-5."
        )
        col_dp1, col_dp2, col_dp3 = st.columns(3)
        with col_dp1:
            dp["target_epsilon"] = st.slider(
                "Target ε (epsilon)", 0.5, 20.0, float(dp["target_epsilon"]), 0.5,
                help="Privacy budget. Lower means more private."
            )
        with col_dp2:
            dp["max_grad_norm"] = st.slider(
                "Max gradient norm", 0.1, 5.0, float(dp["max_grad_norm"]), 0.1,
                help="Gradient clipping threshold."
            )
        with col_dp3:
            dp["virtual_batch_size"] = st.number_input(
                "Virtual batch size", min_value=4, max_value=256,
                value=int(dp["virtual_batch_size"]),
                help="Physical batch size processed per step (memory optimisation)."
            )
        dp["target_delta"] = float(
            st.number_input(
                "Target δ (delta)", min_value=1e-8, max_value=1e-3,
                value=float(dp["target_delta"]), format="%.2e"
            )
        )
        st.session_state.dpsgd_params = dp
    else:
        st.caption("Standard SGD/Adam training — no differential privacy applied.")

    st.markdown("---")

    # ------------------------------------------------------------------ #
    # Navigation
    # ------------------------------------------------------------------ #
    col_back, _, col_fwd = st.columns([1, 3, 1])
    with col_back:
        if st.button("← Back", use_container_width=True):
            st.session_state.stage = 0
            st.rerun()
    with col_fwd:
        disabled = len(selected_attacks) == 0
        if st.button(
            "Start Audit →", type="primary", use_container_width=True, disabled=disabled
        ):
            st.session_state.train_config = tc
            st.session_state.audit_config = ac
            st.session_state.stage = 2
            st.rerun()
