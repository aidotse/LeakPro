"""Stage 1 — Configure training parameters, select attacks, and optionally enable DP-SGD."""

from __future__ import annotations

import streamlit as st

_AVAILABLE_ATTACKS = [
    ("rmia",  "RMIA — Relative Membership Inference Attack"),
    ("base",  "Base — Shadow-model baseline"),
    ("lira",  "LiRA — Likelihood Ratio Attack"),
    ("ramia", "RaMIA — Randomised Augmentation MIA"),
    ("qmia",  "QMIA — Quantile-based MIA"),
]

_DEFAULT_DPSGD_PARAMS: dict = {
    "target_epsilon": 3.5,
    "target_delta": 1e-5,
    "max_grad_norm": 1.2,
    "virtual_batch_size": 16,
}


def render_configure() -> None:
    """Render the configuration stage."""
    from leakpro.ui.runner import LeakProRunner  # noqa: PLC0415

    st.title("⚙️  Stage 1 — Configure")
    st.caption("Adjust training and audit parameters, then click **Start Audit** to proceed.")

    runner = LeakProRunner()

    if not st.session_state.get("train_config"):
        st.session_state.train_config = runner.default_train_config()
    if not st.session_state.get("audit_config"):
        st.session_state.audit_config = runner.default_audit_config()
    if "dpsgd_enabled" not in st.session_state:
        st.session_state.dpsgd_enabled = False
    if not st.session_state.get("dpsgd_params"):
        st.session_state.dpsgd_params = dict(_DEFAULT_DPSGD_PARAMS)

    tc = st.session_state.train_config
    ac = st.session_state.audit_config
    reaudit = st.session_state.get("reaudit_mode", False)

    if reaudit:
        st.info("Re-audit mode — the trained model will be reused. Only attack selection and parameters can be changed.")
    else:
        st.subheader("Training parameters")
        _render_training_params(tc)
        st.markdown("---")

    st.subheader("Attacks to run")
    st.caption("Select which membership inference attacks to include in the audit.")
    selected_attacks = _render_attack_selection(ac)
    if not selected_attacks:
        st.warning("Select at least one attack.")
    ac.setdefault("audit", {})["attack_list"] = selected_attacks
    st.markdown("---")

    if not reaudit:
        st.subheader("Differential Privacy (DP-SGD)")
        _render_dpsgd_section()
        st.markdown("---")

    col_back, _, col_fwd = st.columns([1, 3, 1])
    with col_back:
        if st.button("← Back", use_container_width=True):
            st.session_state.pop("reaudit_mode", None)
            st.session_state.stage = 0
            st.rerun()
    with col_fwd:
        next_stage = 3 if reaudit else 2
        label = "Run Attacks →" if reaudit else "Start Audit →"
        if st.button(
            label, type="primary",
            use_container_width=True, disabled=(len(selected_attacks) == 0),
        ):
            st.session_state.train_config = tc
            st.session_state.audit_config = ac
            st.session_state.stage = next_stage
            st.rerun()


def _render_training_params(tc: dict) -> None:
    """Render training hyper-parameter inputs and mutate tc in place."""
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
            value=float(tc["train"]["learning_rate"]), format="%.5f",
        )
        tc["train"]["weight_decay"] = st.number_input(
            "Weight decay", min_value=0.0, max_value=0.1,
            value=float(tc["train"]["weight_decay"]), format="%.6f",
        )
    with col3:
        tc["data"]["dataset"] = st.selectbox(
            "Dataset", ["cifar10", "cifar100"],
            index=0 if tc["data"]["dataset"] == "cifar10" else 1,
        )
        tc["data"]["f_train"] = st.slider(
            "Train fraction", 0.1, 0.9, float(tc["data"]["f_train"]), 0.05
        )
        tc["run"]["random_seed"] = st.number_input(
            "Random seed", value=int(tc["run"].get("random_seed", 1236))
        )


def _render_attack_selection(ac: dict) -> list[dict]:
    """Render attack checkboxes + per-attack parameter expanders."""
    current_list = ac.get("audit", {}).get("attack_list", [])
    # Build a lookup of existing params keyed by attack id (last entry wins for duplicates)
    current_params: dict[str, dict] = {}
    for entry in current_list:
        current_params[entry["attack"]] = entry

    selected: list[dict] = []

    # ── checkboxes ──────────────────────────────────────────────────────────
    cols = st.columns(len(_AVAILABLE_ATTACKS))
    enabled: dict[str, bool] = {}
    for col, (attack_id, label) in zip(cols, _AVAILABLE_ATTACKS):
        with col:
            enabled[attack_id] = st.checkbox(
                label.split(" — ")[0],
                value=(attack_id in current_params),
            )

    # ── per-attack parameter expanders ──────────────────────────────────────
    for attack_id, label in _AVAILABLE_ATTACKS:
        if not enabled[attack_id]:
            continue
        prev = current_params.get(attack_id, {})
        entry: dict = {"attack": attack_id}

        with st.expander(f"⚙ {label.split(' — ')[0]} parameters", expanded=False):
            if attack_id in ("rmia", "base"):
                c1, c2, c3 = st.columns(3)
                entry["num_shadow_models"] = c1.number_input(
                    "Shadow models", min_value=1, max_value=32,
                    value=int(prev.get("num_shadow_models", 2)),
                    key=f"{attack_id}_nshadow",
                )
                entry["training_data_fraction"] = c2.slider(
                    "Train data fraction", 0.1, 1.0,
                    float(prev.get("training_data_fraction", 0.5)), 0.05,
                    key=f"{attack_id}_frac",
                )
                entry["online"] = c3.checkbox(
                    "Online mode",
                    value=bool(prev.get("online", False)),
                    key=f"{attack_id}_online",
                    help="Online RMIA/BASE trains one shadow model per audit point — much slower but more accurate.",
                )

            elif attack_id == "lira":
                c1, c2 = st.columns(2)
                entry["num_shadow_models"] = c1.number_input(
                    "Shadow models", min_value=1, max_value=64,
                    value=int(prev.get("num_shadow_models", 4)),
                    key="lira_nshadow",
                )
                entry["online"] = c2.checkbox(
                    "Online mode", value=bool(prev.get("online", False)),
                    key="lira_online",
                )

            elif attack_id == "ramia":
                c1, c2, c3 = st.columns(3)
                entry["num_transforms"] = c1.number_input(
                    "Num transforms", min_value=0, max_value=32,
                    value=int(prev.get("num_transforms", 1)),
                    key="ramia_ntrans",
                )
                entry["n_ops"] = c2.number_input(
                    "Ops per transform", min_value=0, max_value=8,
                    value=int(prev.get("n_ops", 1)),
                    key="ramia_nops",
                )
                entry["augment_strength"] = c3.selectbox(
                    "Augment strength",
                    ["easy", "medium", "strong"],
                    index=["easy", "medium", "strong"].index(
                        prev.get("augment_strength", "strong")
                    ),
                    key="ramia_strength",
                )

            elif attack_id == "qmia":
                entry["num_shadow_models"] = st.number_input(
                    "Shadow models", min_value=1, max_value=32,
                    value=int(prev.get("num_shadow_models", 4)),
                    key="qmia_nshadow",
                )

        selected.append(entry)

    return selected


def _render_dpsgd_section() -> None:
    """Render the DP-SGD toggle and parameter sliders."""
    dp = st.session_state.dpsgd_params

    st.session_state.dpsgd_enabled = st.toggle(
        "Enable DP-SGD training",
        value=st.session_state.dpsgd_enabled,
        help="Train the target model with differential privacy using Opacus.",
    )

    if not st.session_state.dpsgd_enabled:
        st.caption("Standard SGD/Adam training — no differential privacy applied.")
        return

    st.info("Lower ε = stronger privacy guarantee, but lower model accuracy. Typical range: ε ∈ [1, 10], δ = 1e-5.")
    col1, col2, col3 = st.columns(3)
    with col1:
        dp["target_epsilon"] = st.slider(
            "Target ε (epsilon)", 0.5, 20.0, float(dp["target_epsilon"]), 0.5,
            help="Privacy budget. Lower means more private.",
        )
    with col2:
        dp["max_grad_norm"] = st.slider(
            "Max gradient norm", 0.1, 5.0, float(dp["max_grad_norm"]), 0.1,
            help="Gradient clipping threshold.",
        )
    with col3:
        dp["virtual_batch_size"] = st.number_input(
            "Virtual batch size", min_value=4, max_value=256, value=int(dp["virtual_batch_size"]),
            help="Physical batch size per step (memory optimisation).",
        )
    dp["target_delta"] = float(
        st.number_input(
            "Target δ (delta)", min_value=1e-8, max_value=1e-3,
            value=float(dp["target_delta"]), format="%.2e",
        )
    )
    st.session_state.dpsgd_params = dp
