"""Stage 1 — Per-model configuration: training parameters, attacks, and DP-SGD settings."""

from __future__ import annotations

import copy

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

_MAX_MODELS = 5


def _default_model(name: str, train_config: dict) -> dict:
    """Return a fresh model spec dict with default values."""
    return {
        "name": name,
        "dpsgd_enabled": False,
        "dpsgd_params": None,
        "target_folder": "",
        "train_config": copy.deepcopy(train_config),
        "attack_list": [],
        "train_result_dict": None,
        "audit_results": None,
    }


def render_configure() -> None:
    """Render the configuration stage."""
    from leakpro.ui.runner import LeakProRunner  # noqa: PLC0415

    st.title("⚙️  Stage 1 — Configure")

    runner = LeakProRunner()
    default_tc = runner.default_train_config()

    reaudit = st.session_state.get("reaudit_mode", False)

    if reaudit:
        st.info("Re-audit mode — trained models will be reused. Only attack selection can be changed.")
        _render_reaudit(default_tc)
        return

    config_phase = st.session_state.get("config_phase", 1)

    if config_phase == 1:
        _render_phase1(default_tc)
    else:
        _render_phase2(default_tc)


# ---------------------------------------------------------------------------
# Phase 1 — model count selector
# ---------------------------------------------------------------------------

def _render_phase1(default_tc: dict) -> None:
    """Render the model count selection screen."""
    st.caption("First, decide how many models to train and compare in this audit.")

    st.markdown("---")
    st.subheader("How many models?")
    st.caption(
        "Each model is trained independently and audited with its own attacks. "
        "Add a DP-SGD model alongside a standard one to compare privacy-accuracy trade-offs."
    )

    current_count = len(st.session_state.get("models") or [])
    default_count = current_count if 1 <= current_count <= _MAX_MODELS else 1

    model_count = st.number_input(
        "Number of models", min_value=1, max_value=_MAX_MODELS,
        value=default_count, step=1,
    )

    st.markdown("---")
    col_back, _, col_fwd = st.columns([1, 3, 1])
    with col_back:
        if st.button("← Back", use_container_width=True):
            st.session_state.pop("reaudit_mode", None)
            st.session_state.stage = 0
            st.rerun()
    with col_fwd:
        if st.button("Configure Models →", type="primary", use_container_width=True):
            _init_models(int(model_count), default_tc)
            st.session_state.config_phase = 2
            st.rerun()


def _init_models(count: int, default_tc: dict) -> None:
    """Initialise the models list to `count` entries, preserving existing configs."""
    existing = st.session_state.get("models") or []
    models: list[dict] = []
    for i in range(count):
        if i < len(existing):
            # Backfill new keys for models created before this redesign
            m = existing[i]
            if "train_config" not in m:
                m["train_config"] = copy.deepcopy(default_tc)
            if "attack_list" not in m:
                m["attack_list"] = []
            models.append(m)
        else:
            name = "Standard" if i == 0 else f"Model {i + 1}"
            models.append(_default_model(name, default_tc))
    st.session_state.models = models


# ---------------------------------------------------------------------------
# Phase 2 — per-model tabs
# ---------------------------------------------------------------------------

def _render_phase2(default_tc: dict) -> None:  # noqa: ARG001
    """Render per-model configuration tabs."""
    models: list[dict] = st.session_state.get("models") or []
    if not models:
        st.warning("No models found. Please go back.")
        return

    st.caption(
        f"Configure each of your {len(models)} model(s) independently — "
        "training parameters, attacks, and DP-SGD settings."
    )

    tab_labels = [f"{i + 1}. {m['name']}" for i, m in enumerate(models)]
    tabs = st.tabs(tab_labels)

    all_valid = True
    for tab, model, i in zip(tabs, models, range(len(models))):
        with tab:
            _render_model_tab(model, i)
            if not model.get("attack_list"):
                all_valid = False

    if not all_valid:
        st.warning("Each model must have at least one attack selected.")

    st.markdown("---")
    col_back, _, col_fwd = st.columns([1, 3, 1])
    with col_back:
        if st.button("← Change model count", use_container_width=True):
            st.session_state.config_phase = 1
            st.rerun()
    with col_fwd:
        if st.button(
            "Start Audit →", type="primary", use_container_width=True,
            disabled=not all_valid,
        ):
            _assign_target_folders()
            st.session_state.models = models
            st.session_state.stage = 2
            st.rerun()


def _render_model_tab(model: dict, i: int) -> None:
    """Render the full configuration for one model inside its tab."""
    # --- Name ---
    model["name"] = st.text_input("Model name", model["name"], key=f"m_name_{i}")

    st.markdown("---")

    # --- Training parameters ---
    st.subheader("Training parameters")
    _render_training_params(model["train_config"], i)

    st.markdown("---")

    # --- Attack selection ---
    st.subheader("Attacks to run")
    st.caption("Select which membership inference attacks to include for this model.")
    model["attack_list"] = _render_attack_selection(model.get("attack_list", []), prefix=f"m{i}")
    if not model["attack_list"]:
        st.warning("Select at least one attack.")

    st.markdown("---")

    # --- DP-SGD ---
    st.subheader("Differential Privacy (DP-SGD)")
    _render_dpsgd_section(model, i)


def _render_dpsgd_section(model: dict, i: int) -> None:
    """Render DP-SGD toggle and parameter sliders for one model."""
    model["dpsgd_enabled"] = st.toggle(
        "Enable DP-SGD training",
        value=model["dpsgd_enabled"],
        key=f"m_dpsgd_{i}",
        help="Train with differential privacy (Opacus). Lower ε = stronger privacy, lower accuracy.",
    )

    if model["dpsgd_enabled"]:
        if model["dpsgd_params"] is None:
            model["dpsgd_params"] = dict(_DEFAULT_DPSGD_PARAMS)
        dp = model["dpsgd_params"]
        st.info("Lower ε = stronger privacy guarantee, but lower accuracy. Typical range: ε ∈ [1, 10].")
        c1, c2, c3 = st.columns(3)
        dp["target_epsilon"] = c1.slider(
            "Target ε", 0.5, 20.0, float(dp["target_epsilon"]), 0.5, key=f"m_eps_{i}",
            help="Privacy budget.",
        )
        dp["max_grad_norm"] = c2.slider(
            "Max grad norm", 0.1, 5.0, float(dp["max_grad_norm"]), 0.1, key=f"m_clip_{i}",
            help="Gradient clipping threshold (C).",
        )
        dp["virtual_batch_size"] = c3.number_input(
            "Virtual batch size", min_value=4, max_value=256,
            value=int(dp["virtual_batch_size"]), key=f"m_vbs_{i}",
        )
        dp["target_delta"] = float(st.number_input(
            "Target δ", min_value=1e-8, max_value=1e-3,
            value=float(dp["target_delta"]), format="%.2e", key=f"m_delta_{i}",
        ))
    else:
        model["dpsgd_params"] = None
        st.caption("Standard SGD/Adam training — no differential privacy applied.")


# ---------------------------------------------------------------------------
# Re-audit mode (attack-only reconfiguration)
# ---------------------------------------------------------------------------

def _render_reaudit(default_tc: dict) -> None:  # noqa: ARG001
    """Render re-audit mode: only attack lists can be changed per model."""
    models: list[dict] = st.session_state.get("models") or []
    if not models:
        st.warning("No trained models found in session.")
        return

    tab_labels = [f"{i + 1}. {m['name']}" for i, m in enumerate(models)]
    tabs = st.tabs(tab_labels)

    all_valid = True
    for tab, model, i in zip(tabs, models, range(len(models))):
        with tab:
            st.caption(f"Model: **{model['name']}** — select attacks to re-run.")
            model["attack_list"] = _render_attack_selection(
                model.get("attack_list", []), prefix=f"ra{i}"
            )
            if not model.get("attack_list"):
                st.warning("Select at least one attack.")
                all_valid = False

    st.markdown("---")
    col_back, _, col_fwd = st.columns([1, 3, 1])
    with col_back:
        if st.button("← Back", use_container_width=True):
            st.session_state.pop("reaudit_mode", None)
            st.session_state.stage = 0
            st.rerun()
    with col_fwd:
        if st.button(
            "Run Attacks →", type="primary", use_container_width=True, disabled=not all_valid,
        ):
            st.session_state.models = models
            st.session_state.pop("reaudit_mode", None)
            st.session_state.stage = 3
            st.rerun()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assign_target_folders() -> None:
    """Set unique target_folder names for each model based on type."""
    std_count = dp_count = 0
    for m in st.session_state.models:
        if m["dpsgd_enabled"]:
            m["target_folder"] = f"target_dpsgd_{dp_count}"
            dp_count += 1
        else:
            m["target_folder"] = f"target_model_{std_count}"
            std_count += 1


def _render_training_params(tc: dict, model_idx: int) -> None:
    """Render training hyper-parameter inputs and mutate tc in place."""
    col1, col2, col3 = st.columns(3)
    with col1:
        tc["train"]["epochs"] = st.number_input(
            "Epochs", min_value=1, max_value=500, value=tc["train"]["epochs"],
            key=f"tc_epochs_{model_idx}",
        )
        tc["train"]["batch_size"] = st.number_input(
            "Batch size", min_value=8, max_value=2048, step=8, value=tc["train"]["batch_size"],
            key=f"tc_batch_{model_idx}",
        )
    with col2:
        tc["train"]["learning_rate"] = st.number_input(
            "Learning rate", min_value=1e-5, max_value=1.0,
            value=float(tc["train"]["learning_rate"]), format="%.5f",
            key=f"tc_lr_{model_idx}",
        )
        tc["train"]["weight_decay"] = st.number_input(
            "Weight decay", min_value=0.0, max_value=0.1,
            value=float(tc["train"]["weight_decay"]), format="%.6f",
            key=f"tc_wd_{model_idx}",
        )
    with col3:
        tc["data"]["dataset"] = st.selectbox(
            "Dataset", ["cifar10", "cifar100"],
            index=0 if tc["data"]["dataset"] == "cifar10" else 1,
            key=f"tc_ds_{model_idx}",
        )
        tc["data"]["f_train"] = st.slider(
            "Train fraction", 0.1, 0.9, float(tc["data"]["f_train"]), 0.05,
            key=f"tc_ftrain_{model_idx}",
        )
        tc["run"]["random_seed"] = st.number_input(
            "Random seed", value=int(tc["run"].get("random_seed", 1236)),
            key=f"tc_seed_{model_idx}",
        )


def _render_attack_selection(current_list: list, prefix: str = "") -> list[dict]:
    """Render attack checkboxes + per-attack parameter expanders."""
    current_params: dict[str, dict] = {entry["attack"]: entry for entry in current_list}

    selected: list[dict] = []

    cols = st.columns(len(_AVAILABLE_ATTACKS))
    enabled: dict[str, bool] = {}
    for col, (attack_id, label) in zip(cols, _AVAILABLE_ATTACKS):
        with col:
            enabled[attack_id] = st.checkbox(
                label.split(" — ")[0],
                value=(attack_id in current_params),
                key=f"{prefix}_{attack_id}_cb",
            )

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
                    key=f"{prefix}_{attack_id}_nshadow",
                )
                entry["training_data_fraction"] = c2.slider(
                    "Train data fraction", 0.1, 1.0,
                    float(prev.get("training_data_fraction", 0.5)), 0.05,
                    key=f"{prefix}_{attack_id}_frac",
                )
                entry["online"] = c3.checkbox(
                    "Online mode", value=bool(prev.get("online", False)),
                    key=f"{prefix}_{attack_id}_online",
                    help="Online mode trains one shadow model per audit point — slower but more accurate.",
                )

            elif attack_id == "lira":
                c1, c2 = st.columns(2)
                entry["num_shadow_models"] = c1.number_input(
                    "Shadow models", min_value=1, max_value=64,
                    value=int(prev.get("num_shadow_models", 4)),
                    key=f"{prefix}_{attack_id}_nshadow",
                )
                entry["online"] = c2.checkbox(
                    "Online mode", value=bool(prev.get("online", False)),
                    key=f"{prefix}_{attack_id}_online",
                )

            elif attack_id == "ramia":
                c1, c2, c3 = st.columns(3)
                entry["num_transforms"] = c1.number_input(
                    "Num transforms", min_value=0, max_value=32,
                    value=int(prev.get("num_transforms", 1)),
                    key=f"{prefix}_{attack_id}_ntrans",
                )
                entry["n_ops"] = c2.number_input(
                    "Ops per transform", min_value=0, max_value=8,
                    value=int(prev.get("n_ops", 1)),
                    key=f"{prefix}_{attack_id}_nops",
                )
                entry["augment_strength"] = c3.selectbox(
                    "Augment strength", ["easy", "medium", "strong"],
                    index=["easy", "medium", "strong"].index(
                        prev.get("augment_strength", "strong")
                    ),
                    key=f"{prefix}_{attack_id}_strength",
                )

            elif attack_id == "qmia":
                entry["num_shadow_models"] = st.number_input(
                    "Shadow models", min_value=1, max_value=32,
                    value=int(prev.get("num_shadow_models", 4)),
                    key=f"{prefix}_{attack_id}_nshadow",
                )

        selected.append(entry)

    return selected
