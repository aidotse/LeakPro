"""Stage 2 — Train the target model and display training metrics."""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go


def render_train() -> None:
    """Render the training stage."""
    from leakpro.ui.runner import LeakProRunner  # noqa: PLC0415

    st.title("🏋️  Stage 2 — Train Target Model")
    dpsgd = st.session_state.get("dpsgd_enabled", False)
    mode_label = "DP-SGD (differential privacy)" if dpsgd else "standard (no privacy)"
    st.caption(f"Training mode: **{mode_label}**")

    train_config = st.session_state.train_config
    runner = LeakProRunner()

    # ------------------------------------------------------------------ #
    # Step 2a — Prepare dataset
    # ------------------------------------------------------------------ #
    st.subheader("Step 1 — Prepare Dataset")

    if "data_result" not in st.session_state:
        if st.button("Download & Prepare Data", type="primary"):
            log_container = st.empty()
            status_msg = st.empty()
            with st.spinner("Preparing dataset…"):
                try:
                    data_result = runner.prepare_data(train_config, log_container=status_msg)
                    st.session_state.data_result = data_result
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Data preparation failed: {e}")
    else:
        dr = st.session_state.data_result
        st.success(
            f"Dataset ready — {len(dr['train_indices'])} train / "
            f"{len(dr['test_indices'])} test samples ({dr['dataset_name']})."
        )

    # ------------------------------------------------------------------ #
    # Step 2b — Train model
    # ------------------------------------------------------------------ #
    if "data_result" in st.session_state:
        st.markdown("---")
        st.subheader("Step 2 — Train Model")

        if "train_result_dict" not in st.session_state:
            if st.button("Train Model", type="primary"):
                log_placeholder = st.empty()
                with st.spinner("Training… this may take several minutes."):
                    try:
                        if dpsgd:
                            result = runner.train_dpsgd(
                                train_config,
                                st.session_state.dpsgd_params,
                                st.session_state.data_result,
                                log_container=log_placeholder,
                            )
                        else:
                            result = runner.train_standard(
                                train_config,
                                st.session_state.data_result,
                                log_container=log_placeholder,
                            )
                        st.session_state.train_result_dict = result
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Training failed: {e}")
        else:
            _show_training_metrics(st.session_state.train_result_dict, dpsgd)

    # ------------------------------------------------------------------ #
    # Navigation
    # ------------------------------------------------------------------ #
    if "train_result_dict" in st.session_state:
        st.markdown("---")
        col_back, _, col_fwd = st.columns([1, 3, 1])
        with col_back:
            if st.button("← Reconfigure", use_container_width=True):
                # Clear training results when going back
                st.session_state.pop("train_result_dict", None)
                st.session_state.pop("data_result", None)
                st.session_state.stage = 1
                st.rerun()
        with col_fwd:
            if st.button("Run Attacks →", type="primary", use_container_width=True):
                st.session_state.stage = 3
                st.rerun()
    elif "data_result" not in st.session_state:
        st.markdown("---")
        col_back, _ = st.columns([1, 4])
        with col_back:
            if st.button("← Reconfigure", use_container_width=True):
                st.session_state.stage = 1
                st.rerun()


def _show_training_metrics(result: dict, dpsgd: bool) -> None:
    train_result = result["train_result"]
    test_result = result["test_result"]

    train_acc_history = train_result.metrics.extra.get("accuracy_history", [])
    train_loss_history = train_result.metrics.extra.get("loss_history", [])

    # Summary cards
    cols = st.columns(4)
    cols[0].metric("Final Train Accuracy", f"{train_result.metrics.accuracy:.3f}")
    cols[1].metric("Test Accuracy", f"{test_result.accuracy:.3f}")
    cols[2].metric("Final Train Loss", f"{train_result.metrics.loss:.4f}")
    if dpsgd and "dpsgd_params" in result:
        cols[3].metric("Target ε", f"{result['dpsgd_params']['target_epsilon']}")

    # Training curves
    if train_acc_history:
        epochs = list(range(1, len(train_acc_history) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=train_acc_history, name="Train Accuracy",
                                 line=dict(color="#4C9BE8")))
        fig.add_scatter(x=[epochs[-1]], y=[test_result.accuracy],
                        mode="markers", name="Test Accuracy",
                        marker=dict(color="#E84C4C", size=10, symbol="star"))
        fig.update_layout(title="Accuracy over Epochs", xaxis_title="Epoch",
                          yaxis_title="Accuracy", height=300, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

    if train_loss_history:
        epochs = list(range(1, len(train_loss_history) + 1))
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=epochs, y=train_loss_history, name="Train Loss",
                                  line=dict(color="#F5A623")))
        fig2.update_layout(title="Loss over Epochs", xaxis_title="Epoch",
                           yaxis_title="Loss", height=300, margin=dict(t=40))
        st.plotly_chart(fig2, use_container_width=True)

    st.success(f"Model saved to: `{result['target_folder']}`")
