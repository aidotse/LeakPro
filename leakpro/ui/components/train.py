"""Stage 2 — Train the target model and display training metrics."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st


def render_train() -> None:
    """Render the training stage."""
    from leakpro.ui.runner import LeakProRunner  # noqa: PLC0415

    st.title("🏋️  Stage 2 — Train Target Model")
    dpsgd = st.session_state.get("dpsgd_enabled", False)
    mode_label = "DP-SGD (differential privacy)" if dpsgd else "standard (no privacy)"
    st.caption(f"Training mode: **{mode_label}**")

    train_config = st.session_state.train_config
    runner = LeakProRunner()

    st.subheader("Step 1 — Prepare Dataset")
    _render_data_prep(runner, train_config)

    if st.session_state.get("data_result"):
        st.markdown("---")
        st.subheader("Step 2 — Train Model")
        _render_model_training(runner, train_config, dpsgd)

    _render_navigation()


def _render_data_prep(runner: object, train_config: dict) -> None:
    """Render dataset download / status section."""
    # Always allow re-preparing data so stale session indices don't carry over.
    if st.session_state.get("data_result"):
        dr = st.session_state.data_result
        st.success(
            f"Dataset ready — {len(dr['train_indices'])} train / "
            f"{len(dr['test_indices'])} test samples ({dr['dataset_name']})."
        )
        if st.button("Re-prepare Data"):
            st.session_state.pop("data_result", None)
            st.session_state.pop("train_result_dict", None)
            st.rerun()
        return

    if st.button("Download & Prepare Data", type="primary"):
        status_msg = st.empty()
        with st.spinner("Preparing dataset…"):
            try:
                data_result = runner.prepare_data(train_config, log_container=status_msg)  # type: ignore[attr-defined]
                st.session_state.data_result = data_result
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Data preparation failed: {e}")


def _render_model_training(runner: object, train_config: dict, dpsgd: bool) -> None:
    """Render model training button and metrics."""
    if not st.session_state.get("train_result_dict"):
        if st.button("Train Model", type="primary"):
            log_placeholder = st.empty()
            with st.spinner("Training… this may take several minutes."):
                try:
                    if dpsgd:
                        result = runner.train_dpsgd(  # type: ignore[attr-defined]
                            train_config,
                            st.session_state.dpsgd_params,
                            st.session_state.data_result,
                            log_container=log_placeholder,
                        )
                    else:
                        result = runner.train_standard(  # type: ignore[attr-defined]
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


def _render_navigation() -> None:
    """Render back / forward navigation buttons."""
    if st.session_state.get("train_result_dict"):
        st.markdown("---")
        col_back, _, col_fwd = st.columns([1, 3, 1])
        with col_back:
            if st.button("← Reconfigure", use_container_width=True):
                st.session_state.pop("train_result_dict", None)
                st.session_state.pop("data_result", None)
                st.session_state.stage = 1
                st.rerun()
        with col_fwd:
            if st.button("Run Attacks →", type="primary", use_container_width=True):
                st.session_state.stage = 3
                st.rerun()
    elif not st.session_state.get("data_result"):
        st.markdown("---")
        col_back, _ = st.columns([1, 4])
        with col_back:
            if st.button("← Reconfigure", use_container_width=True):
                st.session_state.stage = 1
                st.rerun()


def _show_training_metrics(result: dict, dpsgd: bool) -> None:
    """Render post-training accuracy / loss charts and summary cards."""
    train_result = result["train_result"]
    test_result = result["test_result"]
    train_acc_history = train_result.metrics.extra.get("accuracy_history", [])
    train_loss_history = train_result.metrics.extra.get("loss_history", [])

    cols = st.columns(4)
    cols[0].metric("Final Train Accuracy", f"{train_result.metrics.accuracy:.3f}")
    cols[1].metric("Test Accuracy", f"{test_result.accuracy:.3f}")
    cols[2].metric("Final Train Loss", f"{train_result.metrics.loss:.4f}")
    if dpsgd and "dpsgd_params" in result:
        cols[3].metric("Target ε", f"{result['dpsgd_params']['target_epsilon']}")

    if train_acc_history:
        epochs = list(range(1, len(train_acc_history) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=epochs, y=train_acc_history, name="Train Accuracy",
            line={"color": "#4C9BE8"},
        ))
        fig.add_scatter(
            x=[epochs[-1]], y=[test_result.accuracy],
            mode="markers", name="Test Accuracy",
            marker={"color": "#E84C4C", "size": 10, "symbol": "star"},
        )
        fig.update_layout(
            title="Accuracy over Epochs", xaxis_title="Epoch",
            yaxis_title="Accuracy", height=300, margin={"t": 40},
        )
        st.plotly_chart(fig, use_container_width=True)

    if train_loss_history:
        epochs = list(range(1, len(train_loss_history) + 1))
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=epochs, y=train_loss_history, name="Train Loss",
            line={"color": "#F5A623"},
        ))
        fig2.update_layout(
            title="Loss over Epochs", xaxis_title="Epoch",
            yaxis_title="Loss", height=300, margin={"t": 40},
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.success(f"Model saved to: `{result['target_folder']}`")
