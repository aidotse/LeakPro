"""Stage 2 — Train all target models and display per-model metrics."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st


def render_train() -> None:
    """Render the training stage."""
    from leakpro.ui.runner import LeakProRunner  # noqa: PLC0415

    st.title("🏋️  Stage 2 — Train Models")
    models: list[dict] = st.session_state.get("models", [])
    st.caption(f"{len(models)} model(s) defined — each will be trained to its own folder.")

    runner = LeakProRunner()

    # Use first model's train_config for data prep (dataset choice); falls back to default
    first_tc = models[0].get("train_config") if models else {}
    if not first_tc:
        first_tc = runner.default_train_config()

    st.subheader("Step 1 — Prepare Dataset")
    _render_data_prep(runner, first_tc)

    if st.session_state.get("data_result"):
        st.markdown("---")
        st.subheader("Step 2 — Train Models")
        _render_all_model_training(runner, models)

    _render_navigation(models)


def _render_data_prep(runner: object, train_config: dict) -> None:
    """Render dataset download / status section."""
    if st.session_state.get("data_result"):
        dr = st.session_state.data_result
        st.success(
            f"Dataset ready — {len(dr['train_indices'])} train / "
            f"{len(dr['test_indices'])} test samples ({dr['dataset_name']})."
        )
        if st.button("Re-prepare Data"):
            st.session_state.pop("data_result", None)
            for m in st.session_state.get("models", []):
                m["train_result_dict"] = None
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


def _render_all_model_training(runner: object, models: list[dict]) -> None:
    """Render the Train All Models button and per-model result cards."""
    all_trained = all(m.get("train_result_dict") for m in models)

    if not all_trained:
        pending = [m for m in models if not m.get("train_result_dict")]
        st.caption(f"{len(pending)} model(s) still need training.")
        if st.button("Train All Models", type="primary"):
            for model in models:
                if model.get("train_result_dict"):
                    continue
                with st.spinner(f"Training **{model['name']}**…"):
                    log_ph = st.empty()
                    model_tc = model.get("train_config") or {}
                    try:
                        if model["dpsgd_enabled"]:
                            result = runner.train_dpsgd(  # type: ignore[attr-defined]
                                model_tc,
                                model["dpsgd_params"],
                                st.session_state.data_result,
                                target_folder=model["target_folder"],
                                log_container=log_ph,
                            )
                        else:
                            result = runner.train_standard(  # type: ignore[attr-defined]
                                model_tc,
                                st.session_state.data_result,
                                target_folder=model["target_folder"],
                                log_container=log_ph,
                            )
                        model["train_result_dict"] = result
                        st.success(f"✅ {model['name']} trained.")
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Training failed for **{model['name']}**: {e}")
            st.session_state.models = models
            st.rerun()

    # Show results for already-trained models
    trained = [m for m in models if m.get("train_result_dict")]
    if trained:
        st.markdown("##### Trained Models")
        _render_training_summary_table(trained)
        for model in trained:
            with st.expander(f"📈 {model['name']} — training curves", expanded=False):
                _show_training_metrics(model["train_result_dict"], model["dpsgd_enabled"])


def _render_training_summary_table(models: list[dict]) -> None:
    """Render a summary table of all trained models."""
    import pandas as pd  # noqa: PLC0415

    rows = []
    for m in models:
        tr = m["train_result_dict"]
        train_acc = tr["train_result"].metrics.accuracy
        test_acc = tr["test_result"].accuracy
        dp_str = (
            f"ε={m['dpsgd_params']['target_epsilon']}" if m["dpsgd_enabled"] else "No"
        )
        rows.append({
            "Model": m["name"],
            "Train Acc": f"{train_acc:.3f}",
            "Test Acc": f"{test_acc:.3f}",
            "DP-SGD": dp_str,
            "Folder": tr["target_folder"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_navigation(models: list[dict]) -> None:
    """Render back / forward navigation buttons."""
    all_trained = bool(models) and all(m.get("train_result_dict") for m in models)
    if all_trained:
        st.markdown("---")
        col_back, _, col_fwd = st.columns([1, 3, 1])
        with col_back:
            if st.button("← Reconfigure", use_container_width=True):
                for m in models:
                    m["train_result_dict"] = None
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
