"""Stage 4 — Signal Histograms tab: member vs non-member score distributions."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st


def render_histograms(models: list) -> None:
    """Render signal histogram plots with a model and attack selector."""
    audited = [m for m in models if m.get("audit_results")]
    if not audited:
        st.info("No signal-value data available for histogram visualisation.")
        return

    if len(audited) > 1:
        model_name = st.selectbox(
            "Select model", [m["name"] for m in audited], key="hist_model_select"
        )
        model = next(m for m in audited if m["name"] == model_name)
    else:
        model = audited[0]

    hist_results = [r for r in (model.get("audit_results") or []) if r.signal_values is not None]
    if not hist_results:
        st.info(f"No signal-value data available for **{model['name']}**.")
        return

    st.caption(
        f"Model: **{model['name']}** — "
        "a larger separation between the **member** (blue) and **non-member** (red) "
        "distributions indicates stronger privacy leakage."
    )

    attack_names = [r.result_name for r in hist_results]
    selected_name = st.selectbox("Select attack", attack_names, key="hist_attack_select")
    result = next(r for r in hist_results if r.result_name == selected_name)

    signal = np.asarray(result.signal_values).ravel()
    labels = np.asarray(result.true).ravel()
    members = signal[labels == 1]
    non_members = signal[labels == 0]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=members,
        name="Members (training data)",
        nbinsx=100,
        opacity=0.6,
        marker_color="#4C9BE8",
        hovertemplate="Score: %{x:.4f}<br>Count: %{y}<extra>Members</extra>",
    ))
    fig.add_trace(go.Histogram(
        x=non_members,
        name="Non-members",
        nbinsx=100,
        opacity=0.6,
        marker_color="#E84C4C",
        hovertemplate="Score: %{x:.4f}<br>Count: %{y}<extra>Non-members</extra>",
    ))
    fig.update_layout(
        barmode="overlay",
        xaxis_title="Attack signal score",
        yaxis_title="Count",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.3, "xanchor": "center", "x": 0.5},
        height=400,
        margin={"t": 20},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Decision threshold explorer")
    st.caption("Move the threshold to see how TP/FP/TN/FN change.")

    score_min = float(signal.min())
    score_max = float(signal.max())
    default_threshold = float(np.median(signal))

    threshold = st.slider(
        "Decision threshold", min_value=score_min, max_value=score_max,
        value=default_threshold, step=(score_max - score_min) / 200,
        key="hist_threshold_slider",
    )

    preds = (signal >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    total_pos = tp + fn
    total_neg = tn + fp

    tpr = tp / total_pos if total_pos else 0
    fpr = fp / total_neg if total_neg else 0
    acc = (tp + tn) / len(labels) if len(labels) else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("TPR", f"{tpr:.4f}", help="True positive rate at this threshold.")
    c2.metric("FPR", f"{fpr:.4f}", help="False positive rate at this threshold.")
    c3.metric("Accuracy", f"{acc:.4f}")
    c4.metric("TP / FN", f"{tp} / {fn}")
    c5.metric("FP / TN", f"{fp} / {tn}")
