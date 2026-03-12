"""Stage 4 — Signal Histograms tab: member vs non-member score distributions."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st


def render_histograms(results: list) -> None:
    """Render signal histogram plots for each attack result."""
    hist_results = [r for r in results if r.signal_values is not None]

    if not hist_results:
        st.info("No signal-value data available for histogram visualisation.")
        return

    st.caption(
        "A larger separation between the **member** (blue) and **non-member** (red) "
        "distributions indicates stronger privacy leakage."
    )

    # Attack picker
    attack_names = [r.result_name for r in hist_results]
    selected_name = st.selectbox("Select attack", attack_names)
    result = next(r for r in hist_results if r.result_name == selected_name)

    signal = np.asarray(result.signal_values).ravel()
    labels = np.asarray(result.true).ravel()
    members = signal[labels == 1]
    non_members = signal[labels == 0]

    # ---- Histogram ------------------------------------------------------
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

    # ---- Interactive threshold slider -----------------------------------
    st.markdown("##### Decision threshold explorer")
    st.caption("Move the threshold to see how TP/FP/TN/FN change.")

    score_min = float(signal.min())
    score_max = float(signal.max())
    default_threshold = float(np.median(signal))

    threshold = st.slider(
        "Decision threshold", min_value=score_min, max_value=score_max,
        value=default_threshold, step=(score_max - score_min) / 200,
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
    with c4:
        st.markdown("**TP / FN**")
        st.markdown(f"{tp} / {fn}")
    with c5:
        st.markdown("**FP / TN**")
        st.markdown(f"{fp} / {tn}")
