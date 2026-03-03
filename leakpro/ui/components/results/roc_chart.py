"""Stage 4 — ROC Analysis tab: interactive Plotly ROC curves for all attacks."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st


_PALETTE = [
    "#4C9BE8", "#E84C4C", "#27AE60", "#F5A623",
    "#9B59B6", "#1ABC9C", "#E67E22", "#3498DB",
]


def render_roc(results: list) -> None:
    """Render the interactive ROC curve tab."""
    roc_results = [r for r in results if r.roc_auc is not None and r.fpr is not None]

    if not roc_results:
        st.info("No ROC-capable results available (attacks with full score distribution required).")
        return

    st.caption(
        "Log-log scale — hover to inspect TPR/FPR values. "
        "The random-guess baseline (AUC = 0.5) is the dashed diagonal."
    )

    # Attack selector
    attack_labels = [
        f"{r.result_name} (AUC={r.roc_auc:.4f})" for r in roc_results
    ]
    selected = st.multiselect(
        "Highlight attacks", options=attack_labels, default=attack_labels,
        help="Deselect attacks to hide them from the chart."
    )
    selected_set = set(selected)

    fig = go.Figure()

    for i, r in enumerate(roc_results):
        label = f"{r.result_name} (AUC={r.roc_auc:.4f})"
        visible = label in selected_set
        colour = _PALETTE[i % len(_PALETTE)]

        fpr = np.asarray(r.fpr)
        tpr = np.asarray(r.tpr)

        # Shaded fill under curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            fill="tozeroy",
            fillcolor=f"rgba{tuple(int(colour.lstrip('#')[j:j+2], 16) for j in (0, 2, 4)) + (0.08,)}",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            visible=visible or "legendonly",
        ))

        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=label,
            line=dict(color=colour, width=2),
            hovertemplate=(
                f"<b>{r.result_name}</b><br>"
                "FPR: %{x:.5f}<br>TPR: %{y:.5f}<extra></extra>"
            ),
            visible=True if visible else "legendonly",
        ))

    # Random guess baseline
    baseline = np.linspace(1e-5, 1)
    fig.add_trace(go.Scatter(
        x=baseline, y=baseline,
        mode="lines",
        name="Random guess (AUC=0.5)",
        line=dict(dash="dash", color="#888888", width=1),
        hoverinfo="skip",
    ))

    fig.update_layout(
        xaxis=dict(type="log", title="False Positive Rate (FPR)", range=[-5, 0]),
        yaxis=dict(type="log", title="True Positive Rate (TPR)", range=[-5, 0]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
        height=520,
        margin=dict(t=20, b=120),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Metrics table
    st.markdown("##### TPR at fixed FPR thresholds")
    rows = []
    for r in roc_results:
        fpr_table = r.fixed_fpr_table or {}
        rows.append({
            "Attack": r.result_name,
            "AUC": f"{r.roc_auc:.4f}",
            "TPR@10%FPR": f"{fpr_table.get('TPR@10%FPR', 0):.4f}",
            "TPR@1%FPR":  f"{fpr_table.get('TPR@1%FPR', 0):.4f}",
            "TPR@0.1%FPR": f"{fpr_table.get('TPR@0.1%FPR', 0):.4f}",
            "TPR@0%FPR":  f"{fpr_table.get('TPR@0%FPR', 0):.4f}",
        })
    import pandas as pd  # noqa: PLC0415
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
