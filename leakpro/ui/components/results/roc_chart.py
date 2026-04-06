"""Stage 4 — ROC Analysis tab: interactive Plotly ROC curves for all models and attacks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# One base colour per model; attacks within a model share the same hue family
_MODEL_COLOURS = [
    "#4C9BE8",  # blue
    "#E84C4C",  # red
    "#27AE60",  # green
    "#F5A623",  # orange
    "#9B59B6",  # purple
]


def render_roc(models: list) -> None:
    """Render the interactive ROC curve tab with per-model colour coding."""
    roc_entries = [
        (mi, m, r)
        for mi, m in enumerate(models)
        for r in (m.get("audit_results") or [])
        if r.roc_auc is not None and r.fpr is not None
    ]

    if not roc_entries:
        st.info("No ROC-capable results available (attacks with full score distribution required).")
        return

    st.caption(
        "Log-log scale — hover to inspect TPR/FPR values. "
        "Curves are colour-coded by model. "
        "The random-guess baseline (AUC = 0.5) is the dashed diagonal."
    )

    model_names = list({m["name"] for _, m, _ in roc_entries})
    if len(model_names) > 1:
        selected_model_names = set(st.multiselect(
            "Show models", options=model_names, default=model_names,
            key="roc_model_select",
        ))
    else:
        selected_model_names = set(model_names)

    fig = go.Figure()

    for mi, m, r in roc_entries:
        if m["name"] not in selected_model_names:
            continue

        colour = _MODEL_COLOURS[mi % len(_MODEL_COLOURS)]
        label = f"{m['name']} / {r.result_name} (AUC={r.roc_auc:.4f})"

        fpr_arr = np.asarray(r.fpr)
        tpr_arr = np.asarray(r.tpr)

        fig.add_trace(go.Scatter(
            x=fpr_arr, y=tpr_arr,
            fill="tozeroy",
            fillcolor=f"rgba{tuple(int(colour.lstrip('#')[j:j+2], 16) for j in (0, 2, 4)) + (0.07,)}",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter(
            x=fpr_arr, y=tpr_arr,
            mode="lines",
            name=label,
            line={"color": colour, "width": 2},
            hovertemplate=(
                f"<b>{label}</b><br>"
                "FPR: %{x:.5f}<br>TPR: %{y:.5f}<extra></extra>"
            ),
        ))

    baseline = np.linspace(1e-5, 1)
    fig.add_trace(go.Scatter(
        x=baseline, y=baseline,
        mode="lines",
        name="Random guess (AUC=0.5)",
        line={"dash": "dash", "color": "#888888", "width": 1},
        hoverinfo="skip",
    ))

    fig.update_layout(
        xaxis={"type": "log", "title": "False Positive Rate (FPR)", "range": [-5, 0]},
        yaxis={"type": "log", "title": "True Positive Rate (TPR)", "range": [-5, 0]},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.45, "xanchor": "center", "x": 0.5},
        height=540,
        margin={"t": 20, "b": 140},
        hovermode="closest",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### TPR at fixed FPR thresholds")
    rows = []
    for _, m, r in roc_entries:
        if m["name"] not in selected_model_names:
            continue
        fpr_table = r.fixed_fpr_table or {}
        rows.append({
            "Model": m["name"],
            "Attack": r.result_name,
            "AUC": f"{r.roc_auc:.4f}",
            "TPR@10%FPR": f"{fpr_table.get('TPR@10%FPR', 0):.4f}",
            "TPR@1%FPR":  f"{fpr_table.get('TPR@1%FPR', 0):.4f}",
            "TPR@0.1%FPR": f"{fpr_table.get('TPR@0.1%FPR', 0):.4f}",
            "TPR@0%FPR":  f"{fpr_table.get('TPR@0%FPR', 0):.4f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
