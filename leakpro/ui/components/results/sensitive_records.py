"""Stage 4 — Sensitive Records tab: top-N riskiest training samples + CSV export."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_CIFAR_DIR = Path(__file__).parent.parent.parent.parent.parent / "examples" / "mia" / "cifar"


def render_sensitive_records(results: list) -> None:
    """Render the sensitive records explorer."""
    # Pick best result by AUC
    valid = [r for r in results if r.signal_values is not None and r.roc_auc is not None]
    if not valid:
        # Fall back to any result with signal values
        valid = [r for r in results if r.signal_values is not None]
    if not valid:
        st.info("No signal-value data available for sensitive records analysis.")
        return

    # Attack selector
    attack_names = [r.result_name for r in valid]
    selected_name = st.selectbox("Select attack", attack_names, key="rec_attack_select")
    result = next(r for r in valid if r.result_name == selected_name)

    signal = np.asarray(result.signal_values).ravel()
    labels = np.asarray(result.true).ravel()

    # Filter to members only (training set samples)
    member_mask = labels == 1
    member_signals = signal[member_mask]
    member_original_indices = np.where(member_mask)[0]

    # Sort members by descending signal score
    sorted_order = np.argsort(-member_signals)
    sorted_member_signals = member_signals[sorted_order]
    sorted_member_audit_indices = member_original_indices[sorted_order]  # index into audit set

    top_n = st.slider(
        "Top-N riskiest samples", min_value=5, max_value=min(200, len(sorted_member_signals)),
        value=min(20, len(sorted_member_signals)), key="rec_topn"
    )

    top_signals = sorted_member_signals[:top_n]
    top_audit_indices = sorted_member_audit_indices[:top_n]

    # ---- Load training images if available ------------------------------
    data_result = st.session_state.get("data_result")
    train_indices = None
    population_data = None

    if data_result is not None:
        train_indices = np.asarray(data_result["train_indices"])
        population_data = data_result["data"]  # tensor (N, C, H, W) in [0,1]

    # ---- Image grid (CIFAR) ---------------------------------------------
    if population_data is not None and train_indices is not None:
        st.markdown("##### Top-N riskiest training images")
        st.caption(
            "These training images were most confidently identified as training members "
            "by the attack — they are the most 'exposed' samples in your dataset."
        )

        # Map audit-set member index → train_indices position
        # The i-th member in the audit set corresponds to train_indices[i]
        # (approximate — valid when the audit set preserves training order)
        num_members = len(train_indices)
        valid_mask = top_audit_indices < num_members
        display_indices = top_audit_indices[valid_mask]
        display_signals = top_signals[valid_mask]

        pop_indices = train_indices[display_indices]
        images = population_data[pop_indices]  # (M, C, H, W) tensor in [0,1]

        cols_per_row = 5
        rows = [
            list(range(i, min(i + cols_per_row, len(images))))
            for i in range(0, len(images), cols_per_row)
        ]

        for row_indices in rows:
            cols = st.columns(cols_per_row)
            for col, idx in zip(cols, row_indices):
                img = images[idx].numpy().transpose(1, 2, 0)  # (H, W, C)
                img = (img * 255).clip(0, 255).astype("uint8")
                score = display_signals[idx]
                col.image(img, caption=f"score: {score:.4f}", use_container_width=True)
    else:
        st.info(
            "Training images not available in this session. "
            "Run a full audit (starting from Stage 2) to enable image display."
        )

    # ---- CSV export for all data types ----------------------------------
    st.markdown("---")
    st.markdown("##### Export riskiest records as CSV")

    csv_rows = []
    for rank, (audit_idx, score) in enumerate(
        zip(top_audit_indices[:top_n], top_signals[:top_n]), start=1
    ):
        row: dict = {
            "rank": rank,
            "audit_set_index": int(audit_idx),
            "risk_score": float(score),
            "is_member": int(labels[audit_idx]) if audit_idx < len(labels) else "unknown",
        }
        if train_indices is not None and audit_idx < len(train_indices):
            row["population_index"] = int(train_indices[audit_idx])
        csv_rows.append(row)

    df_export = pd.DataFrame(csv_rows)
    csv_bytes = df_export.to_csv(index=False).encode()

    st.download_button(
        label=f"Download top-{top_n} records as CSV",
        data=csv_bytes,
        file_name=f"leakpro_sensitive_records_{selected_name}_top{top_n}.csv",
        mime="text/csv",
        type="primary",
    )

    # Preview table
    with st.expander("Preview CSV content"):
        st.dataframe(df_export, use_container_width=True, hide_index=True)
