"""Stage 4 — Summary tab: multi-model comparison table and risk verdict."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def _risk_level(auc: float | None) -> tuple[str, str, str]:
    """Return (label, icon, colour) for a given AUC."""
    if auc is None:
        return "Unknown", "❓", "#888888"
    if auc >= 0.75:
        return "HIGH", "🔴", "#E84C4C"
    if auc >= 0.60:
        return "MEDIUM", "🟡", "#F5A623"
    return "LOW", "🟢", "#27AE60"


def render_summary(models: list) -> None:
    """Render the multi-model summary comparison dashboard."""
    if not models:
        st.warning("No results to display.")
        return

    st.subheader("Model Comparison")
    _render_comparison_table(models)
    st.markdown("---")
    st.subheader("Verdict")
    _render_verdict(models)


def _model_best_auc(model: dict) -> float | None:
    """Return the best (highest) AUC across all attacks for a model."""
    results = model.get("audit_results") or []
    valid = [r.roc_auc for r in results if r.roc_auc is not None]
    return max(valid) if valid else None


def _render_comparison_table(models: list) -> None:
    """Render one row per model with key metrics."""
    rows = []
    for m in models:
        results = m.get("audit_results") or []
        valid = [r for r in results if r.roc_auc is not None]
        best_auc = max((r.roc_auc for r in valid), default=None)
        worst_tpr = max(
            (r.fixed_fpr_table.get("TPR@0.1%FPR", 0) for r in valid if r.fixed_fpr_table),
            default=None,
        )
        tr = m.get("train_result_dict") or {}
        test_acc = tr.get("test_result")
        risk_label, risk_icon, _ = _risk_level(best_auc)
        dp_str = (
            f"ε={m['dpsgd_params']['target_epsilon']}"
            if m.get("dpsgd_enabled") and m.get("dpsgd_params")
            else "No"
        )
        rows.append({
            "Model": m["name"],
            "Best AUC": f"{best_auc:.4f}" if best_auc is not None else "N/A",
            "TPR@0.1%FPR": f"{worst_tpr:.4f}" if worst_tpr is not None else "N/A",
            "Test Accuracy": f"{test_acc.accuracy:.3f}" if test_acc else "N/A",
            "DP-SGD": dp_str,
            "Risk": f"{risk_icon} {risk_label}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Traffic-light cards for each model
    if len(models) > 1:
        cols = st.columns(len(models))
        for col, m in zip(cols, models):
            best_auc = _model_best_auc(m)
            _, risk_icon, risk_colour = _risk_level(best_auc)
            auc_str = f"{best_auc:.3f}" if best_auc is not None else "N/A"
            col.markdown(
                f"""
                <div style="text-align:center; padding:12px; border:2px solid {risk_colour};
                            border-radius:8px; background:rgba(0,0,0,0.04);">
                    <div style="font-size:24px">{risk_icon}</div>
                    <strong>{m['name']}</strong><br/>
                    <small>AUC = {auc_str}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_verdict(models: list) -> None:
    """Render plain-English risk verdict for each model."""
    for m in models:
        best_auc = _model_best_auc(m)
        _, _, colour = _risk_level(best_auc)
        dp_note = ""
        if m.get("dpsgd_enabled") and m.get("dpsgd_params"):
            eps = m["dpsgd_params"].get("target_epsilon", "?")
            dp_note = f" (DP-SGD ε={eps})"

        if best_auc is None:
            msg = f"**{m['name']}{dp_note}**: No AUC data available."
        elif best_auc >= 0.75:
            msg = (
                f"**{m['name']}{dp_note}**: HIGH risk — AUC = {best_auc:.3f}. "
                "An attacker can reliably identify training members."
            )
        elif best_auc >= 0.60:
            msg = (
                f"**{m['name']}{dp_note}**: MEDIUM risk — AUC = {best_auc:.3f}. "
                "Attacks are partially effective."
            )
        else:
            msg = (
                f"**{m['name']}{dp_note}**: LOW risk — AUC = {best_auc:.3f}. "
                "Attacks perform close to random guessing."
            )
        st.markdown(
            f'<div style="padding:8px; border-left:4px solid {colour}; margin-bottom:8px;">'
            f"{msg}</div>",
            unsafe_allow_html=True,
        )

    # DP comparison narrative when multiple models present
    dp_models = [m for m in models if m.get("dpsgd_enabled")]
    std_models = [m for m in models if not m.get("dpsgd_enabled")]
    if dp_models and std_models:
        st.markdown("---")
        st.subheader("Privacy-Accuracy Trade-off")
        for std in std_models:
            std_auc = _model_best_auc(std)
            std_tr = (std.get("train_result_dict") or {}).get("test_result")
            for dp in dp_models:
                dp_auc = _model_best_auc(dp)
                dp_tr = (dp.get("train_result_dict") or {}).get("test_result")
                eps = (dp.get("dpsgd_params") or {}).get("target_epsilon", "?")
                auc_delta = (
                    f"AUC {std_auc:.3f} → {dp_auc:.3f}" if std_auc and dp_auc else "AUC N/A"
                )
                acc_delta = ""
                if std_tr and dp_tr:
                    diff = dp_tr.accuracy - std_tr.accuracy
                    sign = "+" if diff >= 0 else ""
                    acc_delta = f", accuracy {sign}{diff:.3f}"
                st.info(
                    f"**{std['name']}** vs **{dp['name']}** (ε={eps}): "
                    f"{auc_delta}{acc_delta}"
                )
