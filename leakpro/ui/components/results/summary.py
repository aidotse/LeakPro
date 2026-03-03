"""Stage 4 — Summary tab: risk cards, traffic-light indicator, and plain-English verdict."""

from __future__ import annotations

import streamlit as st


def render_summary(results: list) -> None:
    """Render the summary risk dashboard."""
    if not results:
        st.warning("No results to display.")
        return

    # Gather metrics
    valid = [r for r in results if r.roc_auc is not None]
    best_auc = max((r.roc_auc for r in valid), default=None)
    worst_tpr = max(
        (r.fixed_fpr_table.get("TPR@0.1%FPR", 0) for r in valid if r.fixed_fpr_table),
        default=None,
    )
    num_attacks = len(results)

    # DP-SGD context (if available)
    dpsgd_enabled = st.session_state.get("dpsgd_enabled", False)
    dpsgd_params = st.session_state.get("dpsgd_params", {})
    train_result_dict = st.session_state.get("train_result_dict", {})

    # ---- Risk level calculation ----------------------------------------
    def _risk_level(auc: float | None) -> tuple[str, str, str]:
        if auc is None:
            return "Unknown", "❓", "#888888"
        if auc >= 0.75:
            return "HIGH", "🔴", "#E84C4C"
        if auc >= 0.60:
            return "MEDIUM", "🟡", "#F5A623"
        return "LOW", "🟢", "#27AE60"

    risk_label, risk_icon, risk_colour = _risk_level(best_auc)

    # ---- Top metric cards -----------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Attacks run", num_attacks)
    with col2:
        st.metric(
            "Best attack AUC",
            f"{best_auc:.4f}" if best_auc is not None else "N/A",
            help="Area under the ROC curve — 0.5 = random, 1.0 = perfect attack.",
        )
    with col3:
        st.metric(
            "Worst TPR @ 0.1% FPR",
            f"{worst_tpr:.4f}" if worst_tpr is not None else "N/A",
            help="True positive rate when accepting only 0.1% false positives — "
                 "a strict real-world scenario.",
        )
    with col4:
        st.markdown(
            f"""
            <div style="text-align:center; padding:16px; border:2px solid {risk_colour};
                        border-radius:10px; background:rgba(0,0,0,0.05);">
                <div style="font-size:32px">{risk_icon}</div>
                <div style="font-size:20px; font-weight:bold; color:{risk_colour}">
                    {risk_label} RISK
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Plain-English verdict ------------------------------------------
    st.markdown("---")
    st.subheader("Verdict")

    if best_auc is not None:
        if best_auc >= 0.75:
            verdict = (
                f"Your model is at **HIGH risk** of membership inference (best AUC = {best_auc:.3f}). "
                "An attacker can reliably distinguish training samples from non-members. "
                "Consider retraining with differential privacy (DP-SGD)."
            )
        elif best_auc >= 0.60:
            verdict = (
                f"Your model shows **moderate leakage** (best AUC = {best_auc:.3f}). "
                "Attacks are partially effective. DP-SGD training may reduce this risk."
            )
        else:
            verdict = (
                f"Your model appears **well-protected** (best AUC = {best_auc:.3f}). "
                "Membership inference attacks perform close to random guessing."
            )
        st.info(verdict)
    else:
        st.info("No AUC-capable results available for verdict.")

    # ---- DP-SGD context panel ------------------------------------------
    if dpsgd_enabled and dpsgd_params:
        st.markdown("---")
        st.subheader("Differential Privacy Impact")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Target ε", dpsgd_params.get("target_epsilon", "—"))
        c2.metric("Target δ", f"{dpsgd_params.get('target_delta', 0):.2e}")
        c3.metric("Max grad norm", dpsgd_params.get("max_grad_norm", "—"))
        if train_result_dict:
            test_acc = train_result_dict.get("test_result")
            if test_acc:
                c4.metric("Model test accuracy", f"{test_acc.accuracy:.3f}")

        if best_auc is not None:
            st.markdown(
                f"> DP-SGD (ε={dpsgd_params.get('target_epsilon')}) "
                f"→ Best attack AUC = **{best_auc:.3f}** "
                f"{'(significantly reduced — DP is working!)' if best_auc < 0.60 else '(DP may need stronger ε)'}"
            )
