"""Stage 3 — Run MIA attacks with live log streaming."""

from __future__ import annotations

import streamlit as st


def render_audit() -> None:
    """Render the attack execution stage."""
    from leakpro.ui.runner import LeakProRunner  # noqa: PLC0415

    st.title("🔍  Stage 3 — Run Attacks")
    dpsgd = st.session_state.get("dpsgd_enabled", False)

    ac = st.session_state.get("audit_config", {})
    attack_list = ac.get("audit", {}).get("attack_list", [])
    attack_names = [a["attack"] for a in attack_list]

    st.caption(
        f"{'DP-SGD model' if dpsgd else 'Standard model'} — "
        f"attacks: {', '.join(attack_names) if attack_names else 'none selected'}"
    )

    runner = LeakProRunner()

    if "audit_results" not in st.session_state:
        st.markdown(
            """
            Click **Run Attacks** to execute the selected membership inference attacks
            against the trained target model. Log output will stream below.
            """
        )
        if st.button("Run Attacks", type="primary"):
            log_placeholder = st.empty()
            with st.spinner("Running attacks… this may take a while."):
                try:
                    results = runner.run_audit(dpsgd=dpsgd, log_container=log_placeholder)
                    st.session_state.audit_results = results
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Audit failed: {e}")
    else:
        results = st.session_state.audit_results
        st.success(f"Audit complete — {len(results)} attack result(s) available.")

        # Quick summary table
        rows = []
        for r in results:
            auc = f"{r.roc_auc:.4f}" if r.roc_auc is not None else "N/A"
            tpr_01 = "N/A"
            if r.fixed_fpr_table:
                tpr_01 = f"{r.fixed_fpr_table.get('TPR@0.1%FPR', 0):.4f}"
            rows.append({"Attack": r.result_name, "AUC": auc, "TPR@0.1%FPR": tpr_01})

        import pandas as pd  # noqa: PLC0415
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------ #
    # Navigation
    # ------------------------------------------------------------------ #
    st.markdown("---")
    col_back, _, col_fwd = st.columns([1, 3, 1])
    with col_back:
        if st.button("← Re-train", use_container_width=True):
            st.session_state.pop("audit_results", None)
            st.session_state.stage = 2
            st.rerun()
    with col_fwd:
        if "audit_results" in st.session_state:
            if st.button("Explore Results →", type="primary", use_container_width=True):
                st.session_state.stage = 4
                st.rerun()
