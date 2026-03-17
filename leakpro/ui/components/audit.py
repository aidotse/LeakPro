"""Stage 3 — Run MIA attacks on all trained models with live log streaming."""

from __future__ import annotations

import streamlit as st


def render_audit() -> None:
    """Render the attack execution stage."""
    from leakpro.ui.runner import LeakProRunner  # noqa: PLC0415

    st.title("🔍  Stage 3 — Run Attacks")

    models: list[dict] = st.session_state.get("models", [])
    trained_models = [m for m in models if m.get("train_result_dict")]

    # Collect all unique attack names across all models
    all_attacks: set[str] = set()
    for m in models:
        for a in (m.get("attack_list") or []):
            all_attacks.add(a["attack"])
    st.caption(f"Attacks (across all models): {', '.join(sorted(all_attacks)) if all_attacks else 'none selected'}")

    if not trained_models:
        st.warning("No trained models found. Go back to Stage 2 and train your models first.")
        if st.button("← Go to Stage 2"):
            st.session_state.stage = 2
            st.rerun()
        return

    runner = LeakProRunner()
    all_audited = all(m.get("audit_results") for m in trained_models)

    if not all_audited:
        pending = [m for m in trained_models if not m.get("audit_results")]
        st.caption(f"{len(pending)} model(s) still need auditing.")
        st.markdown(
            "Click **Run Attacks** to execute the selected membership inference attacks "
            "against each trained model. Log output will stream below."
        )
        if st.button("Run Attacks", type="primary"):
            _run_audit_loop(runner, trained_models, models)
    else:
        st.success(f"All {len(trained_models)} model(s) audited.")
        _render_audit_summary(trained_models)

    st.markdown("---")
    col_back, _, col_fwd = st.columns([1, 3, 1])
    with col_back:
        if st.button("← Re-train", use_container_width=True):
            for m in models:
                m["audit_results"] = None
            st.session_state.stage = 2
            st.rerun()
    with col_fwd:
        if all_audited and st.button("Explore Results →", type="primary", use_container_width=True):
            st.session_state.stage = 4
            st.rerun()


def _run_audit_loop(runner: object, trained_models: list[dict], all_models: list[dict]) -> None:
    """Run audits sequentially for all untrained models and trigger a rerun."""
    for model in trained_models:
        if model.get("audit_results"):
            continue
        with st.spinner(f"Auditing **{model['name']}**…"):
            log_ph = st.empty()
            try:
                results = runner.run_audit(  # type: ignore[attr-defined]
                    target_folder=model["target_folder"],
                    attack_list=model.get("attack_list"),
                    dpsgd=model["dpsgd_enabled"],
                    log_container=log_ph,
                )
                model["audit_results"] = results
                st.success(f"✅ {model['name']} — {len(results)} attack(s) done.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Audit failed for **{model['name']}**: {e}")
    st.session_state.models = all_models
    st.rerun()


def _render_audit_summary(models: list[dict]) -> None:
    """Render a quick per-model AUC summary table after auditing."""
    import pandas as pd  # noqa: PLC0415

    rows = []
    for m in models:
        for r in (m.get("audit_results") or []):
            auc = f"{r.roc_auc:.4f}" if r.roc_auc is not None else "N/A"
            tpr_01 = "N/A"
            if r.fixed_fpr_table:
                tpr_01 = f"{r.fixed_fpr_table.get('TPR@0.1%FPR', 0):.4f}"
            rows.append({
                "Model": m["name"],
                "Attack": r.result_name,
                "AUC": auc,
                "TPR@0.1%FPR": tpr_01,
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
