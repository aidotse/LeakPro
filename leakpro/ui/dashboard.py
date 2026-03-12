"""LeakPro Streamlit Auditor — main entry point.

Launch with:
    streamlit run leakpro/ui/dashboard.py

from the project root directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---- Page config --------------------------------------------------------
st.set_page_config(
    page_title="LeakPro — Privacy Auditor",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Session state defaults --------------------------------------------
_STAGE_DEFAULTS: dict = {
    "stage": 0,
    "train_config": None,
    "audit_config": None,
    "dpsgd_enabled": False,
    "dpsgd_params": None,
    "data_result": None,
    "train_result_dict": None,
    "audit_results": None,
}

for key, default in _STAGE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---- Sidebar pipeline tracker ------------------------------------------
_STAGES = [
    (0, "Overview",        "🏠"),
    (1, "Configure",       "⚙️"),
    (2, "Train Model",     "🏋️"),
    (3, "Run Attacks",     "🔍"),
    (4, "Explore Results", "📊"),
]

_STAGE_DONE_FLAGS = {
    0: lambda: st.session_state.stage > 0,
    1: lambda: st.session_state.train_config is not None and st.session_state.stage > 1,
    2: lambda: st.session_state.train_result_dict is not None,
    3: lambda: st.session_state.audit_results is not None,
    4: lambda: False,  # results stage is never "done"
}


def _stage_icon(stage_idx: int) -> str:
    current = st.session_state.stage
    if _STAGE_DONE_FLAGS[stage_idx]():
        return "✅"
    if stage_idx == current:
        return "▶️"
    return "○"


with st.sidebar:
    st.markdown("## 🔒 LeakPro")
    st.markdown("**Privacy Audit Pipeline**")
    st.markdown("---")

    for idx, name, emoji in _STAGES:
        icon = _stage_icon(idx)
        label = f"{icon}  {emoji} {name}"

        # Allow clicking completed stages to jump back
        if idx < st.session_state.stage:
            if st.button(label, key=f"nav_{idx}", use_container_width=True):
                st.session_state.stage = idx
                st.rerun()
        else:
            colour = "#4C9BE8" if idx == st.session_state.stage else "#666"
            st.markdown(
                f'<p style="color:{colour}; margin:4px 0; padding:6px 8px;">{label}</p>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # DP-SGD indicator in sidebar
    if st.session_state.dpsgd_enabled:
        dp = st.session_state.dpsgd_params or {}
        st.markdown(
            f"""
            <div style="padding:8px; border:1px solid #27AE60; border-radius:6px;
                        background:rgba(39,174,96,0.1); font-size:13px;">
                🛡️ <strong>DP-SGD ON</strong><br/>
                ε = {dp.get('target_epsilon', '?')}<br/>
                δ = {f"{dp['target_delta']:.2e}" if isinstance(dp.get('target_delta'), float) else '?'}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="padding:8px; border:1px solid #666; border-radius:6px;
                        font-size:13px; color:#888;">
                🔓 No DP-SGD
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---- Helper rendered inside Stage 4 ------------------------------------

def _render_attack_details(results: list) -> None:
    """Render per-attack expandable detail sections."""
    for r in results:
        auc_str = f"{r.roc_auc:.4f}" if r.roc_auc is not None else "N/A"
        with st.expander(f"{r.result_name}  (AUC = {auc_str})"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Configuration**")
                if r.metadata:
                    import pandas as pd  # noqa: PLC0415
                    meta = r.metadata if isinstance(r.metadata, dict) else r.metadata.model_dump()
                    st.dataframe(
                        pd.DataFrame(
                            [{"Parameter": k, "Value": v} for k, v in meta.items()]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption("No configuration metadata available.")

            with col2:
                st.markdown("**Metrics**")
                rows = [
                    {"Metric": "AUC", "Value": f"{r.roc_auc:.5f}" if r.roc_auc is not None else "N/A"},
                ]
                if r.fixed_fpr_table:
                    for k, v in r.fixed_fpr_table.items():
                        rows.append({"Metric": k, "Value": f"{v:.5f}"})
                import pandas as pd  # noqa: PLC0415
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---- Route to current stage --------------------------------------------
current_stage = st.session_state.stage

if current_stage == 0:
    from leakpro.ui.components.overview import render_overview  # noqa: PLC0415
    render_overview()

elif current_stage == 1:
    from leakpro.ui.components.configure import render_configure  # noqa: PLC0415
    render_configure()

elif current_stage == 2:
    from leakpro.ui.components.train import render_train  # noqa: PLC0415
    render_train()

elif current_stage == 3:
    from leakpro.ui.components.audit import render_audit  # noqa: PLC0415
    render_audit()

elif current_stage == 4:
    # ---- Results stage with sub-tabs -----------------------------------
    st.title("📊  Stage 4 — Explore Results")

    results = st.session_state.get("audit_results", [])

    if not results:
        st.warning("No audit results found. Run the audit first (Stage 3).")
        if st.button("← Go to Stage 3"):
            st.session_state.stage = 3
            st.rerun()
    else:
        tab_summary, tab_roc, tab_hist, tab_records, tab_detail = st.tabs([
            "Summary", "ROC Analysis", "Signal Histograms", "Sensitive Records", "Attack Details"
        ])

        with tab_summary:
            from leakpro.ui.components.results.summary import render_summary  # noqa: PLC0415
            render_summary(results)

        with tab_roc:
            from leakpro.ui.components.results.roc_chart import render_roc  # noqa: PLC0415
            render_roc(results)

        with tab_hist:
            from leakpro.ui.components.results.histograms import render_histograms  # noqa: PLC0415
            render_histograms(results)

        with tab_records:
            from leakpro.ui.components.results.sensitive_records import render_sensitive_records  # noqa: PLC0415
            render_sensitive_records(results)

        with tab_detail:
            _render_attack_details(results)

        # Back button
        st.markdown("---")
        if st.button("← Run Another Audit"):
            st.session_state.stage = 1
            for key in ["data_result", "train_result_dict", "audit_results"]:
                st.session_state.pop(key, None)
            st.rerun()
