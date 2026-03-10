"""Stage 0 — Overview / landing page."""

from pathlib import Path

import streamlit as st

_CIFAR_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "examples" / "mia" / "cifar"


def render_overview() -> None:
    """Render the landing page with intro text, pipeline diagram, and entry buttons."""

    st.title("LeakPro — Privacy Auditor")
    st.markdown(
        """
        **What is this tool?**

        LeakPro helps you measure how much private information your machine learning model leaks.
        Even after training is finished, an attacker may be able to determine which data points
        were used to train your model — a threat known as a **Membership Inference Attack (MIA)**.

        This tool guides you through the full privacy audit journey:
        """
    )

    # Pipeline diagram using columns
    col1, col2, col3, col4, col5 = st.columns(5)
    steps = [
        ("⚙️", "Configure", "Set training parameters and select attacks"),
        ("🏋️", "Train", "Train target model (standard or DP-SGD)"),
        ("🔍", "Audit", "Run membership inference attacks"),
        ("📊", "Explore", "Visualise leakage, sensitive records, and risk scores"),
    ]
    for col, (icon, title, desc) in zip([col2, col3, col4, col5], steps):
        with col:
            st.markdown(
                f"""
                <div style="text-align:center; padding:12px; border:1px solid #444;
                            border-radius:8px; min-height:100px;">
                    <div style="font-size:28px">{icon}</div>
                    <strong>{title}</strong><br/>
                    <small style="color:#aaa">{desc}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Check for existing results to offer resume.
    # Derive output_dir from session audit config or the default audit.yaml.
    audit_cfg = st.session_state.get("audit_config")
    if audit_cfg is None:
        try:
            from leakpro.ui.runner import LeakProRunner  # noqa: PLC0415
            audit_cfg = LeakProRunner.default_audit_config()
        except Exception:
            audit_cfg = {}
    _output_dir_rel = audit_cfg.get("audit", {}).get("output_dir", "./leakpro_output")
    resume_dir = (_CIFAR_OUTPUT_DIR / _output_dir_rel).resolve()
    has_results = (resume_dir / "data_objects").exists() and any(
        (resume_dir / "data_objects").glob("*.json")
    )

    col_start, col_resume = st.columns([1, 1])

    with col_start:
        st.markdown("### Start a new audit")
        st.caption("Configure, train a model from scratch, and run attacks.")
        if st.button("Start New Audit", type="primary", use_container_width=True):
            # Reset any previous run state
            for key in ["data_result", "train_result_dict", "audit_results",
                        "dpsgd_enabled", "dpsgd_params", "train_config", "audit_config"]:
                st.session_state.pop(key, None)
            st.session_state.stage = 1
            st.rerun()

    with col_resume:
        st.markdown("### Resume previous run")
        if has_results:
            disp_path = resume_dir.relative_to(Path.cwd()) if resume_dir.is_relative_to(Path.cwd()) else resume_dir
            st.caption(f"Previous results found in `{disp_path}`.")
            if st.button("Load Existing Results", use_container_width=True):
                from leakpro.ui.runner import LeakProRunner  # noqa: PLC0415
                results = LeakProRunner.load_audit_results_from_disk(str(resume_dir))
                if results:
                    st.session_state.audit_results = results
                    st.session_state.stage = 4
                    st.rerun()
                else:
                    st.error("Could not load results from disk.")
        else:
            st.caption("No previous results detected.")
            st.button("Load Existing Results", disabled=True, use_container_width=True)

    st.markdown("---")
    st.markdown(
        """
        <small style="color:#888">
        LeakPro is a privacy auditing framework by <a href="https://github.com/aidotse/LeakPro" target="_blank">AI Sweden</a>.
        MVP targets CIFAR-10/100 with Membership Inference Attacks.
        </small>
        """,
        unsafe_allow_html=True,
    )
