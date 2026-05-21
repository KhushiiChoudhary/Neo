from __future__ import annotations

import subprocess
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

PLOT_BROWN = "#92400e"

load_dotenv()

st.set_page_config(
    page_title="Neo: AutoML Agent",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── base ── */
    .stApp { background-color: #ffffff; }
    header[data-testid="stHeader"] { display: none; }
    .block-container { padding-top: 1.5rem; }

    /* ── header ── */
    .main-header {
        font-size: 1.6rem;
        font-weight: 700;
        line-height: 1.5;
        color: #92400e;
        margin-bottom: 0.1rem;
    }
    .sub-header { color: #78716c; font-size: 0.85rem; }

    /* ── progress stepper ── */
    .stepper { display: flex; align-items: flex-start; margin: 1.2rem 0 1.6rem 0; }
    .step { display: flex; flex-direction: column; align-items: center; flex: 1; }
    .step-circle {
        width: 30px; height: 30px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.8rem; font-weight: 700;
    }
    .step-circle.done   { background: #92400e; color: #fff; }
    .step-circle.active {
        background: linear-gradient(135deg, #92400e, #b45309);
        color: #fff;
        box-shadow: 0 0 14px #92400e55;
    }
    .step-circle.pending { background: #f5f0ed; color: #a8a29e; border: 1px solid #e7e0da; }
    .step-label { font-size: 0.68rem; margin-top: 5px; text-align: center; }
    .step-label.done    { color: #92400e; }
    .step-label.active  { color: #b45309; font-weight: 600; }
    .step-label.pending { color: #a8a29e; }
    .step-connector { flex: 1; height: 2px; margin: 14px 3px 0 3px; }
    .step-connector.done    { background: #92400e; }
    .step-connector.pending { background: #e7e0da; }

    /* ── stat cards ── */
    .stat-row { display: flex; gap: 12px; margin: 0.8rem 0; }
    .stat-card {
        flex: 1;
        background: #faf7f5;
        border: 1px solid #e7e0da;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        text-align: center;
    }
    .stat-value { font-size: 1.35rem; font-weight: 700; color: #1c1917; }
    .stat-label { font-size: 0.72rem; color: #78716c; margin-top: 3px; }

    /* ── confirm panel ── */
    .confirm-box {
        background: #faf7f5;
        border: 1px solid #d4a574;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0 1rem 0;
    }

    /* ── sidebar ── */
    section[data-testid="stSidebar"] { background: #faf7f5; border-right: 1px solid #e7e0da; }
    .sidebar-section { margin-bottom: 1.4rem; }
    .sidebar-title { font-size: 0.7rem; font-weight: 700; color: #a8a29e; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.5rem; }
    .tech-badge {
        display: inline-block;
        background: #ffffff;
        border: 1px solid #e7e0da;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.72rem;
        color: #78716c;
        margin: 2px;
    }

    /* ── hero row (upload screen) ── */
    .hero-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: #faf7f5;
        border: 1px solid #e7e0da;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
    }
    .hero-item { display: flex; align-items: center; gap: 0.5rem; flex: 1; }
    .hero-num {
        width: 26px; height: 26px; border-radius: 50%;
        background: #92400e; color: #fff;
        font-size: 0.75rem; font-weight: 700;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
    }
    .hero-label { font-size: 0.82rem; color: #1c1917; font-weight: 500; }
    .hero-sep { color: #d4a574; font-size: 1rem; font-weight: 300; }

    /* ── best model callout ── */
    .callout-box {
        background: linear-gradient(135deg, #fdf6ee, #faf7f5);
        border-left: 4px solid #92400e;
        border-radius: 0 10px 10px 0;
        padding: 1rem 1.4rem;
        margin: 1rem 0;
    }
    .callout-title { font-size: 1rem; font-weight: 700; color: #92400e; margin-bottom: 0.35rem; }
    .callout-box p { margin: 0.2rem 0; font-size: 0.88rem; color: #44403c; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── session state defaults ────────────────────────────────────────────────────
defaults: dict = {
    "stage": "upload",
    "messages": [],
    "df": None,
    "filename": None,
    "user_goal": None,
    "suggested_target": None,
    "suggested_problem_type": None,
    "suggested_reasoning": None,
    "confirmed_target": None,
    "confirmed_problem_type": None,
    "agent_state": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── helpers ───────────────────────────────────────────────────────────────────

def add_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


def render_messages() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def stat_cards(stats: list[tuple[str, str]]) -> None:
    """Render a row of custom stat cards. stats = [(label, value), ...]"""
    cards_html = "".join(
        f'<div class="stat-card"><div class="stat-value">{v}</div><div class="stat-label">{l}</div></div>'
        for l, v in stats
    )
    st.markdown(f'<div class="stat-row">{cards_html}</div>', unsafe_allow_html=True)


STAGE_ORDER = ["upload", "confirm", "running", "results"]
STAGE_LABELS = {"upload": "Upload", "confirm": "Confirm", "running": "Training", "results": "Results"}
STAGE_ICONS  = {"upload": "1", "confirm": "2", "running": "3", "results": "✓"}


def render_stepper(current: str) -> None:
    current_idx = STAGE_ORDER.index(current)
    parts = []
    for i, stage in enumerate(STAGE_ORDER):
        if i < current_idx:
            cls = "done"
            icon = "✓"
        elif i == current_idx:
            cls = "active"
            icon = STAGE_ICONS[stage]
        else:
            cls = "pending"
            icon = str(i + 1)

        parts.append(
            f'<div class="step">'
            f'<div class="step-circle {cls}">{icon}</div>'
            f'<div class="step-label {cls}">{STAGE_LABELS[stage]}</div>'
            f'</div>'
        )
        if i < len(STAGE_ORDER) - 1:
            connector_cls = "done" if i < current_idx else "pending"
            parts.append(f'<div class="step-connector {connector_cls}"></div>')

    st.markdown(f'<div class="stepper">{"".join(parts)}</div>', unsafe_allow_html=True)


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="main-header">Neo</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Your autonomous ML engineer</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="sidebar-section"><div class="sidebar-title">How it works</div></div>', unsafe_allow_html=True)
    st.markdown(
        """
1. Upload a CSV  
2. Describe what you want to predict  
3. Confirm the target column  
4. Agents train, tune, and explain the best model  
        """,
        unsafe_allow_html=False,
    )
    st.divider()

    st.markdown('<div class="sidebar-title">Tech stack</div>', unsafe_allow_html=True)
    tech = ["GPT-5.4", "LangGraph", "Optuna", "MLflow", "SHAP", "XGBoost", "LightGBM"]
    st.markdown(
        " ".join(f'<span class="tech-badge">{t}</span>' for t in tech),
        unsafe_allow_html=True,
    )
    st.divider()

    if st.button("Reset session", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    if st.session_state.stage == "results" and st.session_state.agent_state:
        state = st.session_state.agent_state
        st.divider()
        st.markdown('<div class="sidebar-title">Best model</div>', unsafe_allow_html=True)
        st.markdown(f"**{state.get('best_model_name', 'N/A')}**")
        for k, v in (state.get("best_metrics") or {}).items():
            st.markdown(f"`{k}`: **{v}**")


# ── main header ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">Neo: AutoML Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload data · describe your goal · get a tuned model</div>', unsafe_allow_html=True)

render_stepper(st.session_state.stage)
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — DATA INPUT
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.stage == "upload":
    from utils.data_loader import (
        SAMPLE_DATASETS, load_sample, load_from_url, load_from_text, load_from_database,
    )

    st.markdown(
        '<div class="hero-row">'
        '<div class="hero-item"><div class="hero-num">1</div><div class="hero-label">Upload a dataset</div></div>'
        '<div class="hero-sep">→</div>'
        '<div class="hero-item"><div class="hero-num">2</div><div class="hero-label">Describe your goal</div></div>'
        '<div class="hero-sep">→</div>'
        '<div class="hero-item"><div class="hero-num">3</div><div class="hero-label">Neo trains &amp; tunes 4 models</div></div>'
        '<div class="hero-sep">→</div>'
        '<div class="hero-item"><div class="hero-num">4</div><div class="hero-label">Get results, SHAP &amp; a download</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    tab_sample, tab_upload, tab_url, tab_paste, tab_db = st.tabs(
        ["Sample datasets", "Upload CSV", "From URL", "Paste CSV", "Database"]
    )

    # restore previously loaded df across reruns
    df: pd.DataFrame | None = st.session_state.get("staged_df", None)
    source_name: str = st.session_state.get("staged_source", "")

    # ── tab: sample datasets ─────────────────────────────────────────────────
    with tab_sample:
        st.markdown("Pick a built-in dataset to try the agent instantly. No file needed.")
        choice = st.selectbox(
            "Dataset",
            options=list(SAMPLE_DATASETS.keys()),
            label_visibility="collapsed",
        )
        st.caption(SAMPLE_DATASETS[choice]["description"])
        if st.button("Load dataset", key="load_sample"):
            with st.spinner("Loading…"):
                try:
                    df = load_sample(choice)
                    source_name = choice.split(":")[0].strip()
                    st.session_state.staged_df = df
                    st.session_state.staged_source = source_name
                except Exception as e:
                    st.error(f"Failed to load: {e}")

    # ── tab: file upload ─────────────────────────────────────────────────────
    with tab_upload:
        uploaded = st.file_uploader("Drop a CSV file here", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            source_name = uploaded.name
            st.session_state.staged_df = df
            st.session_state.staged_source = source_name

    # ── tab: url ─────────────────────────────────────────────────────────────
    with tab_url:
        st.markdown("Paste a link to any public CSV. GitHub, Google Sheets, and raw URLs all work.")
        url_input = st.text_input("URL", placeholder="https://raw.githubusercontent.com/…/data.csv")
        if st.button("Fetch", key="fetch_url") and url_input.strip():
            with st.spinner("Fetching…"):
                try:
                    df = load_from_url(url_input.strip())
                    source_name = url_input.strip().split("/")[-1] or "url_data"
                    st.session_state.staged_df = df
                    st.session_state.staged_source = source_name
                except Exception as e:
                    st.error(f"Could not load URL: {e}")

    # ── tab: paste ───────────────────────────────────────────────────────────
    with tab_paste:
        st.markdown("Paste CSV text directly. First row must be the header.")
        pasted = st.text_area("CSV content", height=200, placeholder="col1,col2,col3\n1,a,0.5\n2,b,0.8")
        if st.button("Parse", key="parse_paste") and pasted.strip():
            try:
                df = load_from_text(pasted)
                source_name = "pasted_data"
                st.session_state.staged_df = df
                st.session_state.staged_source = source_name
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")

    # ── tab: database ────────────────────────────────────────────────────────
    with tab_db:
        st.markdown("Connect to a database and run a query.")
        conn_col, _ = st.columns([2, 1])
        with conn_col:
            conn_str = st.text_input(
                "Connection string",
                placeholder="sqlite:///my_data.db  or  postgresql://user:pass@localhost/db",
            )
        query = st.text_area(
            "SQL query",
            value="SELECT * FROM my_table LIMIT 10000",
            height=80,
        )
        if st.button("Run query", key="run_query") and conn_str.strip():
            with st.spinner("Querying…"):
                try:
                    df = load_from_database(conn_str.strip(), query.strip())
                    source_name = "database_query"
                    st.session_state.staged_df = df
                    st.session_state.staged_source = source_name
                except Exception as e:
                    st.error(f"Database error: {e}")

    # ── preview + goal (shared across all tabs) ───────────────────────────────
    if df is not None:
        st.divider()
        st.success(f"**{source_name}** loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

        stat_cards([
            ("Rows", f"{df.shape[0]:,}"),
            ("Columns", str(df.shape[1])),
            ("Missing cells", f"{df.isnull().sum().sum():,}"),
            ("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.0f} KB"),
        ])

        with st.expander("Preview (first 5 rows)", expanded=True):
            st.dataframe(df.head(), use_container_width=True)

        goal = st.text_area(
            "What do you want to predict?",
            placeholder='e.g. "Predict which customers will churn" or "Forecast house prices"',
            height=100,
            key="staged_goal",
        )

        if st.button("Analyze Data", type="primary", disabled=not goal.strip()):
            with st.spinner("Analyzing your data…"):
                try:
                    from agents.data_agent import profile_dataframe, identify_target
                    profile = profile_dataframe(df)
                    identification = identify_target(profile, goal.strip())
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.stop()

            # only record the message after a successful API call
            add_message(
                "user",
                f"**Goal:** {goal.strip()}\n\n**Source:** {source_name} "
                f"({df.shape[0]:,} rows × {df.shape[1]} cols)",
            )
            st.session_state.df = df
            st.session_state.filename = source_name
            st.session_state.user_goal = goal.strip()
            st.session_state.suggested_target = identification["target_col"]
            st.session_state.suggested_problem_type = identification["problem_type"]
            st.session_state.suggested_reasoning = identification["reasoning"]
            st.session_state.stage = "confirm"
            st.session_state.pop("staged_df", None)
            st.session_state.pop("staged_source", None)
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — HUMAN-IN-THE-LOOP CONFIRMATION
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "confirm":
    render_messages()

    df = st.session_state.df

    with st.chat_message("assistant"):
        st.markdown(
            f"I analyzed your dataset. Here's what I found. **Please confirm or adjust** "
            f"before I start training:\n\n"
            f"> _{st.session_state.suggested_reasoning}_"
        )

    st.markdown('<div class="confirm-box">', unsafe_allow_html=True)
    col_target, col_type = st.columns(2)
    with col_target:
        st.markdown("**Target column**")
        confirmed_target = st.selectbox(
            "Target column",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(st.session_state.suggested_target)
            if st.session_state.suggested_target in df.columns else 0,
            label_visibility="collapsed",
        )
    with col_type:
        st.markdown("**Problem type**")
        confirmed_type = st.radio(
            "Problem type",
            options=["classification", "regression"],
            index=0 if st.session_state.suggested_problem_type == "classification" else 1,
            horizontal=True,
            label_visibility="collapsed",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    if confirmed_target:
        stat_cards([
            ("Unique values", str(df[confirmed_target].nunique())),
            ("Null %", f"{df[confirmed_target].isnull().mean() * 100:.1f}%"),
            ("Dtype", str(df[confirmed_target].dtype)),
        ])

        if confirmed_type == "classification":
            counts = df[confirmed_target].value_counts()
            fig_dist, ax_dist = plt.subplots(figsize=(6, 2.5))
            ax_dist.barh(
                [str(v) for v in counts.index],
                counts.values,
                color=PLOT_BROWN,
            )
            ax_dist.set_xlabel("Count")
            ax_dist.set_title("Class distribution")
            ax_dist.spines["top"].set_visible(False)
            ax_dist.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_dist, use_container_width=True)
            plt.close(fig_dist)

            # imbalance warning: majority class > 80 %
            majority_pct = counts.iloc[0] / counts.sum()
            if majority_pct > 0.80:
                st.warning(
                    f"Class imbalance detected: the majority class represents "
                    f"{majority_pct:.0%} of the data. Consider using AUC / F1 as "
                    f"your primary metric rather than accuracy."
                )

    from utils.data_quality import run_checks
    issues = run_checks(df, confirmed_target, confirmed_type)
    if issues:
        st.markdown("**Data quality checks**")
        for issue in issues:
            if issue["level"] == "error":
                st.error(issue["message"])
            else:
                st.warning(issue["message"])

    if st.button("Start Training", type="primary"):
        st.session_state.confirmed_target = confirmed_target
        st.session_state.confirmed_problem_type = confirmed_type
        add_message(
            "assistant",
            f"Confirmed: predicting `{confirmed_target}` ({confirmed_type}). Starting pipeline.",
        )
        st.session_state.stage = "running"
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — AGENTS RUNNING
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "running":
    render_messages()

    from graph.pipeline import run_pipeline

    with st.status("Running pipeline…", expanded=True) as pipeline_status:
        def stream_status(msg: str) -> None:
            add_message("assistant", msg)
            pipeline_status.write(msg)

        result = run_pipeline(
            df=st.session_state.df,
            user_goal=st.session_state.user_goal,
            status_callback=stream_status,
            confirmed_target_col=st.session_state.confirmed_target,
            confirmed_problem_type=st.session_state.confirmed_problem_type,
        )
        pipeline_status.update(label="Pipeline complete.", state="complete", expanded=False)

    st.session_state.agent_state = result
    st.session_state.stage = "results"
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "results":
    render_messages()

    state = st.session_state.agent_state
    if state is None:
        st.error("Something went wrong. No results returned.")
        st.stop()

    # ── engineered features ──────────────────────────────────────────────────
    engineered = state.get("engineered_features", [])
    if engineered:
        with st.expander(f"{len(engineered)} new features engineered", expanded=False):
            for f in engineered:
                st.markdown(f"**`{f['name']}`**: {f['rationale']}")
                st.code(f["expression"], language="python")

    # ── report ───────────────────────────────────────────────────────────────
    if state.get("report_md"):
        with st.chat_message("assistant"):
            st.markdown(state["report_md"])

    # ── why this model callout ────────────────────────────────────────────────
    best_name = state.get("best_model_name", "")
    best_metrics = state.get("best_metrics", {})
    top_features = state.get("top_features", [])
    if best_name and best_metrics:
        metric_highlights = " · ".join(f"**{k}**: {v}" for k, v in best_metrics.items())
        feature_highlights = ", ".join(f"`{f}`" for f in top_features[:3]) if top_features else "N/A"
        st.markdown(
            f'<div class="callout-box">'
            f'<div class="callout-title">Best model: {best_name}</div>'
            f'<p>{metric_highlights}</p>'
            f'<p>Top drivers: {feature_highlights}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── model comparison ─────────────────────────────────────────────────────
    if state.get("results_df") is not None:
        results_df = state["results_df"]
        metric_cols = [c for c in results_df.columns if c != "Model"]
        low_is_better = {"rmse", "mae"}

        def _style_leaderboard(df):
            styled = df.style.format(
                {c: "{:.4f}" for c in metric_cols if c in df.columns},
                na_rep="N/A",
            )
            for col in metric_cols:
                if col not in df.columns:
                    continue
                cmap = "RdYlGn_r" if col in low_is_better else "RdYlGn"
                styled = styled.background_gradient(subset=[col], cmap=cmap, axis=0)
            return styled

        with st.expander("Model comparison", expanded=True):
            st.dataframe(_style_leaderboard(results_df), use_container_width=True)

    # ── plots ─────────────────────────────────────────────────────────────────
    is_clf = st.session_state.confirmed_problem_type == "classification"

    # row 1: SHAP + confidence / actual-vs-predicted
    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        if state.get("shap_fig") is not None:
            with st.expander("Feature importance (SHAP)", expanded=True):
                st.pyplot(state["shap_fig"])
    with plot_col2:
        if state.get("confidence_fig") is not None:
            label = "Confidence distribution" if is_clf else "Actual vs Predicted"
            with st.expander(label, expanded=True):
                st.pyplot(state["confidence_fig"])

    # row 2: confusion matrix + ROC (classification) OR residuals (regression)
    if is_clf:
        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            if state.get("confusion_fig") is not None:
                with st.expander("Confusion matrix", expanded=True):
                    st.pyplot(state["confusion_fig"])
        with diag_col2:
            if state.get("roc_fig") is not None:
                with st.expander("ROC curve", expanded=True):
                    st.pyplot(state["roc_fig"])
    else:
        if state.get("residual_fig") is not None:
            with st.expander("Residual analysis", expanded=True):
                st.pyplot(state["residual_fig"])

    # ── prediction sandbox ────────────────────────────────────────────────────
    df_eng = state.get("df_engineered") if state.get("df_engineered") is not None else st.session_state.df
    target_col = st.session_state.confirmed_target
    problem_type = st.session_state.confirmed_problem_type

    if df_eng is not None and state.get("best_model") is not None:
        st.markdown("---")
        with st.expander("Prediction sandbox: try your own inputs", expanded=False):
            st.caption("Enter values for each feature and get a live prediction from the best model.")
            feature_cols = [c for c in df_eng.columns if c != target_col]

            with st.form("sandbox_form"):
                n_cols = 3
                col_groups = [feature_cols[i:i+n_cols] for i in range(0, len(feature_cols), n_cols)]
                input_data = {}
                for group in col_groups:
                    form_cols = st.columns(len(group))
                    for fc, col_name in zip(form_cols, group):
                        with fc:
                            series = df_eng[col_name].dropna()
                            if pd.api.types.is_numeric_dtype(series):
                                input_data[col_name] = st.number_input(
                                    col_name,
                                    value=float(series.median()),
                                    format="%.4g",
                                )
                            else:
                                opts = sorted(series.unique().tolist())
                                input_data[col_name] = st.selectbox(col_name, opts)

                predict_btn = st.form_submit_button("Run prediction", type="primary")

            if predict_btn:
                try:
                    from agents.data_agent import preprocess
                    input_row = pd.DataFrame([input_data])
                    input_row[target_col] = df_eng[target_col].iloc[0]
                    combined = pd.concat([df_eng, input_row], ignore_index=True)
                    X_all, _, _, _, _ = preprocess(combined, target_col, problem_type)
                    X_input = X_all[[-1]]
                    model = state["best_model"]
                    pred = model.predict(X_input)[0]
                    if problem_type == "classification" and hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X_input)[0]
                        conf = float(proba.max()) * 100
                        st.metric("Prediction", str(pred), delta=f"{conf:.1f}% confidence")
                    else:
                        st.metric("Prediction", f"{pred:.4g}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    # ── downloads ────────────────────────────────────────────────────────────
    st.markdown("---")
    dl1, dl2, dl3, dl4 = st.columns(4)

    with dl1:
        if state.get("model_bytes"):
            st.download_button(
                label="Download model (.pkl)",
                data=state["model_bytes"],
                file_name="best_model.pkl",
                mime="application/octet-stream",
                use_container_width=True,
            )
    with dl2:
        if state.get("report_md"):
            import re as _re

            def _md_to_html(md: str) -> str:
                lines = md.splitlines()
                html_lines = []
                for line in lines:
                    # headings
                    m = _re.match(r"^(#{1,4})\s+(.*)", line)
                    if m:
                        level = min(len(m.group(1)) + 1, 4)  # h2–h4
                        html_lines.append(f"<h{level}>{m.group(2)}</h{level}>")
                        continue
                    # horizontal rule
                    if _re.match(r"^---+$", line.strip()):
                        html_lines.append("<hr>")
                        continue
                    # blank line → paragraph break
                    if not line.strip():
                        html_lines.append("<p></p>")
                        continue
                    # inline: escape HTML, then apply markdown
                    text = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    text = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
                    text = _re.sub(r"\*(.+?)\*",     r"<em>\1</em>",         text)
                    text = _re.sub(r"`(.+?)`",        r"<code>\1</code>",     text)
                    # list items
                    if _re.match(r"^[-*]\s+", text):
                        text = "<li>" + _re.sub(r"^[-*]\s+", "", text) + "</li>"
                    else:
                        text = f"<p>{text}</p>"
                    html_lines.append(text)
                return "\n".join(html_lines)

            report_html = (
                "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
                "<title>Neo: Model Report</title><style>"
                "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
                "max-width:820px;margin:48px auto;color:#1c1917;line-height:1.7}"
                "h1,h2,h3,h4{color:#92400e;margin-top:1.8rem}"
                "strong{font-weight:600}"
                "code{background:#faf7f5;padding:2px 6px;border-radius:4px;font-size:.9em}"
                "hr{border:none;border-top:1px solid #e7e0da;margin:32px 0}"
                "li{margin:4px 0}"
                "table{border-collapse:collapse;width:100%}"
                "th,td{border:1px solid #e7e0da;padding:8px 12px;text-align:left}"
                "th{background:#faf7f5;font-weight:600}"
                f"</style></head><body>{_md_to_html(state['report_md'])}</body></html>"
            )
            st.download_button(
                label="Download report (.html)",
                data=report_html.encode(),
                file_name="neo_report.html",
                mime="text/html",
                use_container_width=True,
            )
    with dl3:
        if state.get("inference_zip"):
            st.download_button(
                label="Download inference package (.zip)",
                data=state["inference_zip"],
                file_name="inference_package.zip",
                mime="application/zip",
                help="best_model.pkl + predict.py + serve.py (FastAPI) + README",
                use_container_width=True,
            )
    with dl4:
        try:
            import mlflow  # noqa: F401
            if st.button("Open MLflow tracker", use_container_width=True):
                subprocess.Popen(
                    ["mlflow", "ui", "--port", "5001"],
                    cwd="/Users/khushichoudhary/Neo",
                )
                st.info("Opening [http://localhost:5001](http://localhost:5001) in a new tab.")
        except ImportError:
            pass
