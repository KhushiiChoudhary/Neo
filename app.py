from __future__ import annotations

import subprocess
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AutoML Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── base ── */
    .stApp { background-color: #ffffff; }
    .block-container { padding-top: 1.5rem; }

    /* ── header ── */
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #92400e, #b45309);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .sub-header { color: #78716c; font-size: 0.9rem; }

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
    st.markdown('<div class="main-header">AutoML Agent</div>', unsafe_allow_html=True)
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
    tech = ["GPT-4o", "LangGraph", "Optuna", "MLflow", "SHAP", "XGBoost", "LightGBM"]
    st.markdown(
        " ".join(f'<span class="tech-badge">{t}</span>' for t in tech),
        unsafe_allow_html=True,
    )
    st.divider()

    if st.button("↺ Reset session", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    if st.session_state.stage == "results" and st.session_state.agent_state:
        state = st.session_state.agent_state
        st.divider()
        st.markdown('<div class="sidebar-title">Best model</div>', unsafe_allow_html=True)
        st.markdown(f"**{state.get('best_model_name', '—')}**")
        for k, v in (state.get("best_metrics") or {}).items():
            st.markdown(f"`{k}`: **{v}**")


# ── main header ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">AutoML Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a CSV · describe your goal · get a tuned model</div>', unsafe_allow_html=True)

render_stepper(st.session_state.stage)
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — DATA INPUT
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.stage == "upload":
    from utils.data_loader import (
        SAMPLE_DATASETS, load_sample, load_from_url, load_from_text, load_from_database,
    )

    tab_sample, tab_upload, tab_url, tab_paste, tab_db = st.tabs(
        ["✨ Sample datasets", "📁 Upload CSV", "🌐 From URL", "📋 Paste CSV", "🗄️ Database"]
    )

    df: pd.DataFrame | None = None
    source_name: str = ""

    # ── tab: sample datasets ─────────────────────────────────────────────────
    with tab_sample:
        st.markdown("Pick a built-in dataset to try the agent instantly — no file needed.")
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
                    source_name = choice.split("—")[0].strip()
                except Exception as e:
                    st.error(f"Failed to load: {e}")

    # ── tab: file upload ─────────────────────────────────────────────────────
    with tab_upload:
        uploaded = st.file_uploader("Drop a CSV file here", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            source_name = uploaded.name

    # ── tab: url ─────────────────────────────────────────────────────────────
    with tab_url:
        st.markdown("Paste a link to any public CSV. GitHub, Google Sheets, and raw URLs all work.")
        url_input = st.text_input("URL", placeholder="https://raw.githubusercontent.com/…/data.csv")
        if st.button("Fetch", key="fetch_url") and url_input.strip():
            with st.spinner("Fetching…"):
                try:
                    df = load_from_url(url_input.strip())
                    source_name = url_input.strip().split("/")[-1] or "url_data"
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
                except Exception as e:
                    st.error(f"Database error: {e}")

    # ── preview + goal (shared across all tabs) ───────────────────────────────
    if df is not None:
        st.divider()
        st.success(f"**{source_name}** loaded — {df.shape[0]:,} rows × {df.shape[1]} columns")

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
        )

        if st.button("🔍 Analyze Data", type="primary", disabled=not goal.strip()):
            st.session_state.df = df
            st.session_state.filename = source_name
            st.session_state.user_goal = goal.strip()
            add_message(
                "user",
                f"**Goal:** {goal.strip()}\n\n**Source:** {source_name} "
                f"({df.shape[0]:,} rows × {df.shape[1]} cols)",
            )
            with st.spinner("Analyzing your data…"):
                from agents.data_agent import profile_dataframe, identify_target
                profile = profile_dataframe(df)
                identification = identify_target(profile, goal.strip())

            st.session_state.suggested_target = identification["target_col"]
            st.session_state.suggested_problem_type = identification["problem_type"]
            st.session_state.suggested_reasoning = identification["reasoning"]
            st.session_state.stage = "confirm"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — HUMAN-IN-THE-LOOP CONFIRMATION
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "confirm":
    render_messages()

    df = st.session_state.df

    with st.chat_message("assistant"):
        st.markdown(
            f"I analyzed your dataset. Here's what I found — **please confirm or adjust** "
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

    from utils.data_quality import run_checks
    issues = run_checks(df, confirmed_target, confirmed_type)
    if issues:
        st.markdown("**Data quality checks**")
        for issue in issues:
            if issue["level"] == "error":
                st.error(issue["message"])
            else:
                st.warning(issue["message"])

    if st.button("🚀 Start Training", type="primary"):
        st.session_state.confirmed_target = confirmed_target
        st.session_state.confirmed_problem_type = confirmed_type
        add_message(
            "assistant",
            f"✅ **Confirmed:** predicting `{confirmed_target}` ({confirmed_type}). Starting pipeline…",
        )
        st.session_state.stage = "running"
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — AGENTS RUNNING
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "running":
    render_messages()

    from graph.pipeline import run_pipeline

    def stream_status(msg: str) -> None:
        add_message("assistant", msg)
        with st.chat_message("assistant"):
            st.markdown(msg)

    with st.spinner("Running pipeline…"):
        result = run_pipeline(
            df=st.session_state.df,
            user_goal=st.session_state.user_goal,
            status_callback=stream_status,
            confirmed_target_col=st.session_state.confirmed_target,
            confirmed_problem_type=st.session_state.confirmed_problem_type,
        )

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
        st.error("Something went wrong — no results returned.")
        st.stop()

    # ── engineered features ──────────────────────────────────────────────────
    engineered = state.get("engineered_features", [])
    if engineered:
        with st.expander(f"✨ {len(engineered)} new features engineered", expanded=False):
            for f in engineered:
                st.markdown(f"**`{f['name']}`** — {f['rationale']}")
                st.code(f["expression"], language="python")

    # ── report ───────────────────────────────────────────────────────────────
    if state.get("report_md"):
        with st.chat_message("assistant"):
            st.markdown(state["report_md"])

    # ── model comparison ─────────────────────────────────────────────────────
    if state.get("results_df") is not None:
        with st.expander("Model comparison", expanded=True):
            st.dataframe(state["results_df"], use_container_width=True)

    # ── plots side by side ───────────────────────────────────────────────────
    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        if state.get("shap_fig") is not None:
            with st.expander("Feature importance (SHAP)", expanded=True):
                st.pyplot(state["shap_fig"])
    with plot_col2:
        if state.get("confidence_fig") is not None:
            is_clf = st.session_state.confirmed_problem_type == "classification"
            label = "Confidence distribution" if is_clf else "Actual vs Predicted"
            with st.expander(label, expanded=True):
                st.pyplot(state["confidence_fig"])

    # ── downloads ────────────────────────────────────────────────────────────
    st.markdown("---")
    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        if state.get("model_bytes"):
            st.download_button(
                label="⬇️ Model (.pkl)",
                data=state["model_bytes"],
                file_name="best_model.pkl",
                mime="application/octet-stream",
                use_container_width=True,
            )
    with dl2:
        if state.get("inference_zip"):
            st.download_button(
                label="⬇️ Inference package (.zip)",
                data=state["inference_zip"],
                file_name="inference_package.zip",
                mime="application/zip",
                help="best_model.pkl + predict.py + serve.py (FastAPI) + README",
                use_container_width=True,
            )
    with dl3:
        try:
            import mlflow  # noqa: F401
            if st.button("📊 Open MLflow tracker", use_container_width=True):
                subprocess.Popen(
                    ["mlflow", "ui", "--port", "5001"],
                    cwd="/Users/khushichoudhary/Neo",
                )
                st.info("Opening [http://localhost:5001](http://localhost:5001) — check a new tab.")
        except ImportError:
            pass
