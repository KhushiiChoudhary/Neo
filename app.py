from __future__ import annotations

import re
import subprocess
from contextlib import contextmanager
from html import escape

import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

PLOT_ACCENT = "#4f46e5"

load_dotenv()

st.set_page_config(
    page_title="Neo: AutoML Agent",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    :root {
        --neo-bg: #FAFAF8;
        --neo-panel: #FFFFFF;
        --neo-panel-strong: #FFFFFF;
        --neo-border: #E8E6E0;
        --neo-border-strong: #D0CEC8;
        --neo-text: #1A1A18;
        --neo-muted: #6B6A66;
        --neo-soft: #9A9892;
        --neo-accent: #5C4EE5;
        --neo-accent-soft: #6366F1;
        --neo-accent-bg: #EEEDFE;
        --neo-accent-text: #534AB7;
    }

    /* ── base ── */
    .stApp {
        background: var(--neo-bg);
        color: var(--neo-text);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header[data-testid="stHeader"] { display: none; }
    section[data-testid="stSidebar"] { display: none; }
    div[data-testid="collapsedControl"] { display: none; }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1120px;
        margin: 0 auto;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .stMarkdown p,
    .stMarkdown span,
    .stMarkdown div,
    label,
    .stCaption {
        color: var(--neo-text);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    div[data-testid="stCaptionContainer"] p {
        color: var(--neo-muted) !important;
        font-size: 13px !important;
        line-height: 1.5 !important;
    }
    div[data-testid="stVerticalBlock"] > div:has(> .neo-shell) {
        gap: 0;
    }

    /* ── header ── */
    .neo-shell {
        border: 1px solid var(--neo-border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        background: var(--neo-panel);
    }
    .neo-header-grid {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        gap: 0.4rem;
    }
    .neo-brand-name {
        font-size: clamp(1.2rem, 2vw, 1.6rem);
        font-weight: 400;
        letter-spacing: -0.01em;
        color: var(--neo-accent);
    }
    .neo-title {
        margin: 0;
        font-size: clamp(1rem, 1.8vw, 1.2rem);
        font-weight: 400;
        line-height: 1.5;
        letter-spacing: -0.01em;
        color: var(--neo-muted);
    }
    /* ── progress stepper ── */
    .stepper {
        display: flex;
        align-items: center;
        margin: 0 0 1.5rem 0;
        padding: 1rem 1.1rem;
        border-radius: 12px;
        border: 1px solid var(--neo-border);
        background: var(--neo-panel);
    }
    .step { display: flex; flex-direction: column; align-items: center; flex: 1; }
    .step-circle {
        width: 36px; height: 36px; border-radius: 999px;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.82rem; font-weight: 500;
        border: 1px solid var(--neo-border-strong);
    }
    .step-circle.done {
        background: var(--neo-accent);
        color: #fff;
        border-color: var(--neo-accent);
    }
    .step-circle.active {
        background: var(--neo-accent);
        color: #fff;
        border-color: var(--neo-accent);
    }
    .step-circle.pending {
        background: #FFFFFF;
        color: var(--neo-soft);
        border: 1px solid var(--neo-border-strong);
    }
    .step-label {
        font-size: 12px;
        margin-top: 0.45rem;
        text-align: center;
        font-weight: 400;
    }
    .step-label.done    { color: var(--neo-accent); }
    .step-label.active  { color: var(--neo-accent); font-weight: 500; }
    .step-label.pending { color: var(--neo-muted); }
    .step-connector {
        flex: 1;
        height: 1px;
        margin: 0 0.75rem;
        background: var(--neo-border);
    }
    .step-connector.done {
        background: var(--neo-border);
    }

    /* ── stat cards ── */
    .stat-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.75rem;
        margin: 1rem 0 1.25rem 0;
    }
    .stat-card {
        background: #FFFFFF;
        border: 1px solid var(--neo-border);
        border-radius: 12px;
        padding: 1rem;
    }
    .stat-value { font-size: 1.3rem; font-weight: 500; color: var(--neo-text); }
    .stat-label {
        font-size: 0.74rem;
        color: var(--neo-muted);
        margin-top: 0.35rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* ── panels ── */
    .neo-section-title {
        margin: 0 0 0.5rem 0;
        color: var(--neo-muted);
        font-size: 13px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .neo-section-copy {
        margin: 0 0 0.6rem 0;
        color: var(--neo-muted);
        font-size: 0.86rem;
        line-height: 1.45;
    }
    .neo-card {
        border-radius: 12px;
        border: 1px solid var(--neo-border);
        background: var(--neo-panel);
        padding: 1.15rem 1.2rem;
        margin-bottom: 1rem;
    }
    .neo-card.soft {
        background: var(--neo-panel);
    }
    .neo-callout {
        border-radius: 12px;
        border: 1px solid var(--neo-border);
        padding: 1.2rem 1.25rem;
        background: #FFFFFF;
        color: var(--neo-text);
    }
    .neo-callout h3 {
        margin: 0;
        font-size: 1.05rem;
        font-weight: 500;
        color: var(--neo-text);
    }
    .neo-callout p {
        margin: 0.55rem 0 0 0;
        color: var(--neo-muted);
        line-height: 1.6;
        font-size: 0.92rem;
    }
    .neo-pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 0.95rem;
    }
    .neo-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 6px 14px;
        border-radius: 999px;
        border: none;
        background: var(--neo-accent-bg);
        color: var(--neo-accent-text);
        font-size: 0.8rem;
        font-weight: 500;
    }

    /* ── pipeline loaders ── */
    .neo-loader-card {
        border-radius: 12px;
        border: 1px solid var(--neo-border);
        background: #FFFFFF;
        padding: 1.25rem 1.3rem;
        margin-bottom: 1rem;
    }
    .neo-loader-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
    }
    .neo-loader-stack {
        display: flex;
        align-items: center;
        gap: 0.95rem;
    }
    .neo-loader-orb {
        position: relative;
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: linear-gradient(135deg, var(--neo-accent), var(--neo-accent-soft));
        overflow: hidden;
    }
    .neo-loader-orb::before,
    .neo-loader-orb::after {
        content: "";
        position: absolute;
        border-radius: 999px;
        background: rgba(255,255,255,0.86);
    }
    .neo-loader-orb::before {
        width: 18px;
        height: 18px;
        top: 10px;
        left: 18px;
        animation: loaderFloat 1.7s ease-in-out infinite;
    }
    .neo-loader-orb::after {
        width: 8px;
        height: 8px;
        bottom: 11px;
        right: 12px;
        animation: loaderPulse 1.2s ease-in-out infinite;
    }
    .neo-loader-kicker {
        color: var(--neo-soft);
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }
    .neo-loader-title {
        color: var(--neo-text);
        font-size: 1rem;
        font-weight: 500;
        margin: 0;
    }
    .neo-loader-copy {
        color: var(--neo-muted);
        font-size: 0.88rem;
        line-height: 1.6;
        margin: 0.35rem 0 0 0;
    }
    .neo-loader-progress {
        min-width: 110px;
        text-align: right;
        color: var(--neo-text);
        font-size: 1.35rem;
        font-weight: 500;
        letter-spacing: -0.04em;
    }
    .neo-progress-track {
        width: 100%;
        height: 8px;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.08);
        overflow: hidden;
        margin-top: 1rem;
    }
    .neo-progress-fill {
        height: 100%;
        border-radius: 999px;
        background: var(--neo-accent);
        transition: width 0.25s ease;
    }
    .neo-log-hint {
        margin: 0.15rem 0 0 0;
        color: var(--neo-soft);
        font-size: 0.78rem;
    }

    /* ── native widgets ── */
    .stButton > button,
    .stDownloadButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        min-height: 2.6rem;
        border-radius: 8px;
        border: 1px solid var(--neo-accent);
        background: var(--neo-accent);
        color: white;
        font-weight: 500;
        padding: 0.75rem 1.25rem;
        box-shadow: none !important;
        transition: background 0.18s ease, border-color 0.18s ease, color 0.18s ease;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #534AB7;
        border-color: #534AB7;
    }
    div[data-testid="column"]:has(.neo-reset-anchor) .stButton > button {
        background: transparent;
        border-color: var(--neo-accent);
        color: var(--neo-accent);
    }
    div[data-testid="column"]:has(.neo-reset-anchor) .stButton > button:hover {
        background: var(--neo-accent-bg);
        border-color: var(--neo-accent);
        color: var(--neo-accent-text);
    }
    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox [data-baseweb="select"] > div,
    .stMultiSelect [data-baseweb="select"] > div,
    .stNumberInput input {
        border-radius: 10px !important;
        border: 1px solid #E0DED8 !important;
        background: #FFFFFF !important;
        color: var(--neo-text) !important;
        box-shadow: none !important;
    }
    .stSelectbox svg,
    .stMultiSelect svg {
        color: var(--neo-accent) !important;
        fill: var(--neo-accent) !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.25rem;
        background: transparent;
        border-bottom: 1px solid var(--neo-border);
    }
    .stTabs [data-baseweb="tab"] {
        height: auto;
        padding: 0 0 0.8rem 0;
        margin-bottom: -1px;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0;
        background: transparent !important;
        color: var(--neo-muted) !important;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: var(--neo-accent) !important;
        border-bottom-color: var(--neo-accent) !important;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--neo-text) !important;
        border-bottom-color: #D9D6FF !important;
    }
    div[data-testid="stExpander"] {
        border: 1px solid var(--neo-border) !important;
        border-radius: 12px !important;
        background: #FFFFFF !important;
        overflow: hidden;
        box-shadow: none !important;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--neo-border);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: none !important;
    }
    div[data-testid="stMetric"] {
        border: 1px solid var(--neo-border);
        border-radius: 12px;
        background: #FFFFFF;
        padding: 0.7rem;
        box-shadow: none !important;
    }
    div[data-testid="stStatusWidget"] {
        border-radius: 12px;
        border: 1px solid var(--neo-border);
        background: #FFFFFF;
        box-shadow: none !important;
    }
    @keyframes loaderFloat {
        0%, 100% { transform: translateY(0px); opacity: 0.9; }
        50% { transform: translateY(14px); opacity: 0.45; }
    }
    @keyframes loaderPulse {
        0%, 100% { transform: scale(0.9); opacity: 0.55; }
        50% { transform: scale(1.3); opacity: 1; }
    }
    .stButton > button:focus-visible,
    .stDownloadButton > button:focus-visible,
    div[data-testid="stFormSubmitButton"] > button:focus-visible,
    .stTextInput input:focus,
    .stTextArea textarea:focus,
    .stSelectbox [data-baseweb="select"] > div:focus-within,
    .stMultiSelect [data-baseweb="select"] > div:focus-within,
    .stNumberInput input:focus {
        outline: none !important;
        border-color: var(--neo-accent) !important;
        box-shadow: 0 0 0 2px rgba(92, 78, 229, 0.18) !important;
    }
    .stTabs [data-baseweb="tab"]:focus-visible {
        outline: none !important;
        border-bottom-color: var(--neo-accent) !important;
        box-shadow: none !important;
    }

    @media (max-width: 900px) {
        .neo-shell {
            padding: 1.25rem;
        }
        .neo-loader-row { flex-direction: column; align-items: flex-start; }
        .neo-loader-progress { text-align: left; }
        .stepper { gap: 0.3rem; }
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
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
        f'<div class="stat-card"><div class="stat-value">{escape(v)}</div><div class="stat-label">{escape(l)}</div></div>'
        for l, v in stats
    )
    st.markdown(f'<div class="stat-row">{cards_html}</div>', unsafe_allow_html=True)


STAGE_ORDER = ["upload", "confirm", "running", "results"]
STAGE_LABELS = {"upload": "Upload", "confirm": "Confirm", "running": "Training", "results": "Results"}
STAGE_ICONS  = {"upload": "1", "confirm": "2", "running": "3", "results": "✓"}
MODEL_LABELS = {
    "LogisticRegression": "Logistic Regression",
    "RandomForest": "Random Forest",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "Ridge": "Ridge",
}


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

def render_section_header(title: str, copy: str | None = None) -> None:
    body = f'<div class="neo-section-title">{escape(title)}</div>'
    if copy:
        body += f'<p class="neo-section-copy">{escape(copy)}</p>'
    st.markdown(body, unsafe_allow_html=True)


def render_feature_band(items: list[tuple[str, str, str]]) -> None:
    cards = "".join(
        (
            '<div class="neo-feature-card">'
            f'<div class="neo-feature-kicker">{escape(kicker)}</div>'
            f'<div class="neo-feature-title">{escape(title)}</div>'
            f'<p class="neo-feature-copy">{escape(copy)}</p>'
            '</div>'
        )
        for kicker, title, copy in items
    )
    st.markdown(f'<div class="neo-feature-grid">{cards}</div>', unsafe_allow_html=True)


def render_metric_pills(items: list[tuple[str, str]]) -> None:
    pills = "".join(
        f'<div class="neo-pill"><span>{escape(label)}</span><span>{escape(value)}</span></div>'
        for label, value in items
    )
    st.markdown(f'<div class="neo-pill-row">{pills}</div>', unsafe_allow_html=True)


def render_loader_markup(title: str, detail: str, progress: int | None = None, kicker: str = "Working live") -> str:
    progress_markup = (
        f'<div class="neo-loader-progress">{progress}%</div>'
        if progress is not None
        else '<div class="neo-loader-progress">Live</div>'
    )
    track_markup = (
        '<div class="neo-progress-track">'
        f'<div class="neo-progress-fill" style="width:{progress}%"></div>'
        '</div>'
        if progress is not None
        else ""
    )
    return (
        '<div class="neo-loader-card">'
        '<div class="neo-loader-row">'
        '<div class="neo-loader-stack">'
        '<div class="neo-loader-orb"></div>'
        '<div>'
        f'<div class="neo-loader-kicker">{escape(kicker)}</div>'
        f'<div class="neo-loader-title">{escape(title)}</div>'
        f'<p class="neo-loader-copy">{escape(detail)}</p>'
        '</div>'
        '</div>'
        f'{progress_markup}'
        '</div>'
        f'{track_markup}'
        '<p class="neo-log-hint">Updates stream live.</p>'
        '</div>'
    )


@contextmanager
def branded_loader(title: str, detail: str, kicker: str = "Loading") -> None:
    placeholder = st.empty()
    placeholder.markdown(
        render_loader_markup(title, detail, None, kicker=kicker),
        unsafe_allow_html=True,
    )
    with st.spinner(title):
        yield
    placeholder.empty()


def pipeline_model_order(problem_type: str | None) -> list[str]:
    if problem_type == "classification":
        return ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM"]
    return ["Ridge", "RandomForest", "XGBoost", "LightGBM"]


def derive_pipeline_status(message: str, progress_state: dict) -> dict:
    clean = re.sub(r"\s+", " ", re.sub(r"\*\*", "", message)).strip()
    progress = progress_state.get("progress", 6)
    title = progress_state.get("title", "Starting your pipeline")
    detail = clean

    message_lower = clean.lower()
    if "profiling dataset" in message_lower or "target column confirmed" in message_lower or "target column identified" in message_lower:
        title = "Profiling data"
        progress = max(progress, 14)
    elif "feature engineering" in message_lower or "new features created" in message_lower:
        title = "Engineering features"
        progress = max(progress, 32)
    elif clean.startswith("Tuning") or clean.startswith("Optimizing"):
        match = re.search(r"\*\*(.+?)\*\*", message)
        model_key = match.group(1) if match else clean.split(":")[0].replace("Tuning ", "").replace("Optimizing ", "")
        model_order = pipeline_model_order(st.session_state.get("confirmed_problem_type"))
        if model_key in model_order:
            idx = model_order.index(model_key)
            title = f"Tuning {MODEL_LABELS.get(model_key, model_key)}"
            progress = max(progress, 42 + idx * 11)
        trial_match = re.search(r"trial\s+(\d+)/(\d+)", clean.lower())
        if trial_match and model_key in model_order:
            idx = model_order.index(model_key)
            trial_num = int(trial_match.group(1))
            trial_total = max(int(trial_match.group(2)), 1)
            stage_base = 42 + idx * 11
            stage_span = 10
            progress = max(progress, min(86, stage_base + int((trial_num / trial_total) * stage_span)))
            title = f"Tuning {MODEL_LABELS.get(model_key, model_key)}"
    elif clean.startswith("Best model:"):
        title = "Selecting best model"
        progress = max(progress, 90)
    elif "reporter agent" in message_lower:
        title = "Building report"
        progress = max(progress, 94)
    elif "generating shap" in message_lower:
        title = "Rendering SHAP"
        progress = max(progress, 96)
    elif "generating confidence" in message_lower or "actual vs predicted" in message_lower:
        title = "Preparing charts"
        progress = max(progress, 97)
    elif "generating confusion matrix" in message_lower or "generating roc" in message_lower or "generating residual" in message_lower:
        title = "Finalizing diagnostics"
        progress = max(progress, 98)
    elif "writing report" in message_lower:
        title = "Writing report"
        progress = max(progress, 99)

    return {"title": title, "detail": clean, "progress": min(progress, 99)}


def render_page_header() -> None:
    st.markdown(
        """
        <div class="neo-shell">
            <div class="neo-header-grid">
                <div class="neo-brand-name">Neo - Auto ML</div>
                <h1 class="neo-title">Upload data. Let Neo handle the rest.</h1>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer_actions() -> None:
    _, footer_col = st.columns([9, 1.4])
    with footer_col:
        st.markdown('<div class="neo-reset-anchor"></div>', unsafe_allow_html=True)
        if st.button("Reset session", width="stretch"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ── main header ───────────────────────────────────────────────────────────────
render_page_header()
render_stepper(st.session_state.stage)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — DATA INPUT
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.stage == "upload":
    from utils.data_loader import (
        SAMPLE_DATASETS, load_sample, load_from_url, load_from_text, load_from_database,
    )

    render_section_header("Choose a dataset")

    tab_sample, tab_upload, tab_url, tab_paste, tab_db = st.tabs(
        ["Sample datasets", "Upload CSV", "From URL", "Paste CSV", "Database"]
    )

    # restore previously loaded df across reruns
    df: pd.DataFrame | None = st.session_state.get("staged_df", None)
    source_name: str = st.session_state.get("staged_source", "")

    # ── tab: sample datasets ─────────────────────────────────────────────────
    with tab_sample:
        st.caption("Try a built-in dataset.")
        choice = st.selectbox(
            "Dataset",
            options=list(SAMPLE_DATASETS.keys()),
            label_visibility="collapsed",
        )
        st.caption(SAMPLE_DATASETS[choice]["description"])
        if st.button("Load dataset", key="load_sample"):
            with branded_loader(
                "Loading sample dataset",
                    "Preparing sample data.",
            ):
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
        st.caption("Paste a public CSV URL.")
        url_input = st.text_input("URL", placeholder="https://raw.githubusercontent.com/…/data.csv")
        if st.button("Fetch", key="fetch_url") and url_input.strip():
            with branded_loader(
                "Fetching remote CSV",
                    "Fetching data.",
            ):
                try:
                    df = load_from_url(url_input.strip())
                    source_name = url_input.strip().split("/")[-1] or "url_data"
                    st.session_state.staged_df = df
                    st.session_state.staged_source = source_name
                except Exception as e:
                    st.error(f"Could not load URL: {e}")

    # ── tab: paste ───────────────────────────────────────────────────────────
    with tab_paste:
        st.caption("Paste raw CSV text.")
        pasted = st.text_area("CSV content", height=200, placeholder="col1,col2,col3\n1,a,0.5\n2,b,0.8")
        if st.button("Parse", key="parse_paste") and pasted.strip():
            with branded_loader(
                "Parsing pasted rows",
                "Parsing data.",
            ):
                try:
                    df = load_from_text(pasted)
                    source_name = "pasted_data"
                    st.session_state.staged_df = df
                    st.session_state.staged_source = source_name
                except Exception as e:
                    st.error(f"Could not parse CSV: {e}")

    # ── tab: database ────────────────────────────────────────────────────────
    with tab_db:
        st.caption("Run a database query.")
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
            with branded_loader(
                "Running database query",
                "Running query.",
            ):
                try:
                    df = load_from_database(conn_str.strip(), query.strip())
                    source_name = "database_query"
                    st.session_state.staged_df = df
                    st.session_state.staged_source = source_name
                except Exception as e:
                    st.error(f"Database error: {e}")

    # ── preview + goal (shared across all tabs) ───────────────────────────────
    if df is not None:
        render_section_header("Dataset ready")
        st.success(f"**{source_name}** loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

        stat_cards([
            ("Rows", f"{df.shape[0]:,}"),
            ("Columns", str(df.shape[1])),
            ("Missing cells", f"{df.isnull().sum().sum():,}"),
            ("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.0f} KB"),
        ])

        preview_col, goal_col = st.columns([1.15, 0.85], vertical_alignment="top")
        with preview_col:
            with st.expander("Preview", expanded=True):
                st.dataframe(df.head(), width="stretch")
        with goal_col:
            render_section_header("Goal")
            goal = st.text_area(
                "What do you want to predict?",
                placeholder='e.g. "Predict which customers will churn" or "Forecast house prices"',
                height=140,
                key="staged_goal",
            )
            if st.button("Analyze Data", type="primary", disabled=not goal.strip()):
                with branded_loader(
                    "Analyzing your dataset",
                    "Inferring the target.",
                    kicker="Analyzing",
                ):
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

    render_section_header("Confirm setup")

    st.markdown(
        f"""
        <div class="neo-callout">
            <h3>{escape(str(st.session_state.suggested_target or "Suggested target"))}</h3>
            <p>{escape(st.session_state.suggested_reasoning or "")}</p>
            <div class="neo-pill-row">
                <div class="neo-pill"><span>Problem type</span><span>{escape(str(st.session_state.suggested_problem_type or "pending"))}</span></div>
                <div class="neo-pill"><span>Rows</span><span>{df.shape[0]:,}</span></div>
                <div class="neo-pill"><span>Columns</span><span>{df.shape[1]}</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    control_col, insight_col = st.columns([0.95, 1.05], vertical_alignment="top")
    with control_col:
        render_section_header("Target")
        confirmed_target = st.selectbox(
            "Target column",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(st.session_state.suggested_target)
            if st.session_state.suggested_target in df.columns else 0,
        )
        confirmed_type = st.radio(
            "Problem type",
            options=["classification", "regression"],
            index=0 if st.session_state.suggested_problem_type == "classification" else 1,
            horizontal=True,
        )

        from utils.data_quality import run_checks
        issues = run_checks(df, confirmed_target, confirmed_type)
        if issues:
            render_section_header("Checks")
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

    with insight_col:
        if confirmed_target:
            render_section_header("Snapshot")
            stat_cards([
                ("Unique values", str(df[confirmed_target].nunique())),
                ("Null %", f"{df[confirmed_target].isnull().mean() * 100:.1f}%"),
                ("Dtype", str(df[confirmed_target].dtype)),
            ])

            if confirmed_type == "classification":
                counts = df[confirmed_target].value_counts()
                fig_dist, ax_dist = plt.subplots(figsize=(6, 2.8))
                ax_dist.barh(
                    [str(v) for v in counts.index],
                    counts.values,
                    color=PLOT_ACCENT,
                )
                ax_dist.set_xlabel("Count")
                ax_dist.set_title("Class distribution")
                ax_dist.spines["top"].set_visible(False)
                ax_dist.spines["right"].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_dist, width="stretch")
                plt.close(fig_dist)

                majority_pct = counts.iloc[0] / counts.sum()
                if majority_pct > 0.80:
                    st.warning(
                        f"Class imbalance detected: the majority class represents "
                        f"{majority_pct:.0%} of the data. Consider using AUC / F1 as "
                        f"your primary metric rather than accuracy."
                    )

    if not df.columns.tolist():
        st.warning("No columns available to confirm.")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — AGENTS RUNNING
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "running":
    render_messages()

    render_section_header("Training")

    pipeline_state = {
        "title": "Preparing your run",
        "detail": "Setting up the pipeline.",
        "progress": 8,
    }
    pipeline_panel = st.empty()
    pipeline_panel.markdown(
        render_loader_markup(
            pipeline_state["title"],
            pipeline_state["detail"],
            pipeline_state["progress"],
            kicker="Training live",
        ),
        unsafe_allow_html=True,
    )

    from graph.pipeline import run_pipeline

    with st.status("Live agent log", expanded=True) as pipeline_status:
        def stream_status(msg: str) -> None:
            add_message("assistant", msg)
            pipeline_status.write(msg)
            pipeline_state.update(derive_pipeline_status(msg, pipeline_state))
            pipeline_panel.markdown(
                render_loader_markup(
                    pipeline_state["title"],
                    pipeline_state["detail"],
                    pipeline_state["progress"],
                    kicker="Training live",
                ),
                unsafe_allow_html=True,
            )

        result = run_pipeline(
            df=st.session_state.df,
            user_goal=st.session_state.user_goal,
            status_callback=stream_status,
            confirmed_target_col=st.session_state.confirmed_target,
            confirmed_problem_type=st.session_state.confirmed_problem_type,
        )
        pipeline_status.update(label="Pipeline complete.", state="complete", expanded=False)

    pipeline_panel.markdown(
        render_loader_markup(
            "Pipeline complete",
            "Model and artifacts are ready.",
            100,
            kicker="Complete",
        ),
        unsafe_allow_html=True,
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
        st.error("Something went wrong. No results returned.")
        st.stop()

    best_name = state.get("best_model_name", "")
    best_metrics = state.get("best_metrics", {})
    top_features = state.get("top_features", [])

    render_section_header("Results")

    metric_pills = [(k.upper(), str(v)) for k, v in best_metrics.items()]
    feature_pills = [("Top driver", feature) for feature in top_features[:3]]
    pills_html = "".join(
        f'<div class="neo-pill"><span>{escape(label)}</span><span>{escape(value)}</span></div>'
        for label, value in [("Best model", best_name or "N/A"), *metric_pills, *feature_pills]
    )
    st.markdown(
        (
            '<div class="neo-callout">'
            f'<h3>{escape(best_name or "Model ready")}</h3>'
            '<p>Neo selected this candidate after benchmarking the baseline and tuned models, then generated diagnostics and exportable artifacts around the winner.</p>'
            f'<div class="neo-pill-row">{pills_html}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    engineered = state.get("engineered_features", [])
    if engineered:
        render_section_header("Features")
        with st.expander(f"{len(engineered)} engineered features", expanded=False):
            for f in engineered:
                st.markdown(f"**`{f['name']}`**: {f['rationale']}")
                st.code(f["expression"], language="python")

    if state.get("report_md"):
        render_section_header("Summary")
        st.markdown(state["report_md"])

    if state.get("results_df") is not None:
        render_section_header("Leaderboard")
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
            st.dataframe(_style_leaderboard(results_df), width="stretch")

    render_section_header("Diagnostics")
    is_clf = st.session_state.confirmed_problem_type == "classification"

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

    df_eng = state.get("df_engineered") if state.get("df_engineered") is not None else st.session_state.df
    target_col = st.session_state.confirmed_target
    problem_type = st.session_state.confirmed_problem_type

    if df_eng is not None and state.get("best_model") is not None:
        render_section_header("Prediction sandbox")
        with st.expander("Open prediction sandbox", expanded=False):
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

    render_section_header("Exports")
    dl1, dl2, dl3, dl4 = st.columns(4)

    with dl1:
        if state.get("model_bytes"):
            st.download_button(
                label="Download model (.pkl)",
                data=state["model_bytes"],
                file_name="best_model.pkl",
                mime="application/octet-stream",
                width="stretch",
            )
    with dl2:
        if state.get("report_md"):
            import re as _re

            def _md_to_html(md: str) -> str:
                lines = md.splitlines()
                html_lines = []
                for line in lines:
                    m = _re.match(r"^(#{1,4})\s+(.*)", line)
                    if m:
                        level = min(len(m.group(1)) + 1, 4)
                        html_lines.append(f"<h{level}>{m.group(2)}</h{level}>")
                        continue
                    if _re.match(r"^---+$", line.strip()):
                        html_lines.append("<hr>")
                        continue
                    if not line.strip():
                        html_lines.append("<p></p>")
                        continue
                    text = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    text = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
                    text = _re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
                    text = _re.sub(r"`(.+?)`", r"<code>\1</code>", text)
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
                "max-width:820px;margin:48px auto;color:#111827;line-height:1.7;background:#fff}"
                "h1,h2,h3,h4{color:#111827;margin-top:1.8rem}"
                "strong{font-weight:600}"
                "code{background:#f3f4f6;padding:2px 6px;border-radius:4px;font-size:.9em}"
                "hr{border:none;border-top:1px solid #e5e7eb;margin:32px 0}"
                "li{margin:4px 0}"
                "table{border-collapse:collapse;width:100%}"
                "th,td{border:1px solid #e5e7eb;padding:8px 12px;text-align:left}"
                "th{background:#f9fafb;font-weight:600}"
                f"</style></head><body>{_md_to_html(state['report_md'])}</body></html>"
            )
            st.download_button(
                label="Download report (.html)",
                data=report_html.encode(),
                file_name="neo_report.html",
                mime="text/html",
                width="stretch",
            )
    with dl3:
        if state.get("inference_zip"):
            st.download_button(
                label="Download inference package (.zip)",
                data=state["inference_zip"],
                file_name="inference_package.zip",
                mime="application/zip",
                help="best_model.pkl + predict.py + serve.py (FastAPI) + README",
                width="stretch",
            )
    with dl4:
        try:
            import mlflow  # noqa: F401
            if st.button("Open MLflow tracker", width="stretch"):
                subprocess.Popen(
                    ["mlflow", "ui", "--port", "5001"],
                    cwd="/Users/khushichoudhary/Neo",
                )
                st.info("Opening [http://localhost:5001](http://localhost:5001) in a new tab.")
        except ImportError:
            pass


render_footer_actions()
