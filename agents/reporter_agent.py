from __future__ import annotations

import io
import pathlib
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from utils.llm import chat
from utils.inference_generator import build_zip

warnings.filterwarnings("ignore")

PROMPT_PATH = pathlib.Path(__file__).parent.parent / "prompts" / "reporter_agent.txt"

# ── light plot theme (matches Streamlit UI) ───────────────────────────────────
BROWN = "#92400e"
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#faf7f5",
    "axes.edgecolor": "#e7e0da",
    "axes.labelcolor": "#78716c",
    "text.color": "#1c1917",
    "xtick.color": "#a8a29e",
    "ytick.color": "#a8a29e",
    "grid.color": "#e7e0da",
    "legend.facecolor": "#faf7f5",
    "legend.edgecolor": "#e7e0da",
    "legend.labelcolor": "#1c1917",
})


# ── shared plot helper (DRY) ──────────────────────────────────────────────────

def _horizontal_bar_chart(labels: list, values: list, xlabel: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels[::-1], values[::-1], color=BROWN)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ── SHAP ─────────────────────────────────────────────────────────────────────

def generate_shap_plot(model, X_test: np.ndarray, feature_names: list) -> tuple:
    """
    Compute SHAP values. Returns (figure, top_5_feature_names).
    Falls back to feature_importances_ if SHAP fails.
    """
    model_type = type(model).__name__

    try:
        if model_type in ("RandomForestClassifier", "RandomForestRegressor",
                          "XGBClassifier", "XGBRegressor",
                          "LGBMClassifier", "LGBMRegressor"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_test)

        shap_values = explainer.shap_values(X_test)

        if isinstance(shap_values, list):
            shap_matrix = np.abs(shap_values[1] if len(shap_values) > 1 else shap_values[0])
        else:
            shap_matrix = np.abs(shap_values)

        # collapse to 1-D: mean over samples first, then over any remaining axes
        mean_shap = shap_matrix.mean(axis=0)
        while mean_shap.ndim > 1:
            mean_shap = mean_shap.mean(axis=-1)

    except Exception:
        if not hasattr(model, "feature_importances_"):
            return None, feature_names[:5]
        mean_shap = model.feature_importances_

    mean_shap = np.asarray(mean_shap).ravel()
    top_idx = np.argsort(mean_shap)[::-1][:10]
    top_features = [feature_names[int(i)] for i in top_idx]
    top_vals = mean_shap[top_idx].tolist()

    fig = _horizontal_bar_chart(top_features, top_vals, "Mean |SHAP value|", "Feature Importance (SHAP)")
    return fig, top_features[:5]


# ── confidence / calibration plot ────────────────────────────────────────────

def generate_confidence_plot(model, X_test: np.ndarray, y_test: np.ndarray, problem_type: str):
    """
    Classification: predicted probability histogram split by true class.
    Regression: actual vs predicted scatter.
    Returns a matplotlib figure or None if not applicable.
    """
    try:
        if problem_type == "classification" and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1] if model.predict_proba(X_test).shape[1] == 2 else model.predict_proba(X_test).max(axis=1)
            fig, ax = plt.subplots(figsize=(8, 4))
            classes = np.unique(y_test)
            for cls in classes:
                mask = y_test == cls
                ax.hist(proba[mask], bins=20, alpha=0.6, label=f"Class {cls}")
            ax.set_xlabel("Predicted probability")
            ax.set_ylabel("Count")
            ax.set_title("Prediction Confidence Distribution")
            ax.legend()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            return fig

        if problem_type == "regression":
            preds = model.predict(X_test)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(y_test, preds, alpha=0.4, color=BROWN, s=20)
            mn, mx = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
            ax.plot([mn, mx], [mn, mx], "r--", linewidth=1, label="Perfect fit")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            ax.legend()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            return fig

    except Exception:
        pass

    return None


# ── report writing ────────────────────────────────────────────────────────────

def write_report(
    user_goal: str,
    best_model_name: str,
    best_metrics: dict,
    results_df: pd.DataFrame,
    top_features: list,
) -> str:
    """Call GPT-4o to write the plain English report."""
    system_prompt = PROMPT_PATH.read_text()
    metric_str = " · ".join(f"{k}: {v}" for k, v in best_metrics.items())
    user_msg = (
        f"User goal: {user_goal}\n\n"
        f"Winning model: {best_model_name}\n"
        f"Metrics: {metric_str}\n\n"
        f"All model results:\n{results_df.to_string(index=False)}\n\n"
        f"Top 5 features by importance: {', '.join(top_features)}"
    )
    return chat(system=system_prompt, user=user_msg)


# ── serialisation ─────────────────────────────────────────────────────────────

def save_model(model) -> bytes:
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    return buf.read()


# ── main entry ────────────────────────────────────────────────────────────────

def run(
    user_goal: str,
    best_model,
    best_model_name: str,
    best_metrics: dict,
    results_df: pd.DataFrame,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    problem_type: str,
    status_callback=None,
) -> dict:
    if status_callback:
        status_callback("Generating SHAP feature importance plot.")
    shap_fig, top_features = generate_shap_plot(best_model, X_test, feature_names)

    if status_callback:
        status_callback("Generating confidence plot.")
    confidence_fig = generate_confidence_plot(best_model, X_test, y_test, problem_type)

    if status_callback:
        status_callback("Writing report.")
    report_md = write_report(user_goal, best_model_name, best_metrics, results_df, top_features)

    model_bytes = save_model(best_model)
    inference_zip = build_zip(model_bytes, best_model_name, feature_names, problem_type)

    return {
        "shap_fig": shap_fig,
        "confidence_fig": confidence_fig,
        "top_features": top_features,
        "report_md": report_md,
        "model_bytes": model_bytes,
        "inference_zip": inference_zip,
    }
