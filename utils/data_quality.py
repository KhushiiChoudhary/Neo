from __future__ import annotations

import numpy as np
import pandas as pd

# ── individual checks ─────────────────────────────────────────────────────────

def _check_class_imbalance(df: pd.DataFrame, target_col: str) -> list[dict]:
    series = df[target_col].dropna()
    counts = series.value_counts(normalize=True)
    minority_pct = counts.min() * 100
    if minority_pct < 15:
        return [{"level": "error", "message": f"Severe class imbalance: minority class is only {minority_pct:.1f}% of data. Consider oversampling or class weights."}]
    if minority_pct < 25:
        return [{"level": "warning", "message": f"Mild class imbalance: minority class is {minority_pct:.1f}%. AUC is a better metric than accuracy here."}]
    return []


def _check_high_nulls(df: pd.DataFrame, target_col: str, threshold: float = 0.3) -> list[dict]:
    issues = []
    for col in df.columns:
        if col == target_col:
            continue
        null_pct = df[col].isnull().mean()
        if null_pct > threshold:
            issues.append({"level": "warning", "message": f"`{col}` is {null_pct * 100:.0f}% null — will be imputed, but may add noise."})
    return issues


def _check_leakage_risk(df: pd.DataFrame, target_col: str, threshold: float = 0.9) -> list[dict]:
    issues = []
    target = df[target_col]
    if not pd.api.types.is_numeric_dtype(target):
        return []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == target_col:
            continue
        try:
            corr = abs(df[col].corr(target))
            if corr > threshold:
                issues.append({"level": "error", "message": f"`{col}` has {corr:.2f} correlation with target — possible data leakage. Consider removing it."})
        except Exception:
            continue
    return issues


def _check_near_constant(df: pd.DataFrame, target_col: str, threshold: float = 0.99) -> list[dict]:
    issues = []
    for col in df.columns:
        if col == target_col:
            continue
        top_freq = df[col].value_counts(normalize=True).iloc[0] if df[col].nunique() > 0 else 0
        if top_freq >= threshold:
            issues.append({"level": "warning", "message": f"`{col}` is near-constant ({top_freq * 100:.0f}% one value) — likely uninformative."})
    return issues


def _check_target_nulls(df: pd.DataFrame, target_col: str) -> list[dict]:
    null_count = df[target_col].isnull().sum()
    if null_count > 0:
        return [{"level": "error", "message": f"Target column `{target_col}` has {null_count} null values — these rows will be dropped."}]
    return []


# ── public entry ──────────────────────────────────────────────────────────────

def run_checks(df: pd.DataFrame, target_col: str, problem_type: str) -> list[dict]:
    """
    Run all data quality checks. Returns a list of {level, message} dicts.
    level is one of: "error" | "warning" | "info"
    """
    checks = [
        _check_target_nulls,
        _check_high_nulls,
        _check_near_constant,
        _check_leakage_risk,
    ]
    issues: list[dict] = []
    for check in checks:
        issues.extend(check(df, target_col))

    if problem_type == "classification":
        issues.extend(_check_class_imbalance(df, target_col))

    return issues
