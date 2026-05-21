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
            issues.append({"level": "warning", "message": f"`{col}` is {null_pct * 100:.0f}% null. Will be imputed, but may add noise."})
    return issues


def _check_leakage_risk(df: pd.DataFrame, target_col: str, threshold: float = 0.9) -> list[dict]:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.preprocessing import LabelEncoder

    issues = []
    target = df[target_col].dropna()
    df_clean = df.loc[target.index].copy()

    is_numeric_target = pd.api.types.is_numeric_dtype(target)

    # encode target if categorical
    if not is_numeric_target:
        le = LabelEncoder()
        target_enc = le.fit_transform(target.astype(str))
    else:
        target_enc = target.values

    # pearson correlation for numeric features
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        if col == target_col:
            continue
        try:
            corr = abs(df_clean[col].fillna(0).corr(pd.Series(target_enc, index=df_clean.index)))
            if corr > threshold:
                issues.append({"level": "error", "message": f"`{col}` has {corr:.2f} Pearson correlation with the target. Likely data leakage; consider removing it."})
        except Exception:
            continue

    # mutual information for all features (catches categorical leakage)
    try:
        feature_cols = [c for c in df_clean.columns if c != target_col]
        X_mi = df_clean[feature_cols].copy()
        for c in X_mi.columns:
            if not pd.api.types.is_numeric_dtype(X_mi[c]):
                X_mi[c] = LabelEncoder().fit_transform(X_mi[c].astype(str))
        X_mi = X_mi.fillna(0).values

        mi_fn = mutual_info_classif if not is_numeric_target or len(np.unique(target_enc)) <= 20 else mutual_info_regression
        mi_scores = mi_fn(X_mi, target_enc, random_state=42)
        mi_max = mi_scores.max() if mi_scores.max() > 0 else 1.0
        for col, score in zip(feature_cols, mi_scores):
            normalised = score / mi_max
            if normalised > 0.95 and col not in [i["message"].split("`")[1] for i in issues]:
                issues.append({"level": "error", "message": f"`{col}` has very high mutual information with the target (normalised score {normalised:.2f}). Possible leakage."})
    except Exception:
        pass

    return issues


def _check_near_constant(df: pd.DataFrame, target_col: str, threshold: float = 0.99) -> list[dict]:
    issues = []
    for col in df.columns:
        if col == target_col:
            continue
        top_freq = df[col].value_counts(normalize=True).iloc[0] if df[col].nunique() > 0 else 0
        if top_freq >= threshold:
            issues.append({"level": "warning", "message": f"`{col}` is near-constant ({top_freq * 100:.0f}% one value). Likely uninformative."})
    return issues


def _check_target_nulls(df: pd.DataFrame, target_col: str) -> list[dict]:
    null_count = df[target_col].isnull().sum()
    if null_count > 0:
        return [{"level": "error", "message": f"Target column `{target_col}` has {null_count} null values. These rows will be dropped."}]
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
