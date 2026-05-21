from __future__ import annotations

import json
import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from utils.llm import chat_json

PROMPT_PATH = pathlib.Path(__file__).parent.parent / "prompts" / "data_agent.txt"


# ── profiling ────────────────────────────────────────────────────────────────

def profile_dataframe(df: pd.DataFrame) -> dict:
    """Return a lightweight profile of each column."""
    profile = {}
    for col in df.columns:
        series = df[col]
        null_pct = round(series.isnull().mean() * 100, 1)
        cardinality = int(series.nunique())
        sample_vals = series.dropna().unique()[:5].tolist()
        # make sample values JSON-serialisable
        sample_vals = [v.item() if hasattr(v, "item") else v for v in sample_vals]
        profile[col] = {
            "dtype": str(series.dtype),
            "null_pct": null_pct,
            "cardinality": cardinality,
            "sample_values": sample_vals,
        }
    return profile


# ── target identification ────────────────────────────────────────────────────

def identify_target(profile: dict, user_goal: str) -> dict:
    """Ask GPT-4o to identify the target column and problem type."""
    system_prompt = PROMPT_PATH.read_text()
    user_msg = (
        f"Dataset profile:\n{json.dumps(profile, indent=2)}\n\n"
        f"User goal: {user_goal}"
    )
    result = chat_json(system=system_prompt, user=user_msg)
    return result  # {"target_col": ..., "problem_type": ..., "reasoning": ...}


# ── preprocessing ────────────────────────────────────────────────────────────

def preprocess(
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Clean and split the dataframe.
    Returns (X_train, X_test, y_train, y_test, feature_names).
    """
    df = df.copy()

    # separate target
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # encode target for classification
    if problem_type == "classification":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), name=target_col)

    # split numeric vs categorical features (treat booleans as numeric)
    X = X.copy()
    for col in X.select_dtypes(include=["bool"]).columns:
        X[col] = X[col].astype(int)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # impute + scale numerics
    if num_cols:
        num_imputer = SimpleImputer(strategy="median")
        X[num_cols] = num_imputer.fit_transform(X[num_cols])
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

    # impute + encode categoricals
    if cat_cols:
        # normalise to object dtype: preserve NaN, cast everything else to str
        for col in cat_cols:
            X[col] = X[col].where(pd.isna(X[col]), X[col].astype(str))
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
        for col in cat_cols:
            le_col = LabelEncoder()
            X[col] = le_col.fit_transform(X[col].astype(str))

    feature_names = X.columns.tolist()
    X_arr = X.values
    y_arr = y.values

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, feature_names


# ── main entry ───────────────────────────────────────────────────────────────

def run(
    df: pd.DataFrame,
    user_goal: str,
    confirmed_target_col: str | None = None,
    confirmed_problem_type: str | None = None,
) -> dict:
    """
    Full data agent pipeline. Returns enriched state dict.
    If confirmed_target_col / confirmed_problem_type are provided (from HITL),
    the GPT-4o identification call is skipped.
    """
    profile = profile_dataframe(df)

    if confirmed_target_col and confirmed_problem_type:
        target_col = confirmed_target_col
        problem_type = confirmed_problem_type
        reasoning = "Confirmed by user."
    else:
        identification = identify_target(profile, user_goal)
        target_col = identification["target_col"]
        problem_type = identification["problem_type"]
        reasoning = identification["reasoning"]

    X_train, X_test, y_train, y_test, feature_names = preprocess(
        df, target_col, problem_type
    )

    return {
        "profile": profile,
        "target_col": target_col,
        "problem_type": problem_type,
        "target_reasoning": reasoning,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
    }
