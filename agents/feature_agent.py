from __future__ import annotations

import pathlib
import pandas as pd

from utils.llm import chat_json

PROMPT_PATH = pathlib.Path(__file__).parent.parent / "prompts" / "feature_agent.txt"


def suggest_features(profile: dict, target_col: str, user_goal: str) -> list[dict]:
    """Ask GPT-4o to suggest new engineered features."""
    import json
    system_prompt = PROMPT_PATH.read_text()
    user_msg = (
        f"Dataset profile:\n{json.dumps(profile, indent=2)}\n\n"
        f"Target column: {target_col}\n"
        f"User goal: {user_goal}"
    )
    result = chat_json(system=system_prompt, user=user_msg)
    return result.get("features", [])


def apply_features(df: pd.DataFrame, features: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    """
    Safely execute each feature expression and add the column to df.
    Returns (enriched_df, applied_features). Only features that succeeded are included.
    """
    df = df.copy()
    applied = []

    for feat in features:
        name = feat.get("name", "").strip()
        expr = feat.get("expression", "").strip()
        rationale = feat.get("rationale", "")

        if not name or not expr:
            continue

        # safety: block any dangerous tokens
        blocked = ["import", "exec", "eval", "open", "os.", "sys.", "__"]
        if any(tok in expr for tok in blocked):
            continue

        try:
            new_col = eval(expr, {"df": df, "pd": pd})  # noqa: S307
            df[name] = new_col
            applied.append({"name": name, "expression": expr, "rationale": rationale})
        except Exception:
            # silently skip features that fail (bad column refs, type errors, etc.)
            continue

    return df, applied


def run(
    df: pd.DataFrame,
    profile: dict,
    target_col: str,
    user_goal: str,
    status_callback=None,
) -> dict:
    if status_callback:
        status_callback("**Feature Engineering Agent**: generating new features.")

    features = suggest_features(profile, target_col, user_goal)
    df_enriched, applied = apply_features(df, features)

    if status_callback:
        if applied:
            names = ", ".join(f"`{f['name']}`" for f in applied)
            status_callback(
                f"**{len(applied)} new features created:** {names}\n\n"
                + "\n".join(f"- **{f['name']}**: {f['rationale']}" for f in applied)
            )
        else:
            status_callback("Feature engineering: no safe features could be generated for this dataset.")

    return {
        "df_engineered": df_enriched,
        "engineered_features": applied,
    }
