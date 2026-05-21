from __future__ import annotations  # keep | syntax working on Python 3.9

from typing import Any, Callable, Optional, TypedDict

import pandas as pd
from langgraph.graph import StateGraph, END


# ── shared state ─────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    # inputs
    user_goal: str
    df_raw: Any
    status_callback: Optional[Callable]

    # confirmed by HITL (skips GPT-4o target ID if set)
    confirmed_target_col: Optional[str]
    confirmed_problem_type: Optional[str]

    # data agent outputs
    profile: dict
    target_col: str
    problem_type: str
    target_reasoning: str
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    feature_names: list

    # feature engineering agent outputs
    df_engineered: Any
    engineered_features: list

    # experiment agent outputs
    best_model: Any
    best_model_name: str
    best_metrics: dict
    results_df: Any

    # reporter agent outputs
    shap_fig: Any
    confidence_fig: Any
    top_features: list
    report_md: str
    model_bytes: bytes
    inference_zip: bytes


# ── node functions ────────────────────────────────────────────────────────────

def data_node(state: AgentState) -> AgentState:
    from agents import data_agent

    cb = state.get("status_callback")
    confirmed_target = state.get("confirmed_target_col")
    confirmed_type = state.get("confirmed_problem_type")

    if confirmed_target:
        if cb:
            cb(
                f"**Target column confirmed:** `{confirmed_target}` "
                f"({confirmed_type}). Preprocessing data."
            )
    else:
        if cb:
            cb("**Data Agent**: profiling dataset and identifying target column.")

    result = data_agent.run(
        df=state["df_raw"],
        user_goal=state["user_goal"],
        confirmed_target_col=confirmed_target,
        confirmed_problem_type=confirmed_type,
    )

    if not confirmed_target and cb:
        cb(
            f"**Target column identified:** `{result['target_col']}` "
            f"({result['problem_type']})\n\n"
            f"_{result['target_reasoning']}_"
        )

    return {**state, **result}


def feature_node(state: AgentState) -> AgentState:
    from agents import feature_agent

    cb = state.get("status_callback")

    result = feature_agent.run(
        df=state["df_raw"],
        profile=state["profile"],
        target_col=state["target_col"],
        user_goal=state["user_goal"],
        status_callback=cb,
    )

    # re-preprocess the enriched dataframe so new features are included
    from agents.data_agent import preprocess
    X_train, X_test, y_train, y_test, feature_names = preprocess(
        result["df_engineered"],
        state["target_col"],
        state["problem_type"],
    )

    return {
        **state,
        **result,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
    }


def experiment_node(state: AgentState) -> AgentState:
    from agents import experiment_agent

    cb = state.get("status_callback")
    if cb:
        cb("**Experiment Agent**: running 4 models with Optuna hyperparameter tuning.")

    result = experiment_agent.run(
        X_train=state["X_train"],
        X_test=state["X_test"],
        y_train=state["y_train"],
        y_test=state["y_test"],
        problem_type=state["problem_type"],
        n_trials=25,
        status_callback=cb,
    )

    if cb:
        metric_str = " · ".join(f"{k}: **{v}**" for k, v in result["best_metrics"].items())
        cb(f"**Best model:** {result['best_model_name']} | {metric_str}")

    return {**state, **result}


def reporter_node(state: AgentState) -> AgentState:
    from agents import reporter_agent

    cb = state.get("status_callback")
    if cb:
        cb("**Reporter Agent**: building SHAP plot and writing summary.")

    result = reporter_agent.run(
        user_goal=state["user_goal"],
        best_model=state["best_model"],
        best_model_name=state["best_model_name"],
        best_metrics=state["best_metrics"],
        results_df=state["results_df"],
        X_test=state["X_test"],
        y_test=state["y_test"],
        feature_names=state["feature_names"],
        problem_type=state["problem_type"],
        status_callback=cb,
    )

    return {**state, **result}


# ── graph construction ────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("data", data_node)
    graph.add_node("features", feature_node)
    graph.add_node("experiment", experiment_node)
    graph.add_node("reporter", reporter_node)

    graph.set_entry_point("data")
    graph.add_edge("data", "features")
    graph.add_edge("features", "experiment")
    graph.add_edge("experiment", "reporter")
    graph.add_edge("reporter", END)

    return graph.compile()


_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph()
    return _compiled_graph


# ── public entry point ────────────────────────────────────────────────────────

def run_pipeline(
    df: pd.DataFrame,
    user_goal: str,
    status_callback: Optional[Callable] = None,
    confirmed_target_col: Optional[str] = None,
    confirmed_problem_type: Optional[str] = None,
) -> dict:
    """
    Execute the full Data → Features → Experiment → Reporter pipeline.
    Pass confirmed_target_col / confirmed_problem_type from HITL to skip GPT-4o target ID.
    """
    graph = _get_graph()

    initial_state: AgentState = {
        "user_goal": user_goal,
        "df_raw": df,
        "status_callback": status_callback,
        "confirmed_target_col": confirmed_target_col,
        "confirmed_problem_type": confirmed_problem_type,
    }

    final_state = graph.invoke(initial_state)
    return final_state
