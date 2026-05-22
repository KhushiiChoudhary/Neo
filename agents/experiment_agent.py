from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import optuna
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    mean_squared_error, r2_score, mean_absolute_error,
)
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from utils.mlflow_tracker import log_run

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

EXPERIMENT_NAME = "automl_agent"


# ── model factories ──────────────────────────────────────────────────────────

def _make_objective(model_name: str, problem_type: str, X_train, y_train, X_test, y_test):
    """Return an Optuna objective function for a given model."""

    def objective(trial: optuna.Trial) -> float:
        if model_name == "LogisticRegression":
            C = trial.suggest_float("C", 1e-3, 10.0, log=True)
            model = LogisticRegression(C=C, max_iter=1000, random_state=42)

        elif model_name == "Ridge":
            alpha = trial.suggest_float("alpha", 1e-3, 100.0, log=True)
            model = Ridge(alpha=alpha, random_state=42)

        elif model_name == "RandomForest":
            n_est = trial.suggest_int("n_estimators", 50, 300)
            max_depth = trial.suggest_int("max_depth", 3, 15)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            if problem_type == "classification":
                model = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42,
                    n_jobs=-1,
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=n_est,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42,
                    n_jobs=-1,
                )

        elif model_name == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": 42,
                "verbosity": 0,
                "n_jobs": -1,
            }
            if problem_type == "classification":
                model = XGBClassifier(**params, eval_metric="logloss", use_label_encoder=False)
            else:
                model = XGBRegressor(**params)

        elif model_name == "LightGBM":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }
            if problem_type == "classification":
                model = LGBMClassifier(**params)
            else:
                model = LGBMRegressor(**params)

        model.fit(X_train, y_train)

        if problem_type == "classification":
            preds = model.predict(X_test)
            try:
                proba = model.predict_proba(X_test)[:, 1]
                classes = np.unique(y_train)
                if len(classes) > 2:
                    score = f1_score(y_test, preds, average="weighted")
                else:
                    score = roc_auc_score(y_test, proba)
            except Exception:
                score = f1_score(y_test, preds, average="weighted")
            return score
        else:
            preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            return -rmse  # Optuna maximises by default

    return objective


# ── tune single model ────────────────────────────────────────────────────────

def tune_model(
    model_name: str,
    problem_type: str,
    X_train,
    y_train,
    X_test,
    y_test,
    n_trials: int = 25,
    status_callback=None,
) -> dict:
    """Run Optuna tuning for one model. Returns metrics + best model."""
    objective = _make_objective(model_name, problem_type, X_train, y_train, X_test, y_test)
    study = optuna.create_study(direction="maximize")

    # send a progress ping every 5 trials to keep the WebSocket alive on cloud
    def _trial_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if status_callback and trial.number % 5 == 0 and trial.number > 0:
            best = round(study.best_value, 4)
            status_callback(f"Tuning **{model_name}**: trial {trial.number}/{n_trials}, best so far: {best}")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, callbacks=[_trial_cb])

    best_params = study.best_params

    # retrain best model on full train set with best params
    if model_name == "LogisticRegression":
        best_model = LogisticRegression(C=best_params["C"], max_iter=1000, random_state=42)
    elif model_name == "Ridge":
        best_model = Ridge(alpha=best_params["alpha"], random_state=42)
    elif model_name == "RandomForest":
        kw = {k: best_params[k] for k in ["n_estimators", "max_depth", "min_samples_split"]}
        if problem_type == "classification":
            best_model = RandomForestClassifier(**kw, random_state=42, n_jobs=-1)
        else:
            best_model = RandomForestRegressor(**kw, random_state=42, n_jobs=-1)
    elif model_name == "XGBoost":
        kw = {k: best_params[k] for k in ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"]}
        if problem_type == "classification":
            best_model = XGBClassifier(**kw, random_state=42, verbosity=0, n_jobs=-1, eval_metric="logloss", use_label_encoder=False)
        else:
            best_model = XGBRegressor(**kw, random_state=42, verbosity=0, n_jobs=-1)
    elif model_name == "LightGBM":
        kw = {k: best_params[k] for k in ["n_estimators", "max_depth", "learning_rate", "num_leaves", "subsample"]}
        if problem_type == "classification":
            best_model = LGBMClassifier(**kw, random_state=42, n_jobs=-1, verbose=-1)
        else:
            best_model = LGBMRegressor(**kw, random_state=42, n_jobs=-1, verbose=-1)

    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    best_model.fit(X_train, y_train)
    preds = best_model.predict(X_test)

    if problem_type == "classification":
        classes = np.unique(y_train)
        metrics = {"accuracy": round(accuracy_score(y_test, preds), 4)}
        try:
            proba = best_model.predict_proba(X_test)
            if len(classes) == 2:
                metrics["auc"] = round(roc_auc_score(y_test, proba[:, 1]), 4)
            else:
                metrics["auc"] = round(roc_auc_score(y_test, proba, multi_class="ovr"), 4)
        except Exception:
            pass
        metrics["f1"] = round(f1_score(y_test, preds, average="weighted"), 4)
        primary_metric = metrics.get("auc", metrics["f1"])
        cv_scoring = "roc_auc" if len(classes) == 2 else "f1_weighted"
    else:
        rmse = round(np.sqrt(mean_squared_error(y_test, preds)), 4)
        metrics = {
            "rmse": rmse,
            "mae": round(mean_absolute_error(y_test, preds), 4),
            "r2": round(r2_score(y_test, preds), 4),
        }
        primary_metric = -rmse
        cv_scoring = "r2"

    # 5-fold cross-validation on the full dataset with best params
    try:
        cv_raw = cross_val_score(best_model, X_all, y_all, cv=5, scoring=cv_scoring, n_jobs=-1)
        cv_mean = round(float(cv_raw.mean()), 4)
        cv_std  = round(float(cv_raw.std()), 4)
    except Exception:
        cv_mean, cv_std = None, None

    log_run(
        experiment_name=EXPERIMENT_NAME,
        model_name=model_name,
        params=best_params,
        metrics=metrics,
        model_obj=best_model,
    )

    return {
        "model_name": model_name,
        "model": best_model,
        "params": best_params,
        "metrics": metrics,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "cv_scoring": cv_scoring,
        "primary_metric": primary_metric,
    }


# ── baseline ─────────────────────────────────────────────────────────────────

def _run_baseline(problem_type: str, X_train, y_train, X_test, y_test) -> dict:
    """Train a naive baseline so tuned models have something to beat."""
    if problem_type == "classification":
        model = DummyClassifier(strategy="most_frequent", random_state=42)
    else:
        model = DummyRegressor(strategy="mean")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    if problem_type == "classification":
        metrics = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "f1": round(f1_score(y_test, preds, average="weighted", zero_division=0), 4),
        }
        primary_metric = metrics["f1"]
    else:
        rmse = round(np.sqrt(mean_squared_error(y_test, preds)), 4)
        metrics = {
            "rmse": rmse,
            "mae": round(mean_absolute_error(y_test, preds), 4),
            "r2": round(r2_score(y_test, preds), 4),
        }
        primary_metric = -rmse

    log_run(EXPERIMENT_NAME, "Baseline", {}, metrics)
    return {"model_name": "Baseline", "model": model, "params": {}, "metrics": metrics, "primary_metric": primary_metric}


# ── run all models ────────────────────────────────────────────────────────────

def run(
    X_train,
    X_test,
    y_train,
    y_test,
    problem_type: str,
    n_trials: int = 25,
    status_callback=None,
) -> dict:
    """Tune all 4 models. Returns best model + results dataframe."""
    if problem_type == "classification":
        model_names = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM"]
    else:
        model_names = ["Ridge", "RandomForest", "XGBoost", "LightGBM"]

    # always run baseline first; tuned models must beat it
    baseline = _run_baseline(problem_type, X_train, y_train, X_test, y_test)
    results = [baseline]
    best_result = None  # baseline is never eligible for "best model"

    for name in model_names:
        if status_callback:
            status_callback(f"Tuning **{name}** ({n_trials} trials).")

        result = tune_model(name, problem_type, X_train, y_train, X_test, y_test, n_trials, status_callback)
        results.append(result)

        if best_result is None or result["primary_metric"] > best_result["primary_metric"]:
            best_result = result

        if status_callback:
            metric_str = " · ".join(f"{k}: {v}" for k, v in result["metrics"].items())
            cv_str = f" · cv: {result['cv_mean']} ±{result['cv_std']}" if result.get("cv_mean") is not None else ""
            status_callback(f"**{name}**: {metric_str}{cv_str}")

    # build comparison dataframe
    rows = []
    for r in results:
        row = {"Model": r["model_name"]}
        row.update(r["metrics"])
        if r.get("cv_mean") is not None:
            row["cv_mean"] = r["cv_mean"]
            row["cv_std"]  = r["cv_std"]
        rows.append(row)
    results_df = pd.DataFrame(rows)

    return {
        "best_model": best_result["model"],
        "best_model_name": best_result["model_name"],
        "best_metrics": best_result["metrics"],
        "results_df": results_df,
    }
