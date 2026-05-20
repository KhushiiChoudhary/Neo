"""
MLflow experiment tracking. Gracefully skipped if mlflow is not installed
(e.g. on Streamlit Cloud where the UI is not accessible anyway).
"""
from __future__ import annotations

try:
    import mlflow
    import mlflow.sklearn
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


def log_run(
    experiment_name: str,
    model_name: str,
    params: dict,
    metrics: dict,
    model_obj=None,
) -> str | None:
    """Log a single model run to MLflow. Returns run_id or None if MLflow unavailable."""
    if not _MLFLOW_AVAILABLE:
        return None

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if model_obj is not None:
            mlflow.sklearn.log_model(model_obj, artifact_path="model")
        return run.info.run_id
