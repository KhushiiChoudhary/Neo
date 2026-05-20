import mlflow
import mlflow.sklearn


def log_run(
    experiment_name: str,
    model_name: str,
    params: dict,
    metrics: dict,
    model_obj=None,
) -> str:
    """Log a single model run to MLflow. Returns the run_id."""
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if model_obj is not None:
            mlflow.sklearn.log_model(model_obj, artifact_path="model")
        return run.info.run_id
