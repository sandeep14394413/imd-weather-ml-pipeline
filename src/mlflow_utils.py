"""MLflow utilities for model tracking and performance monitoring."""

import time
from typing import Any, Dict

import mlflow
import mlflow.sklearn
import psutil


def setup_mlflow(experiment_name: str = "IMD-Weather-Baseline") -> None:
    """Initialize MLflow experiment."""
    mlflow.set_experiment(experiment_name)


def log_training_metrics(
    frame: "pd.DataFrame",
    model: Dict[str, Any],
    training_time: float,
) -> None:
    """Log training metrics to MLflow."""
    import pandas as pd
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_type", model["type"])
        mlflow.log_param("target_variable", model["target"])
        mlflow.log_param("num_states", len(model["states"]))
        mlflow.log_param("training_data_rows", len(frame))
        mlflow.log_param("date_range", f"{frame['date'].min()} to {frame['date'].max()}")
        
        # Log metrics
        mlflow.log_metric("training_time_seconds", training_time)
        mlflow.log_metric("global_mean_tmax", model["global_mean"])
        mlflow.log_metric("num_lookup_entries", len(model["lookup"]))
        
        # Log system metrics
        memory_info = psutil.virtual_memory()
        mlflow.log_metric("memory_used_mb", memory_info.used / 1024 / 1024)
        mlflow.log_metric("memory_percent", memory_info.percent)
        mlflow.log_metric("cpu_percent", psutil.cpu_percent(interval=0.1))
        
        # Log dataset statistics
        mlflow.log_metric("mean_tmax", float(frame["tmax_c"].mean()))
        mlflow.log_metric("std_tmax", float(frame["tmax_c"].std()))
        mlflow.log_metric("min_tmax", float(frame["tmax_c"].min()))
        mlflow.log_metric("max_tmax", float(frame["tmax_c"].max()))
        
        # Log tags for better organization
        mlflow.set_tag("model_stage", "baseline")
        mlflow.set_tag("data_source", "IMD")
        mlflow.set_tag("feature_engineering", "seasonal_climatology")


def log_prediction_performance(
    predictions: list,
    actuals: list,
    inference_time: float,
) -> None:
    """Log prediction performance metrics to MLflow."""
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    with mlflow.start_run():
        if len(predictions) > 0 and len(actuals) > 0:
            mae = mean_absolute_error(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            r2 = r2_score(actuals, predictions)
            
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2_score", r2)
        
        # Inference performance
        avg_inference_time = inference_time / len(predictions) if predictions else 0
        mlflow.log_metric("total_inference_time_seconds", inference_time)
        mlflow.log_metric("avg_inference_time_ms", avg_inference_time * 1000)
        mlflow.log_metric("predictions_count", len(predictions))
        
        mlflow.set_tag("evaluation_stage", "test")


def log_model_artifact(model_path: str) -> None:
    """Log model artifact to MLflow."""
    mlflow.log_artifact(model_path, artifact_path="models")
