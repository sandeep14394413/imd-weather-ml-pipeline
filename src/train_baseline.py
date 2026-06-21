import time
from pathlib import Path

import joblib
import pandas as pd

from src.features import build_training_frame
from src.mlflow_utils import log_training_metrics, setup_mlflow


MODEL_PATH = Path("models/baseline_climatology.joblib")


def make_demo_training_data() -> pd.DataFrame:
    dates = pd.date_range("1985-01-01", "2024-12-31", freq="D")
    rows = []
    state_offsets = {
        "Delhi": 1.5,
        "Maharashtra": 0.7,
        "Karnataka": -0.2,
        "Tamil Nadu": 0.8,
        "West Bengal": 0.1,
    }
    for state, offset in state_offsets.items():
        seasonal = 30 + offset + 7 * pd.Series(dates.dayofyear).map(
            lambda day: __import__("math").sin((day - 80) / 365 * 2 * __import__("math").pi)
        )
        rows.extend(
            {
                "date": date,
                "state": state,
                "tmax_c": float(temp),
            }
            for date, temp in zip(dates, seasonal)
        )
    return pd.DataFrame(rows)


def train_baseline(frame: pd.DataFrame) -> dict:
    training = build_training_frame(frame)
    lookup = (
        training.groupby(["state", "day_of_year"])["tmax_c"]
        .mean()
        .round(2)
        .to_dict()
    )
    return {
        "type": "seasonal_climatology",
        "target": "tmax_c",
        "lookup": lookup,
        "states": sorted(training["state"].unique().tolist()),
        "global_mean": round(float(training["tmax_c"].mean()), 2),
    }


def main():
    setup_mlflow()
    
    start_time = time.time()
    
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = make_demo_training_data()
    model = train_baseline(frame)
    joblib.dump(model, MODEL_PATH)
    
    training_time = time.time() - start_time
    
    # Log metrics to MLflow
    log_training_metrics(frame, model, training_time)
    
    print(f"Saved baseline model to {MODEL_PATH}")
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Model states: {model['states']}")
    print(f"Global mean temperature: {model['global_mean']}°C")


if __name__ == "__main__":
    main()
