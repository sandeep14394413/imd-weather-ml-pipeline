import time
from datetime import date
from pathlib import Path

import joblib
from fastapi import FastAPI

from src.schemas import ForecastRequest, ForecastResponse
from src.train_baseline import DEFAULT_DIURNAL, DEFAULT_RAINFALL, DIURNAL_RANGE, RAINFALL_NORMALS

MODEL_PATH = Path("models/baseline_climatology.joblib")

app = FastAPI(title="India Weather Model API")


def load_model() -> dict:
    if not MODEL_PATH.exists():
        from src.train_baseline import main
        main()
    return joblib.load(MODEL_PATH)


MODEL = load_model()


def classify_forecast_type(target_date: date) -> tuple[str, str]:
    days_ahead = (target_date - date.today()).days
    if days_ahead <= 7:
        return "short-range", "medium"
    if days_ahead <= 45:
        return "extended-range", "low-medium"
    return "climate-outlook", "low"


def predict_tmin(state: str, day_of_year: int, tmax: float) -> float:
    """Return tmin from model lookup, falling back to tmax minus diurnal range."""
    tmin_lookup = MODEL.get("tmin_lookup", {})
    if (state, day_of_year) in tmin_lookup:
        return tmin_lookup[(state, day_of_year)]
    # fallback: tmax - climatological diurnal range
    return round(tmax - DIURNAL_RANGE.get(state, DEFAULT_DIURNAL), 1)


def predict_rainfall(state: str, target_date: date) -> float:
    """Return climatological daily mean rainfall (mm) for the state+month."""
    rainfall_lookup = MODEL.get("rainfall_lookup", {})
    month = target_date.month
    key = (state, month)
    if key in rainfall_lookup:
        return round(rainfall_lookup[key], 1)
    # fallback: global monthly normal
    monthly = RAINFALL_NORMALS.get(state, DEFAULT_RAINFALL)
    return round(monthly[month - 1], 1)


@app.get("/health")
def health():
    return {"status": "ok", "model_type": MODEL["type"]}


@app.get("/metrics")
def get_metrics():
    """Get MLflow model metrics and tracking info."""
    import mlflow
    try:
        client = mlflow.tracking.MlflowClient()
        runs = mlflow.search_runs(max_results=1, order_by=["start_time DESC"])
        if runs.empty:
            return {"status": "no_runs_found"}
        latest_run = runs.iloc[0]
        run_id = latest_run.run_id
        run = client.get_run(run_id)
        return {
            "status": "ok",
            "latest_run_id": run_id,
            "parameters": run.data.params,
            "metrics": run.data.metrics,
            "tags": run.data.tags,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/predict", response_model=ForecastResponse)
def predict(payload: ForecastRequest):
    start_time = time.time()

    day_of_year = payload.target_date.timetuple().tm_yday
    month = payload.target_date.month

    # tmax from seasonal climatology lookup
    tmax = MODEL["lookup"].get((payload.state, day_of_year), MODEL["global_mean"])
    tmax = round(float(tmax), 1)

    # tmin from lookup or diurnal range fallback
    tmin = predict_tmin(payload.state, day_of_year, tmax)

    # rainfall from monthly climatological normal
    rainfall = predict_rainfall(payload.state, payload.target_date)

    forecast_type, confidence = classify_forecast_type(payload.target_date)
    note = (
        f"Climatology baseline: tmax from {MODEL.get('data_source','seasonal lookup')}, "
        "tmin derived from IMD diurnal range normals, "
        "rainfall from IMD monthly normals. "
        "Replace with ML model once sufficient data is ingested."
    )

    inference_time = time.time() - start_time

    response = ForecastResponse(
        state=payload.state,
        district=payload.district,
        target_date=payload.target_date,
        tmax_c=tmax,
        tmin_c=tmin,
        rainfall_mm=rainfall,
        confidence=confidence,
        forecast_type=forecast_type,
        note=note,
    )

    import mlflow
    with mlflow.start_run(run_name=f"prediction_{payload.state}"):
        mlflow.log_metric("inference_time_ms", inference_time * 1000)
        mlflow.log_param("forecast_state", payload.state)
        mlflow.log_param("forecast_type", forecast_type)

    return response
