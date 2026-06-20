from datetime import date
from pathlib import Path

import joblib
from fastapi import FastAPI

from src.schemas import ForecastRequest, ForecastResponse


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


@app.get("/health")
def health():
    return {"status": "ok", "model_type": MODEL["type"]}


@app.post("/predict", response_model=ForecastResponse)
def predict(payload: ForecastRequest):
    day_of_year = payload.target_date.timetuple().tm_yday
    tmax = MODEL["lookup"].get((payload.state, day_of_year), MODEL["global_mean"])
    forecast_type, confidence = classify_forecast_type(payload.target_date)
    note = (
        "Baseline prediction from historical seasonal temperature patterns. "
        "Replace this with trained IMD gridded-data models as ingestion is completed."
    )
    return ForecastResponse(
        state=payload.state,
        district=payload.district,
        target_date=payload.target_date,
        tmax_c=round(float(tmax), 1),
        tmin_c=None,
        rainfall_mm=None,
        confidence=confidence,
        forecast_type=forecast_type,
        note=note,
    )
