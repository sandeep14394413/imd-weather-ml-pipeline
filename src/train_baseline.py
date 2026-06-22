import logging
import math
import time
from pathlib import Path

import joblib
import pandas as pd

from src.features import build_training_frame
from src.imd_ingest import load_from_parquet, raw_dir_to_dataframe
from src.mlflow_utils import log_training_metrics, setup_mlflow

log = logging.getLogger(__name__)

MODEL_PATH = Path("models/baseline_climatology.joblib")
PARQUET_PATH = Path("data/processed/imd_tmax_daily.parquet")
RAW_DATA_DIR = Path("data/raw/tmax")

# -----------------------------------------------------------------------
# Climatological diurnal temperature range (tmax - tmin) per state (degC)
# Based on IMD normals. Used to derive tmin when only tmax is trained.
# -----------------------------------------------------------------------
DIURNAL_RANGE: dict[str, float] = {
    "Andhra Pradesh": 10.5,
    "Arunachal Pradesh": 9.0,
    "Assam": 8.5,
    "Bihar": 11.0,
    "Chhattisgarh": 11.5,
    "Delhi": 12.5,
    "Goa": 7.0,
    "Gujarat": 12.0,
    "Haryana": 12.5,
    "Himachal Pradesh": 11.0,
    "Jharkhand": 11.0,
    "Karnataka": 9.5,
    "Kerala": 6.5,
    "Madhya Pradesh": 12.0,
    "Maharashtra": 10.5,
    "Manipur": 9.5,
    "Meghalaya": 8.5,
    "Mizoram": 9.0,
    "Nagaland": 9.0,
    "Odisha": 10.5,
    "Punjab": 12.5,
    "Rajasthan": 14.0,
    "Sikkim": 9.5,
    "Tamil Nadu": 8.0,
    "Telangana": 11.0,
    "Tripura": 8.5,
    "Uttar Pradesh": 12.0,
    "Uttarakhand": 11.0,
    "West Bengal": 9.0,
}
DEFAULT_DIURNAL = 10.5

# -----------------------------------------------------------------------
# Monthly mean rainfall (mm/day) per state — IMD climatological normals.
# Index 0 = January, 11 = December.
# -----------------------------------------------------------------------
RAINFALL_NORMALS: dict[str, list[float]] = {
    "Andhra Pradesh":    [1.5, 1.0, 0.8, 1.2, 3.0, 10.5, 12.0, 11.5, 8.5, 5.0, 3.5, 2.0],
    "Arunachal Pradesh": [5.0, 6.0, 9.0,14.0,18.0, 22.0, 25.0, 24.0,18.0,10.0, 5.0, 4.0],
    "Assam":             [5.0, 6.0,10.0,15.0,19.0, 22.0, 24.0, 23.0,17.0, 9.0, 4.0, 3.0],
    "Bihar":             [2.0, 2.5, 2.0, 2.5, 5.0, 12.0, 15.0, 14.0, 9.0, 3.0, 1.0, 1.0],
    "Chhattisgarh":      [1.5, 2.0, 2.5, 2.0, 5.5, 13.0, 16.0, 15.0,10.0, 4.0, 1.0, 1.0],
    "Delhi":             [1.5, 2.0, 2.0, 1.5, 3.0,  8.0, 12.0, 11.0, 6.0, 2.0, 0.5, 1.0],
    "Goa":               [0.5, 0.5, 0.5, 0.5, 5.0, 22.0, 28.0, 26.0,18.0, 8.0, 3.0, 1.0],
    "Gujarat":           [0.5, 0.5, 0.5, 0.5, 1.5,  8.0, 14.0, 12.0, 5.0, 1.5, 0.5, 0.5],
    "Haryana":           [2.0, 2.0, 1.5, 1.0, 2.5,  6.0, 10.0,  9.0, 5.0, 1.5, 0.5, 1.5],
    "Himachal Pradesh":  [5.0, 5.5, 5.0, 4.0, 5.0,  9.0, 14.0, 13.0, 8.0, 3.0, 1.5, 3.5],
    "Jharkhand":         [2.0, 2.5, 3.0, 3.5, 6.0, 13.0, 16.0, 15.0,10.0, 4.0, 1.0, 1.0],
    "Karnataka":         [0.5, 0.5, 1.5, 4.0, 9.0, 13.0, 12.0, 11.5,10.0, 8.0, 4.0, 1.0],
    "Kerala":            [2.0, 2.5, 4.0,10.0,18.0, 25.0, 22.0, 20.0,18.0,15.0,10.0, 4.0],
    "Madhya Pradesh":    [1.0, 1.5, 1.5, 1.5, 4.0, 11.0, 15.0, 14.0, 9.0, 3.0, 0.5, 0.5],
    "Maharashtra":       [0.5, 0.5, 0.5, 1.0, 4.0, 16.0, 22.0, 20.0,12.0, 5.0, 1.5, 0.5],
    "Manipur":           [4.0, 5.0, 8.0,13.0,17.0, 20.0, 22.0, 21.0,16.0, 9.0, 4.0, 3.0],
    "Meghalaya":         [6.0, 7.0,12.0,18.0,22.0, 26.0, 28.0, 27.0,20.0,12.0, 6.0, 4.0],
    "Mizoram":           [4.0, 5.0, 8.0,14.0,18.0, 22.0, 24.0, 23.0,17.0, 9.0, 4.0, 3.0],
    "Nagaland":          [4.0, 5.0, 8.0,13.0,17.0, 20.0, 22.0, 21.0,16.0, 9.0, 4.0, 3.0],
    "Odisha":            [1.5, 2.0, 2.5, 2.5, 5.5, 13.0, 16.0, 15.5,10.5, 5.0, 2.0, 1.0],
    "Punjab":            [2.5, 2.5, 2.0, 1.5, 2.5,  6.0, 10.5,  9.5, 5.0, 1.5, 0.5, 2.0],
    "Rajasthan":         [1.0, 1.0, 0.5, 0.5, 1.5,  5.0, 10.0,  9.0, 4.0, 1.0, 0.3, 0.5],
    "Sikkim":            [6.0, 7.0,10.0,15.0,20.0, 24.0, 26.0, 25.0,18.0,10.0, 5.0, 4.0],
    "Tamil Nadu":        [2.5, 1.5, 1.0, 1.5, 3.5,  5.0,  6.0,  7.0, 8.0,15.0,14.0, 6.0],
    "Telangana":         [1.0, 1.0, 1.0, 1.5, 4.0, 11.0, 13.0, 12.5, 9.0, 5.0, 3.0, 1.0],
    "Tripura":           [4.5, 5.5, 9.0,14.0,18.0, 21.0, 23.0, 22.0,16.0, 9.0, 4.5, 3.0],
    "Uttar Pradesh":     [2.0, 2.0, 1.5, 1.5, 3.0,  9.0, 13.0, 12.0, 7.0, 2.5, 0.5, 1.5],
    "Uttarakhand":       [5.0, 5.5, 5.0, 4.0, 6.0, 10.0, 15.0, 14.0, 9.0, 3.5, 1.5, 4.0],
    "West Bengal":       [3.0, 3.5, 5.0, 6.5,10.0, 15.0, 18.0, 17.0,13.0, 7.0, 3.5, 2.0],
}
DEFAULT_RAINFALL = [2.0, 2.0, 2.5, 3.0, 5.0, 12.0, 15.0, 14.0, 9.0, 4.0, 2.0, 2.0]


def make_demo_training_data() -> pd.DataFrame:
    """Synthetic fallback -- used only when no scraped data is present."""
    dates = pd.date_range("1985-01-01", "2024-12-31", freq="D")
    state_offsets = {
        "Delhi": 1.5, "Maharashtra": 0.7, "Karnataka": -0.2,
        "Tamil Nadu": 0.8, "West Bengal": 0.1,
    }
    rows = []
    for state, offset in state_offsets.items():
        seasonal = [
            30 + offset + 7 * math.sin((day - 80) / 365 * 2 * math.pi)
            for day in dates.dayofyear
        ]
        rows.extend({"date": d, "state": state, "tmax_c": float(t)} for d, t in zip(dates, seasonal))
    return pd.DataFrame(rows)


def load_training_data() -> tuple[pd.DataFrame, str]:
    """
    Priority:
      1. Processed Parquet store  (produced by scrape_imd_data.py)
      2. Raw .GRD / .nc files in data/raw/tmax/
      3. Synthetic demo data (fallback, logs a warning)
    Returns (frame, source_label).
    """
    if PARQUET_PATH.exists():
        try:
            return load_from_parquet(PARQUET_PATH), "parquet_store"
        except Exception as exc:
            log.warning("Parquet load failed (%s) -- trying raw files", exc)

    if RAW_DATA_DIR.exists() and any(RAW_DATA_DIR.glob("*")):
        try:
            return raw_dir_to_dataframe(RAW_DATA_DIR, product="tmax"), "raw_grd_nc"
        except Exception as exc:
            log.warning("Raw file load failed (%s) -- falling back to synthetic", exc)

    log.warning(
        "No real data found. Using synthetic data.\n"
        "Run first: python scripts/scrape_imd_data.py --mode historical"
    )
    return make_demo_training_data(), "synthetic_demo"


def train_baseline(frame: pd.DataFrame) -> dict:
    training = build_training_frame(frame)

    # tmax climatology lookup: (state, day_of_year) -> mean tmax
    tmax_lookup = (
        training.groupby(["state", "day_of_year"])["tmax_c"]
        .mean().round(2).to_dict()
    )

    # tmin lookup: derived via per-state diurnal range
    # tmin = tmax - diurnal_range, smoothed by same day-of-year grouping
    training["tmin_c"] = training.apply(
        lambda r: r["tmax_c"] - DIURNAL_RANGE.get(r["state"], DEFAULT_DIURNAL),
        axis=1,
    )
    tmin_lookup = (
        training.groupby(["state", "day_of_year"])["tmin_c"]
        .mean().round(2).to_dict()
    )

    # rainfall lookup: (state, month) -> climatological daily mean mm
    # Stored as month index 1-12 keyed by (state, month)
    rainfall_lookup: dict[tuple, float] = {}
    for state, monthly in RAINFALL_NORMALS.items():
        for m, val in enumerate(monthly, start=1):
            rainfall_lookup[(state, m)] = val

    return {
        "type": "seasonal_climatology",
        "target": "tmax_c",
        "lookup": tmax_lookup,
        "tmin_lookup": tmin_lookup,
        "rainfall_lookup": rainfall_lookup,
        "diurnal_range": DIURNAL_RANGE,
        "rainfall_normals": RAINFALL_NORMALS,
        "states": sorted(training["state"].unique().tolist()),
        "global_mean": round(float(training["tmax_c"].mean()), 2),
        "global_tmin_mean": round(float(training["tmin_c"].mean()), 2),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    setup_mlflow()
    start_time = time.time()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    frame, source = load_training_data()
    log.info("Data source: %s  |  rows: %d", source, len(frame))

    model = train_baseline(frame)
    model["data_source"] = source
    joblib.dump(model, MODEL_PATH)

    training_time = time.time() - start_time
    log_training_metrics(frame, model, training_time)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"   Data source  : {source}")
    print(f"   Training time: {training_time:.2f}s")
    print(f"   States       : {model['states']}")
    print(f"   Global mean  : {model['global_mean']} C")


if __name__ == "__main__":
    main()
