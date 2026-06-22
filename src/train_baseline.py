import logging
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


def make_demo_training_data() -> pd.DataFrame:
    """Synthetic fallback — used only when no scraped data is present."""
    import math
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
            log.warning("Parquet load failed (%s) — trying raw files", exc)

    if RAW_DATA_DIR.exists() and any(RAW_DATA_DIR.glob("*")):
        try:
            return raw_dir_to_dataframe(RAW_DATA_DIR, product="tmax"), "raw_grd_nc"
        except Exception as exc:
            log.warning("Raw file load failed (%s) — falling back to synthetic", exc)

    log.warning(
        "No real data found. Using synthetic data.\n"
        "Run first: python scripts/scrape_imd_data.py --mode historical"
    )
    return make_demo_training_data(), "synthetic_demo"


def train_baseline(frame: pd.DataFrame) -> dict:
    training = build_training_frame(frame)
    lookup = (
        training.groupby(["state", "day_of_year"])["tmax_c"]
        .mean().round(2).to_dict()
    )
    return {
        "type": "seasonal_climatology",
        "target": "tmax_c",
        "lookup": lookup,
        "states": sorted(training["state"].unique().tolist()),
        "global_mean": round(float(training["tmax_c"].mean()), 2),
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

    print(f"\n✅ Model saved to {MODEL_PATH}")
    print(f"   Data source  : {source}")
    print(f"   Training time: {training_time:.2f}s")
    print(f"   States       : {model['states']}")
    print(f"   Global mean  : {model['global_mean']}°C")


if __name__ == "__main__":
    main()
