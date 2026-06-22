import logging
import time
from pathlib import Path

import joblib
import pandas as pd

from src.features import build_training_frame
from src.imd_ingest import raw_dir_to_dataframe
from src.mlflow_utils import log_training_metrics, setup_mlflow

log = logging.getLogger(__name__)

MODEL_PATH = Path("models/baseline_climatology.joblib")
RAW_DATA_DIR = Path("data/raw/tmax")


def make_demo_training_data() -> pd.DataFrame:
    """Synthetic fallback used only when no real IMD data is present."""
    import math
    dates = pd.date_range("1985-01-01", "2024-12-31", freq="D")
    state_offsets = {
        "Delhi": 1.5,
        "Maharashtra": 0.7,
        "Karnataka": -0.2,
        "Tamil Nadu": 0.8,
        "West Bengal": 0.1,
    }
    rows = []
    for state, offset in state_offsets.items():
        seasonal = [
            30 + offset + 7 * math.sin((day - 80) / 365 * 2 * math.pi)
            for day in dates.dayofyear
        ]
        rows.extend(
            {"date": date, "state": state, "tmax_c": float(temp)}
            for date, temp in zip(dates, seasonal)
        )
    return pd.DataFrame(rows)


def load_training_data(raw_dir: Path = RAW_DATA_DIR) -> tuple[pd.DataFrame, bool]:
    """
    Try to load real IMD data from *raw_dir*.
    Falls back to synthetic data if the directory is empty or missing.

    Returns
    -------
    (frame, is_real) — DataFrame and whether the data is real IMD data.
    """
    if raw_dir.exists() and any(raw_dir.glob("*")):
        log.info("Loading real IMD data from %s", raw_dir)
        try:
            frame = raw_dir_to_dataframe(raw_dir, product="tmax")
            log.info("Loaded %d rows of real IMD data", len(frame))
            return frame, True
        except Exception as exc:
            log.warning("Failed to load real data (%s) — falling back to synthetic", exc)

    log.warning(
        "No IMD data found in %s. Using synthetic data.\n"
        "Run: python scripts/download_imd_data.py --product tmax --years 5",
        raw_dir,
    )
    return make_demo_training_data(), False


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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    setup_mlflow()

    start_time = time.time()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    frame, is_real = load_training_data(RAW_DATA_DIR)
    data_source = "real_imd" if is_real else "synthetic_demo"
    log.info("Data source: %s  |  rows: %d", data_source, len(frame))

    model = train_baseline(frame)
    model["data_source"] = data_source
    joblib.dump(model, MODEL_PATH)

    training_time = time.time() - start_time
    log_training_metrics(frame, model, training_time)

    print(f"\n✅ Saved baseline model to {MODEL_PATH}")
    print(f"   Data source  : {data_source}")
    print(f"   Training time: {training_time:.2f}s")
    print(f"   States       : {model['states']}")
    print(f"   Global mean  : {model['global_mean']}°C")


if __name__ == "__main__":
    main()
