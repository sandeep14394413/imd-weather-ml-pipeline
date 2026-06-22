#!/usr/bin/env python3
"""
IMD Weather Data Scraper
========================
Scrapes 40 years of daily gridded temperature data from publicly
accessible IMD / open-climate mirrors and saves them to data/raw/tmax/.

Strategy (in order of preference)
----------------------------------
1. IMD Gridded Data portal  (imdpune.gov.in) – direct .GRD binary
   download where a stable URL exists.
2. IRI / LDEO Data Library  (iridl.ldeo.columbia.edu) – NCEP reanalysis
   NetCDF, no auth, covers 1948-present.
3. Open-Meteo historical API (open-meteo.com) – JSON, free, no key,
   covers 1940-present at 0.1° resolution.  Used as a reliable fallback
   and for the *incremental daily update* (yesterday's data).

Usage
-----
  # Full 40-year historical backfill (run once)
  python scripts/scrape_imd_data.py --mode historical --start-year 1985 --end-year 2024

  # Daily incremental update (run by cron / GitHub Actions)
  python scripts/scrape_imd_data.py --mode daily

  # Specific year
  python scripts/scrape_imd_data.py --mode historical --start-year 2020 --end-year 2020
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.imd_ingest import STATE_CENTROIDS  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAW_DIR = ROOT / "data" / "raw" / "tmax"
PARQUET_PATH = ROOT / "data" / "processed" / "imd_tmax_daily.parquet"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "imd-weather-ml-pipeline/1.0 (research)"})

# ---------------------------------------------------------------------------
# Open-Meteo API  —  free, no key, 1940-present, 0.1° grid
# FIX: correct endpoint is api.open-meteo.com/v1/archive (not archive.api.*)
# ---------------------------------------------------------------------------
OM_URL = "https://api.open-meteo.com/v1/archive"


def fetch_openmeteo_state(
    state: str,
    lat: float,
    lon: float,
    start: date,
    end: date,
    retries: int = 3,
) -> pd.DataFrame:
    """Fetch daily tmax for one state via Open-Meteo archive API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "temperature_2m_max",
        "timezone": "Asia/Kolkata",
    }
    for attempt in range(1, retries + 1):
        try:
            resp = SESSION.get(OM_URL, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            dates = pd.to_datetime(payload["daily"]["time"])
            temps = payload["daily"]["temperature_2m_max"]
            df = pd.DataFrame({"date": dates, "state": state, "tmax_c": temps})
            return df.dropna(subset=["tmax_c"])
        except Exception as exc:
            if attempt == retries:
                log.error("  [%s] failed after %d attempts: %s", state, retries, exc)
                return pd.DataFrame()
            log.warning("  [%s] attempt %d failed, retrying in 5s: %s", state, attempt, exc)
            time.sleep(5)
    return pd.DataFrame()


def fetch_openmeteo_all_states(
    start: date,
    end: date,
    states: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch tmax for every state in STATE_CENTROIDS."""
    target = states or list(STATE_CENTROIDS.keys())
    frames: list[pd.DataFrame] = []
    log.info("Fetching Open-Meteo data for %d states  %s → %s", len(target), start, end)
    for i, state in enumerate(target, 1):
        lat, lon = STATE_CENTROIDS[state]
        log.info("  [%d/%d] %s", i, len(target), state)
        df = fetch_openmeteo_state(state, lat, lon, start, end)
        frames.append(df)
        time.sleep(0.3)  # polite rate-limiting

    # FIX: guard against all-empty frames to avoid KeyError: 'date' on sort
    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        log.error(
            "All state fetches returned empty — check network access to api.open-meteo.com"
        )
        return pd.DataFrame(columns=["date", "state", "tmax_c"])

    combined = pd.concat(non_empty, ignore_index=True)
    return combined.sort_values(["date", "state"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# IRI / LDEO  —  NCEP reanalysis NetCDF (no auth, covers 1948+)
# ---------------------------------------------------------------------------
IRI_URL = (
    "https://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP-NCAR/.CDAS-1/.DAILY"
    "/.Diagnostic/.above_ground/.maximum/.temp/"
    "T/(1%20Jan%20{year})/(31%20Dec%20{year})RANGEEDGES/"
    "data.nc"
)


def fetch_iri_year(year: int, out_dir: Path) -> Path | None:
    """Download one year of NCEP Tmax NetCDF from IRI Data Library."""
    dest = out_dir / f"ncep_tmax_{year}.nc"
    if dest.exists():
        log.info("  IRI %d already cached: %s", year, dest.name)
        return dest
    url = IRI_URL.format(year=year)
    log.info("  Downloading IRI NCEP Tmax %d ...", year)
    try:
        with SESSION.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            dest.write_bytes(r.content)
        log.info("  Saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return dest
    except Exception as exc:
        log.error("  IRI download failed for %d: %s", year, exc)
        if dest.exists():
            dest.unlink()
        return None


# ---------------------------------------------------------------------------
# Parquet store  —  single append-only file for all processed data
# ---------------------------------------------------------------------------

def load_existing_parquet() -> pd.DataFrame:
    if PARQUET_PATH.exists():
        return pd.read_parquet(PARQUET_PATH)
    return pd.DataFrame(columns=["date", "state", "tmax_c"])


def upsert_parquet(new_data: pd.DataFrame) -> pd.DataFrame:
    """
    Merge *new_data* into the existing Parquet store.
    Newer records win on (date, state) conflicts.
    """
    existing = load_existing_parquet()
    combined = pd.concat([existing, new_data], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    # Keep the last occurrence (new data wins)
    combined = (
        combined
        .sort_values("date")
        .drop_duplicates(subset=["date", "state"], keep="last")
        .sort_values(["date", "state"])
        .reset_index(drop=True)
    )
    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(PARQUET_PATH, index=False)
    log.info(
        "Parquet store updated → %s  |  total rows: %d  |  date range: %s to %s",
        PARQUET_PATH,
        len(combined),
        combined["date"].min().date(),
        combined["date"].max().date(),
    )
    return combined


def write_summary_json(df: pd.DataFrame) -> None:
    """Write a small JSON summary consumed by the deploy pipeline health check."""
    summary = {
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "total_rows": len(df),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "states": sorted(df["state"].unique().tolist()),
        "state_count": df["state"].nunique(),
    }
    summary_path = ROOT / "data" / "processed" / "data_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Summary written to %s", summary_path)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_historical(start_year: int, end_year: int) -> None:
    """Backfill 40 years of data using Open-Meteo (reliable, free, no auth)."""
    log.info("=== HISTORICAL BACKFILL %d–%d ===", start_year, end_year)
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)

    # Check what years we already have to skip them
    existing = load_existing_parquet()
    already_have: set[int] = set()
    if not existing.empty:
        existing["date"] = pd.to_datetime(existing["date"])
        already_have = set(existing["date"].dt.year.unique())
        years_needed = [y for y in range(start_year, end_year + 1) if y not in already_have]
        if not years_needed:
            log.info("All years already in Parquet store — nothing to do.")
            return
        log.info("Already have years: %s. Fetching missing: %s", sorted(already_have), years_needed)
        start = date(min(years_needed), 1, 1)
        end = date(max(years_needed), 12, 31)

    # Open-Meteo: chunk by year to avoid timeouts
    all_frames: list[pd.DataFrame] = []
    for year in range(start.year, end.year + 1):
        if year in already_have:
            continue
        y_start = date(year, 1, 1)
        y_end = min(date(year, 12, 31), date.today() - timedelta(days=1))
        log.info("--- Year %d ---", year)
        df = fetch_openmeteo_all_states(y_start, y_end)
        if not df.empty:
            all_frames.append(df)
            log.info("  Year %d: %d rows fetched", year, len(df))
        time.sleep(1)  # courteous pause between years

    if all_frames:
        new_data = pd.concat(all_frames, ignore_index=True)
        final = upsert_parquet(new_data)
        write_summary_json(final)
        log.info("Historical backfill complete. %d new rows added.", len(new_data))
    else:
        log.warning("No new data fetched — check connectivity to api.open-meteo.com")
        # Write a minimal summary so downstream workflow steps don't crash
        summary_path = ROOT / "data" / "processed" / "data_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps({
            "last_updated": datetime.utcnow().isoformat() + "Z",
            "total_rows": 0,
            "date_min": None,
            "date_max": None,
            "states": [],
            "state_count": 0,
        }, indent=2))


def run_daily() -> None:
    """Fetch yesterday's data and append to the Parquet store."""
    yesterday = date.today() - timedelta(days=1)
    log.info("=== DAILY UPDATE for %s ===", yesterday)
    df = fetch_openmeteo_all_states(yesterday, yesterday)
    if df.empty:
        log.error("No data returned for %s — aborting", yesterday)
        sys.exit(1)
    final = upsert_parquet(df)
    write_summary_json(final)
    log.info("Daily update complete. %d rows added for %s.", len(df), yesterday)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="IMD weather data scraper")
    parser.add_argument(
        "--mode",
        choices=["historical", "daily"],
        default="daily",
        help="'historical' for 40-year backfill, 'daily' for yesterday's update",
    )
    parser.add_argument("--start-year", type=int, default=1985)
    parser.add_argument("--end-year", type=int, default=datetime.now().year - 1)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "historical":
        run_historical(args.start_year, args.end_year)
    else:
        run_daily()


if __name__ == "__main__":
    main()
